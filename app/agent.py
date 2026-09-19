"""The HR agent: a Groq tool-calling loop driving a real MCP client session.

Architecture
------------
    user question
        -> Groq chat completion, with the MCP server's tools as the schema
        -> model emits tool_calls
        -> each call is dispatched OVER MCP (stdio subprocess), not in-process
        -> results appended as `tool` messages
        -> loop until the model answers or MAX_ITERATIONS is hit

The tool schemas are never written here. They are read from the live MCP
session at connect time and translated to Groq's function-calling format, so
adding a tool to `mcp_server/server.py` is the only edit needed -- the agent
cannot drift out of sync with the server because it has no independent copy.

On tracing
----------
The trace records *operations*: which tool was called, with what arguments,
what came back, how long it took. It deliberately does NOT record the model's
reasoning tokens. A chain-of-thought transcript is not an audit log -- it is
unfaithful to the actual computation, and presenting it to an HR user as an
explanation of a leave decision would be misleading. What is defensible is the
evidence trail: these sections were retrieved, this balance was read, this rule
decided it.

On guardrails
-------------
Three live here; one lives in the server. The server owns the write gate
(`create_mock_hr_ticket` refuses without `confirmed=true`) because a guardrail
that exists only in a system prompt is one jailbreak away from being gone. The
prompt owns grounding, citation and out-of-corpus refusal, because those shape
generation rather than gating an action.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

#: Groq model. Chosen by what the account can actually serve: the Llama 3.3 70B
#: that most Groq tool-calling examples use is not on this account's model list,
#: and a hardcoded unavailable model is a 404 at the worst possible moment.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

#: Hard stop on the tool loop. Without it a model that keeps re-searching for a
#: fact the corpus does not contain will spend the whole rate limit discovering
#: that. Six is enough for the deepest real workflow (profile -> balance ->
#: two policy lookups -> draft -> ticket) with headroom.
MAX_ITERATIONS = 6

#: How many times to wait out a provider rate limit before giving up. The free
#: tier throttles on tokens per minute, so a throttle is an ordinary event to be
#: waited out, not an error to be reported.
RATE_LIMIT_RETRIES = 5

#: Longest we will wait out a throttle. Above this it is not a throttle, it is
#: an exhausted quota, and waiting is the wrong response.
MAX_RETRY_WAIT_SECONDS = 120.0

SERVER_MODULE = "mcp_server.server"

SYSTEM_PROMPT = """\
You are the Northwind Systems HR assistant. You help employees understand HR \
policy and their own entitlements.

GROUNDING
- Every statement about what the rules are must come from a policy passage you \
retrieved this turn. Never answer a policy question from memory.
- Cite the document id and section for each rule you rely on, like (PTO-01 §4). \
Cite inline, next to the claim it supports.
- If search returns likely_out_of_corpus, say plainly that Northwind has no \
policy covering it and suggest who to ask. Do not infer a plausible-sounding \
policy. A confident invented benefit is the worst failure you can produce.

EMPLOYEE DATA
- Facts about a specific person -- balances, hours, tenure, location, benefits \
-- come only from the lookup tools. Policy documents contain worked examples \
with illustrative numbers; those are never anyone's real balance.
- If you do not have an employee id, ask for it. Never guess one.
- A leave balance comes from check_pto_balance and nowhere else. Accrual figures \
on the profile are lifetime totals, not spendable balances; never quote one as \
"your balance". If a balance matters to the answer, call the balance tool.

REASONING
- A request can fail on any one of several conditions. Check each that applies \
-- eligibility, notice period, probation, balance, approver -- and name the \
specific one that decides the outcome. Having enough days does not make leave \
approvable.
- Separate what policy REQUIRES from what you RECOMMEND. Label recommendations \
as such.
- When a rule depends on a threshold, state the threshold and the person's \
actual value side by side.

ACTIONS
- draft_hr_email composes text only; it sends nothing. Say so.
- create_mock_hr_ticket writes data. Before calling it, show the user exactly \
what would be created and ask them to confirm. Only after they agree in their \
own words may you call it with confirmed=true.

STYLE
- Lead with the answer. Be direct about a no.
- Keep it short enough to read on a phone.
"""


class AgentError(RuntimeError):
    """Configuration or transport failure, as opposed to a bad answer."""


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
# Every provider here speaks the OpenAI chat-completions dialect, which is what
# makes swapping one for another a configuration change rather than a rewrite:
# the tool schemas, the `tool_calls` reply and the `tool` result messages are
# identical across all of them. A provider with a genuinely different shape
# (Anthropic's or Gemini's native APIs) would need an adapter, not an entry.
#
# This exists because the binding constraint on this project turned out to be
# the provider's free-tier quota, not the model. Being able to point the same
# agent at a second provider for evaluation runs is worth more than any
# individual model choice.


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    default_model: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.key_env)


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        # Free tier and supports tool calling -- OpenRouter lists many free
        # models that do NOT, and one of those fails only at the first tool
        # call, not at connect time.
        default_model="qwen/qwen3.8-27b:free",
        headers={
            "HTTP-Referer": "https://github.com/Mhmoud-Zidan/hr-agentic-rag",
            "X-Title": "Northwind HR Agent",
        },
    ),
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
    ),
}

#: Which provider to use. Overridden per run with --provider in the evaluation
#: harness, so a rate-limited provider does not block an evaluation.
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")


def resolve_provider(name: str | None = None) -> Provider:
    """Look up a provider, or build one from LLM_BASE_URL for anything else."""
    name = (name or DEFAULT_PROVIDER).lower()
    if name in PROVIDERS:
        return PROVIDERS[name]

    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        known = ", ".join(sorted(PROVIDERS))
        raise AgentError(
            f"Unknown provider {name!r}. Known: {known}. For any other "
            f"OpenAI-compatible endpoint, set LLM_BASE_URL and LLM_API_KEY."
        )
    return Provider(
        name=name,
        base_url=base_url,
        key_env="LLM_API_KEY",
        default_model=os.environ.get("LLM_MODEL", ""),
    )




# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One dispatched tool call, as it will be shown to the user."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: int
    summary: str
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "summary": self.summary,
        }


@dataclass
class Citation:
    doc_id: str
    section_number: str
    section_heading: str
    similarity: float
    snippet: str

    @property
    def label(self) -> str:
        return f"{self.doc_id} §{self.section_number}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "doc_id": self.doc_id,
            "section_number": self.section_number,
            "section_heading": self.section_heading,
            "similarity": self.similarity,
            "snippet": self.snippet,
        }


@dataclass
class AgentResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    latency_ms: int = 0
    model: str = DEFAULT_MODEL
    out_of_corpus: bool = False
    awaiting_confirmation: bool = False
    unsupported_citations: list[str] = field(default_factory=list)
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "trace": [t.to_dict() for t in self.tool_calls],
            "iterations": self.iterations,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "out_of_corpus": self.out_of_corpus,
            "awaiting_confirmation": self.awaiting_confirmation,
            "unsupported_citations": self.unsupported_citations,
            "tokens_used": self.tokens_used,
        }


# ---------------------------------------------------------------------------
# summarising tool results for the trace
# ---------------------------------------------------------------------------

_MAX_SNIPPET = 320


def _summarise(name: str, payload: dict[str, Any]) -> str:
    """A one-line, human-readable account of what a tool returned.

    The raw payload still goes to the model; this is only for the trace the user
    reads, where a wall of JSON would hide rather than show what happened.
    """
    if not isinstance(payload, dict):
        return str(payload)[:120]
    if payload.get("ok") is False:
        return f"failed: {payload.get('error', 'unknown error')}"

    if name == "search_policy_documents":
        hits = payload.get("results", [])
        labels = ", ".join(h.get("citation", "?") for h in hits[:4])
        flag = " (below in-corpus range)" if payload.get("likely_out_of_corpus") else ""
        return f"{len(hits)} passage(s): {labels}{flag}"
    if name == "get_policy_section":
        return f"fetched {payload.get('citation', '?')} ({payload.get('chunk_count', 0)} chunk(s))"
    if name == "lookup_employee_profile":
        d = payload.get("derived", {})
        return (
            f"{payload.get('name')} — {payload.get('employment_type')}, "
            f"{payload.get('scheduled_hours_per_week')}h/wk, "
            f"tenure {d.get('tenure_days')}d, probation={d.get('is_in_probation')}"
        )
    if name == "check_pto_balance":
        return (
            f"PTO {payload.get('pto_balance_days')}d, sick "
            f"{payload.get('sick_balance_days')}d, pending "
            f"{payload.get('pending_request_days')}d"
        )
    if name == "lookup_benefits_status":
        return (
            f"eligible={payload.get('benefits_eligible')}, "
            f"medical={payload.get('medical_plan_tier')}"
        )
    if name == "check_policy_compliance":
        return f"{len(payload.get('policy_evidence', []))} policy passage(s) + profile"
    if name == "draft_hr_email":
        return "draft composed (not sent)"
    if name == "create_mock_hr_ticket":
        ticket = payload.get("ticket", {})
        return f"created {ticket.get('ticket_id', '?')}"
    return "ok"


def _collect_citations(name: str, payload: dict[str, Any]) -> list[Citation]:
    """Pull citable passages out of whichever tool produced them."""
    if not isinstance(payload, dict) or payload.get("ok") is False:
        return []

    if name == "search_policy_documents":
        rows = payload.get("results", [])
    elif name == "check_policy_compliance":
        rows = payload.get("policy_evidence", [])
    elif name == "get_policy_section":
        rows = [
            {
                "doc_id": payload.get("doc_id"),
                "section_number": payload.get("section_number"),
                "section_heading": payload.get("section_heading"),
                "similarity": 1.0,  # exact fetch, not a ranked hit
                "text": payload.get("text", ""),
            }
        ]
    else:
        return []

    citations = []
    for row in rows:
        text = (row.get("text") or "").strip().replace("\n", " ")
        citations.append(
            Citation(
                doc_id=str(row.get("doc_id") or "?"),
                section_number=str(row.get("section_number") or "?"),
                section_heading=str(row.get("section_heading") or ""),
                similarity=float(row.get("similarity") or 0.0),
                snippet=text[:_MAX_SNIPPET] + ("…" if len(text) > _MAX_SNIPPET else ""),
            )
        )
    return citations


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------


class HRAgent:
    """Owns one MCP session and one Groq client for its lifetime.

    Use as an async context manager. The MCP subprocess is spawned once and
    reused: spawning per request would pay the import cost (and, on the first
    search, the embedding model load) on every question.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = MAX_ITERATIONS,
        provider: str | None = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.provider_name = provider or DEFAULT_PROVIDER
        self.provider: Provider | None = None
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._tools: list[dict[str, Any]] = []
        self._client: Any = None

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "HRAgent":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from openai import AsyncOpenAI

        provider = resolve_provider(self.provider_name)
        if not provider.api_key:
            raise AgentError(
                f"No API key for provider '{provider.name}'. Set "
                f"{provider.key_env} in .env at the repository root."
            )
        self.provider = provider
        if self.model == DEFAULT_MODEL and provider.default_model:
            # The caller did not ask for a specific model, and Groq's model id
            # means nothing to OpenRouter. Use the provider's own default.
            self.model = provider.default_model

        self._client = AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            default_headers=provider.headers or None,
        )
        self._stack = AsyncExitStack()

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", SERVER_MODULE],
            cwd=str(REPO_ROOT),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        self._tools = await self._load_tool_schemas()

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._session = None

    async def _load_tool_schemas(self) -> list[dict[str, Any]]:
        """Translate the live MCP tool list into function-calling format."""
        listing = await self._session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": simplify_schema(tool.input_schema),
                },
            }
            for tool in listing.tools
        ]

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._tools]

    async def health(self) -> dict[str, Any]:
        """Connectivity, for GET /health. Cheap: no model call, no embedding."""
        status: dict[str, Any] = {"mcp_connected": self._session is not None}
        try:
            listing = await self._session.list_tools()
            status["tool_count"] = len(listing.tools)
            status["tools"] = [t.name for t in listing.tools]
        except Exception as exc:
            status["mcp_connected"] = False
            status["error"] = str(exc)
        status["model"] = self.model
        status["provider"] = self.provider.name if self.provider else self.provider_name
        status["api_key_present"] = bool(self.provider and self.provider.api_key)
        return status

    # -- dispatch ----------------------------------------------------------

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[dict, ToolCall]:
        started = time.perf_counter()
        try:
            raw = await self._session.call_tool(name, arguments)
            payload = _unwrap(raw)
            ok = bool(payload.get("ok", True)) if isinstance(payload, dict) else True
        except Exception as exc:
            # A transport failure must not kill the turn: hand the model a
            # readable error so it can apologise or try a different tool.
            payload = {"ok": False, "error": "tool_dispatch_failed", "detail": str(exc)}
            ok = False

        duration_ms = int((time.perf_counter() - started) * 1000)
        call = ToolCall(
            name=name,
            arguments=arguments,
            ok=ok,
            duration_ms=duration_ms,
            summary=_summarise(name, payload),
            result=payload if isinstance(payload, dict) else {},
        )
        return payload, call

    async def _complete(self, messages: list[dict[str, Any]]) -> Any:
        """One chat completion, retrying when the provider throttles us.

        Groq's free tier allows 8,000 tokens per MINUTE, and a single
        tool-augmented turn can approach that on its own. Without a retry a
        throttle surfaces as an exception, and in an evaluation run that becomes
        a recorded FAILURE -- so the metrics would be measuring the rate limit
        rather than the agent. The provider tells us how long to wait; honour it
        rather than guessing.
        """
        from openai import RateLimitError

        delay = 5.0
        for attempt in range(1, RATE_LIMIT_RETRIES + 1):
            try:
                return await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tools,
                    tool_choice="auto",
                    temperature=0.1,  # policy answers should be reproducible
                    # 900, not 1400. The model will fill whatever budget it is
                    # given with longer tables; 900 is ample for a policy
                    # decision with citations, and an oversized budget is not
                    # merely wasteful here -- it pushes the request over the
                    # per-minute limit, and the wait then looks like slow
                    # generation.
                    max_tokens=900,
                )
            except RateLimitError as exc:
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                wait = _retry_after(exc) or delay
                # A per-minute throttle is worth waiting out. A daily-quota
                # exhaustion is not: Groq answers that with a retry-after of
                # hours, and an unbounded sleep here blocked a single question
                # for 2.26 hours before succeeding. Fail loudly instead, so the
                # caller learns the budget is gone rather than appearing to hang.
                if wait > MAX_RETRY_WAIT_SECONDS:
                    raise AgentError(
                        f"Provider quota exhausted; it asks to retry in "
                        f"{wait / 60:.0f} minutes. This is the daily token "
                        f"limit, not a transient throttle -- wait for the reset "
                        f"or switch GROQ_MODEL."
                    ) from exc
                await asyncio.sleep(wait)
                delay = min(delay * 2, MAX_RETRY_WAIT_SECONDS)
        raise AgentError("unreachable")

    # -- the loop ----------------------------------------------------------

    async def ask(
        self, question: str, history: Sequence[dict[str, str]] | None = None
    ) -> AgentResponse:
        """Answer one question, calling tools as needed."""
        if self._session is None:
            raise AgentError("Agent is not connected; use `async with HRAgent()`.")

        started = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(dict(m) for m in history)
        messages.append({"role": "user", "content": question})

        trace: list[ToolCall] = []
        citations: list[Citation] = []
        out_of_corpus = False
        awaiting_confirmation = False
        answer = ""
        iterations = 0
        tokens_used = 0

        for iterations in range(1, self.max_iterations + 1):
            completion = await self._complete(messages)
            usage = getattr(completion, "usage", None)
            if usage is not None:
                tokens_used += int(getattr(usage, "total_tokens", 0) or 0)
            message = completion.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                answer = message.content or ""
                break

            # Echo the assistant turn back verbatim: Groq rejects a `tool`
            # message whose tool_call_id has no matching assistant turn.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                payload, call = await self._dispatch(tc.function.name, arguments)
                trace.append(call)

                if isinstance(payload, dict):
                    if payload.get("likely_out_of_corpus"):
                        out_of_corpus = True
                    if payload.get("error") == "confirmation_required":
                        awaiting_confirmation = True
                    citations.extend(_collect_citations(tc.function.name, payload))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        # Tool results are re-sent on every subsequent
                        # iteration, so an oversized payload is paid for
                        # repeatedly. 6000 chars still carries five full policy
                        # passages.
                        "content": json.dumps(payload, ensure_ascii=False)[:6000],
                    }
                )
        else:
            # Loop exhausted without a final answer: say so rather than
            # presenting a half-finished chain as if it were complete.
            answer = (
                "I wasn't able to finish working through that within my tool-call "
                "limit. Here is what I gathered — please ask HR directly, or "
                "narrow the question and I'll try again."
            )

        deduped = _dedupe(citations)
        return AgentResponse(
            answer=answer,
            citations=deduped,
            tool_calls=trace,
            iterations=iterations,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=self.model,
            out_of_corpus=out_of_corpus,
            awaiting_confirmation=awaiting_confirmation,
            unsupported_citations=find_unsupported_citations(answer, deduped),
            tokens_used=tokens_used,
        )


# NOTE the hyphen class. Models routinely render "PTO-01" with a non-breaking
# hyphen (U+2011) or an en dash rather than ASCII "-", and an ASCII-only
# pattern then silently matches nothing. A detector that never fires is
# indistinguishable from a clean result, which is how this was nearly missed.
_HYPHENS = "-‐‑‒–—−"
_CITATION_RE = re.compile(
    rf"\b([A-Z]{{3}})[{_HYPHENS}](\d{{2}})\s*[§]?\s*(\d+(?:\.\d+)*)"
)


def find_unsupported_citations(
    answer: str, citations: Sequence[Citation]
) -> list[str]:
    """Citations in the prose that no retrieved passage backs.

    Observed in the very first runs: asked about Spain, the model answered
    correctly but cited REM-01 §6 and §7.1 when neither had been retrieved. It
    had read a cross-reference inside the Worked Examples section and repeated
    the pointer as if it were a source. The ANSWER was right and the CITATION
    was fabricated, which is the most dangerous combination -- it reads as
    well-sourced and survives a casual check.

    A subsection counts as supported when its parent section was retrieved:
    §2.4 is genuinely inside the §2 text the model was given.
    """
    retrieved = {(c.doc_id, c.section_number) for c in citations}
    parents = {(doc, num.split(".")[0]) for doc, num in retrieved}

    unsupported: list[str] = []
    for prefix, doc_number, number in _CITATION_RE.findall(answer or ""):
        doc_id = f"{prefix}-{doc_number}"
        if (doc_id, number) in retrieved:
            continue
        if (doc_id, number.split(".")[0]) in parents:
            continue
        label = f"{doc_id} §{number}"
        if label not in unsupported:
            unsupported.append(label)
    return unsupported


def simplify_schema(schema: Any) -> Any:
    """Collapse nullable unions so every provider can compile the schema.

    A Python parameter typed `str | None` becomes
    `{"anyOf": [{"type": "string"}, {"type": "null"}]}`. Groq accepts that.
    At least one OpenRouter backend rejects the whole request with

        grammar rejected: parameter "doc_id": more than one JSON reading
        of the same emitted value

    because a constrained-decoding grammar cannot tell which branch an emitted
    token belongs to. The parameter is optional either way -- it is absent from
    `required` -- so the null branch carries no information the model needs, and
    dropping it loses nothing while making the tool portable.

    This is done HERE, at the boundary where MCP schemas are translated for the
    LLM, rather than by weakening the type annotations in the server. The server
    is the contract; this is one client's dialect.
    """
    if isinstance(schema, list):
        return [simplify_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    for union_key in ("anyOf", "oneOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list):
            concrete = [b for b in branches if _schema_type(b) != "null"]
            if len(concrete) == 1 and len(concrete) < len(branches):
                merged = {**{k: v for k, v in schema.items() if k != union_key},
                          **concrete[0]}
                return simplify_schema(merged)

    # "type": ["string", "null"] is the other spelling of the same thing.
    kind = schema.get("type")
    if isinstance(kind, list):
        concrete = [k for k in kind if k != "null"]
        schema = {**schema, "type": concrete[0] if len(concrete) == 1 else concrete}

    return {key: simplify_schema(value) for key, value in schema.items()}


def _schema_type(branch: Any) -> Any:
    return branch.get("type") if isinstance(branch, dict) else None


def _retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, from the header or the message."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens"):
        raw = header.get(key)
        if not raw:
            continue
        match = re.fullmatch(
            r"(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s?", str(raw).strip()
        )
        if match:
            hours = float(match.group(1) or 0)
            minutes = float(match.group(2) or 0)
            return hours * 3600 + minutes * 60 + float(match.group(3))
    # "Please try again in 2h15m51.6s" -- the hours component is why an
    # hours-only pattern silently under-read the wait.
    match = re.search(
        r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(exc)
    )
    if not match:
        return None
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    return hours * 3600 + minutes * 60 + float(match.group(3))


def _unwrap(result: Any) -> Any:
    """Get the payload out of an MCP CallToolResult, whatever shape it took."""
    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if isinstance(structured, dict):
        # MCPServer wraps a non-dict return as {"result": ...}.
        return structured.get("result", structured)

    text = "".join(
        getattr(block, "text", "")
        for block in getattr(result, "content", [])
        if getattr(block, "type", "") == "text"
    )
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"ok": True, "text": text}


def _dedupe(citations: list[Citation]) -> list[Citation]:
    """One entry per section, keeping the best-scoring occurrence."""
    best: dict[str, Citation] = {}
    for citation in citations:
        existing = best.get(citation.label)
        if existing is None or citation.similarity > existing.similarity:
            best[citation.label] = citation
    return sorted(best.values(), key=lambda c: c.similarity, reverse=True)


async def ask_once(
    question: str, model: str = DEFAULT_MODEL, provider: str | None = None
) -> AgentResponse:
    """Convenience for scripts and tests: connect, ask one question, close."""
    async with HRAgent(model=model, provider=provider) as agent:
        return await agent.ask(question)


def main() -> None:
    """CLI: python -m app.agent [--provider NAME] [--model ID] "question"."""
    parser = argparse.ArgumentParser(prog="python -m app.agent")
    parser.add_argument("--provider", default=None, help=f"one of {sorted(PROVIDERS)}")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()

    response = asyncio.run(
        ask_once(" ".join(args.question), model=args.model, provider=args.provider)
    )
    print(f"[{response.model}]\n")
    print(response.answer)
    if response.citations:
        print("\nSources:")
        for citation in response.citations:
            print(f"  {citation.label}  {citation.section_heading} "
                  f"({citation.similarity:.3f})")
    if response.unsupported_citations:
        print("\nUNSUPPORTED CITATIONS (cited in prose, never retrieved):")
        for label in response.unsupported_citations:
            print(f"  {label}")
    print(f"\nTrace ({response.iterations} iteration(s), {response.latency_ms} ms):")
    for call in response.tool_calls:
        args = json.dumps(call.arguments, ensure_ascii=False)
        flag = "" if call.ok else "  [FAILED]"
        print(f"  {call.name}{args} -> {call.summary} "
              f"[{call.duration_ms} ms]{flag}")


if __name__ == "__main__":
    main()
