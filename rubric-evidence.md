# Rubric evidence

Each requirement, what was built for it, and where to verify it. Every number
here is reproducible from the repository.

**Live:** https://northwind-hr-agent-fyp7.onrender.com
**Repo:** https://github.com/Mhmoud-Zidan/hr-agentic-rag

---

## 1. Deployed agentic HR system: policy RAG with correct, cited, grounded responses

**What exists.** A deployed FastAPI application where an agent answers HR policy
questions from a ten-document corpus and acts on employee records through
MCP-exposed tools. Every response carries citations, the passages behind them,
and the full tool-call trace.

**Evidence**

| Claim | Where |
|---|---|
| Deployed and reachable | `GET /health` → 200 with `mcp_connected: true` |
| Answers are cited | Expand **Sources** on any answer — document ID, section, similarity, snippet |
| Answers are grounded | 96.7% groundedness over 30 evaluation questions |
| Correctness | 93.3% overall pass rate, `design-and-evaluation.md` §3 |

**What to say.** The system does not merely cite — it *validates* its own
citations. Every response is checked for sections that appear in the prose but
were never retrieved, and the page warns when it finds them. This was built
after observing the agent cite `REM-01 §6` having only read a cross-reference to
it: a correct answer with a fabricated citation, which is the combination that
survives casual review. Most RAG systems display citations with uniform
confidence; this one distinguishes retrieved evidence from inferred references.

---

## 2. MCP integration: discovery, correct calls, traces, graceful errors

**What exists.** `mcp_server/server.py` exposes **eight** tools over stdio using
the official MCP 2.x SDK, in a separate process. The agent connects as a real
`ClientSession` and **reads the tool schemas from the live session** — it holds
no tool definitions of its own, so server and agent cannot drift apart.

| Category | Tools |
|---|---|
| Retrieval | `search_policy_documents`, `get_policy_section` |
| People data | `lookup_employee_profile`, `check_pto_balance`, `lookup_benefits_status` |
| Reasoning | `check_policy_compliance` |
| Actions | `draft_hr_email`, `create_mock_hr_ticket` |

**Discovery is proven independently of the app.** CI runs a standalone script
that opens a stdio session, lists tools, and asserts at least five
(`.github/workflows/ci.yml`, step "MCP tool discovery"). `tests/test_mcp_server.py`
spawns the server as a subprocess and speaks MCP to it — it does not import the
functions, because importing would test the Python and not the protocol.

**Traces.** Every `/chat` response includes a structured trace: tool name,
arguments, a readable result summary, and duration. Visible in the UI under
**Tool calls**.

**Graceful error handling — the design decision worth explaining.** Tool errors
are *returned as data, not raised*:

```json
{"ok": false, "error": "employee_not_found", "employee_id": "EMP-999",
 "hint": "Ask the user to confirm their employee id rather than guessing.",
 "known_employee_ids": ["EMP-001", "..."]}
```

Raised, this would reach the model as a protocol error — which reads as "the
tool is broken" rather than "that person does not exist", and that distinction
decides whether the agent asks a clarifying question or gives up. Verified by
`test_unknown_employee_fails_gracefully`, and by evaluation question Q19.

---

## 3. At least two end-to-end agentic tasks in the deployed demo

**Task 1 — international work request.** *"I'm EMP-101. Can I work from Spain for
six weeks?"*

Multi-step: searches the policy corpus, then looks up who EMP-101 is, because
the answer depends on entity and location. Produces a **grounded refusal** on two
independent grounds — Spain is not an approved country, and six weeks exceeds the
30-day cap — each cited, with alternatives offered. The difficulty is not
retrieving REM-01; it is applying both rules rather than approving on whichever
was retrieved first.

**Task 2 — leave request escalating into a write.** Three turns:

1. *"I'm EMP-102 and I'd like to take 3 days off next week."* → **declined**, and
   the reason matters: the balance is 3.3 days, which is *sufficient*. It fails
   on probation (day 60 of 91) and on notice (a 3-day request needs 10 business
   days). An assistant checking only the balance would have approved it.
2. *"Then raise an HR ticket about unpaid leave."* → shows exactly what would be
   created and **asks**. Nothing is written.
3. *"Yes, go ahead."* → ticket created; `create_mock_hr_ticket` appears in the
   trace and `tickets.json` gains a record.

Both tasks combine multi-step reasoning, RAG retrieval, and structured mock-data
tool use, and both are reproducible on the deployed URL.

---

## 4. RAG ingestion, indexing, retrieval, citations, guardrails

**Ingestion.** Ten documents, ~52 pages, in **two formats** — eight Markdown and
two HTML — parsed to structurally identical `Section` objects so the chunker
behaves the same across formats (`app/parsers.py`, 19 tests).

**Chunking.** Heading-aware: H2 primary, H3 fallback, greedy packing of
consecutive leaves, sliding window only for a leaf that exceeds the cap alone.
Tokens counted with **the embedding model's own tokenizer**, never whitespace —
`all-MiniLM-L6-v2` ships truncating at 128 tokens, which makes every cap
assertion pass vacuously while real chunks run long. A test pins truncation off.

**Index.** 140 chunks over 94 sections; median 378 tokens, p95 464, max 479
against a 480 cap; none over cap, none under the 80-token floor. Deterministic
chunk IDs, so a rebuild is reproducible.

**Embeddings.** `BAAI/bge-small-en-v1.5`. BGE is asymmetric — queries need an
instruction prefix that passages do not. Rather than assume, this was measured:
`cos(query_embed(q), embed(q)) = 1.000000` proves fastembed does **not** apply
it. The finding and the numbers are recorded in the module docstring, and the
prefix is applied manually in `embed_query`.

**Retrieval surface.** Chroma with cosine space; `similarity = 1 - distance` is
computed at the boundary so raw distance never escapes the store.

**Two corpus quality gates run in CI**

- `scripts/validate_corpus.py` — structure, metadata, scope
- `scripts/check_no_leak.py` — fails the build if any employee-data value appears
  in policy prose

The second enforces a binding authoring rule: **no worked example may use a value
from `data/mock_data/`**. A policy example built on a real balance plants the gold
answer inside a retrievable chunk; the agent then pattern-matches a paragraph
instead of reasoning, and the tools-on/off ablation collapses because the answer
is in the corpus either way. **This rule is what makes §8's ablation meaningful.**

**Guardrails, and where each lives**

| Guardrail | Location | Why there |
|---|---|---|
| Write confirmation | **In the tool** | `create_mock_hr_ticket` refuses without `confirmed=true`, so it holds even if the model skips asking |
| Out-of-corpus refusal | In retrieval | Search flags results below the in-corpus similarity range (real questions 0.68–0.79; "stock options" 0.63) |
| Citation validation | Post-answer | Flags sections cited but never retrieved |
| Frozen clock | In the data | Tenure and probation compute from a fixed date, so answers stay reproducible |

Placement is the point: a guardrail in the prompt is advice; a guardrail in the
tool is enforcement.

---

## 5. Architecture: clear separation of concerns

```
static/ + app/web.py     web app: POST /chat, GET /health
app/agent.py             agent orchestration: tool-calling loop, trace, guardrails
  ⇅ MCP over stdio       real ClientSession, separate process
mcp_server/server.py     MCP server: eight tool definitions
app/vector_store.py      RAG index: Chroma, cosine, Protocol-based
app/mock_data.py         mock HR data: typed loaders, frozen clock
app/agent.py PROVIDERS   LLM provider: Groq / OpenRouter / DeepSeek, swappable
```

**Separations worth pointing at**

- **The agent imports no HR code.** It speaks MCP. Tools can be re-implemented or
  backed by a real HRIS without the agent changing.
- **The vector store is behind a `Protocol`**, so Chroma is replaceable without
  touching retrieval callers.
- **`app/config.py` is the single source** for the embedding model and index path.
  This was added after finding `vector_store.py` defaulted to `storage/chroma`
  while `build_index.py` wrote to `chroma_db` — a reader on the wrong path opens
  an *empty collection without erroring*, which looks like a bad retriever rather
  than a bad path.
- **The LLM provider is configuration, not code.** All three speak the OpenAI
  chat-completions dialect, so switching is a base URL. Selectable per request
  from the UI, and only providers the server holds a key for are accepted.
- **`mcp_server/`, not `mcp/`** — a top-level `mcp/` directory would shadow the
  installed SDK under `pythonpath = ["."]`.

---

## 6. Free-tier deployment, environment variables, cold start documented

**Live on Render**, Docker runtime, free plan, `render.yaml` committed so the
service is reproducible from the repo rather than from settings someone clicked.

`deployed.md` documents:

- **Environment variables** — `GROQ_API_KEY` (required), `GROQ_MODEL`,
  `MOCK_TODAY`, and the optional `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY`
- **Cold start ~20–25 s** — free instances sleep after ~15 min idle; warm requests
  are 1.8 s for a lookup and 2–3 s for a policy question on DeepSeek
- **Provider rate limits** — Groq's free tier is 8k tokens/minute and 200k/day;
  when exhausted `/chat` returns **429**, not 500, because a provider quota is
  not a fault in this application
- **Known limits** — single serialised MCP session; ticket writes are ephemeral

**Two deployment decisions worth explaining**

- **`python:3.12-slim`, not local 3.14.** `chromadb`, `onnxruntime` and
  `fastembed` ship compiled wheels that lag new CPython releases; a missing wheel
  would first surface during a deploy.
- **The index is built at image build time.** A cold start is a model *load*, not
  a *download*, and a broken corpus fails the build instead of production.
  Verified with `docker run --network none`: retrieval still returns all 140
  chunks.

---

## 7. CI/CD on push/PR with build/start checks and an MCP test

`.github/workflows/ci.yml`, three jobs, green on every push:

**`test`** — install → import check → `validate_corpus.py` → `check_no_leak.py` →
`pytest` (140 tests) → **standalone MCP tool-discovery check**

**`docker`** — builds the deployable image, runs the container, and polls until
it answers `/health`. A build that succeeds but cannot boot is not a passing
build.

**`deploy`** — `needs: [test, docker]`, so deployment only happens on green.

The MCP discovery step is deliberately separate from `pytest` so a failure is
legible in the job log rather than buried in test output. The whole suite passes
**with no API key and no pre-built index**, so CI works on a clean checkout.

---

## 8. Evaluation: groundedness, citations, tool selection, workflow, safety, latency

**30 questions across seven behaviours** — straightforward, multi-document,
tool-requiring, ambiguous, out-of-scope, escalation, action-safety. Gold answers
derive from the corpus specification and the employee records, **not** from the
policy prose, so a document that drifts from the spec fails the evaluation rather
than agreeing with itself.

| Metric | Result |
|---|---|
| Overall pass rate | **93.3%** (28/30) |
| Groundedness | 96.7% |
| Citation accuracy | 94.4% |
| Tool-selection accuracy | 100% |
| Workflow completion | 100% |
| Escalation accuracy | 100% |
| Action safety | 100% |
| Latency p50 / p95 | 2.9 s / 4.4 s |
| Cost per full run | $0.013 |

**Ablation — tools on vs off.** Same model, same prompts, tool list emptied:

| | Tools ON | Tools OFF | Δ |
|---|---|---|---|
| Overall | 86.7% | 26.7% | **−60.0** |
| Workflow completion | 96.7% | 26.7% | −70.0 |
| Escalation accuracy | 100% | 50% | −50.0 |
| Action safety | 100% | 100% | **0** |
| Latency p50 | 2,803 ms | 1,067 ms | −1,736 ms |

Three readings: the retrieval layer is doing the work; **without tools the system
is faster and wrong**, so optimising on latency alone would damage it; and action
safety is unchanged because with no tools there is nothing to write — the
guardrail was never carried by the prompt.

**What makes this evaluation credible rather than decorative**

- Each metric is stated with **what it is worth**. Groundedness is a *lower bound*
  on hallucination — it detects fabricated citations but not uncited false claims.
- The **noise floor is measured and published**: two identical configurations
  scored 93.3% and 86.7%, so differences under ~7 points are not read as real.
- **The evaluation's own bugs are documented** (`design-and-evaluation.md` §6):
  two runs contaminated by rate limits scored as wrong answers; a citation checker
  that parsed `EMP-004` as "EMP-00 §4"; an earlier version of that checker that
  matched *nothing* because the model writes a non-breaking hyphen; a
  forbidden-phrase check that fired inside a negation. Every change made after
  seeing results is recorded, because that is exactly how scores get fixed.
- **The two remaining failures are written up**, not hidden.

---

## 9. Design documentation and demo

| Document | Contents |
|---|---|
| `README.md` | Architecture, quick start, corpus, retrieval decisions, tools, guardrails |
| `design-and-evaluation.md` | Design decisions with rationale, full results, ablation, failure analysis, evaluation-infrastructure bugs, limitations |
| `ai-tooling.md` | How AI tooling was used — and **where it was wrong**, with each error and how it was caught |
| `deployed.md` | Live URL, verified results, configuration, cold start, known limits |
| `demo-deck.pptx` | 12 slides with speaker notes |
| `demo-script.md` | Timed runbook, recovery steps, what not to do |

The demo presents features, architecture, deployment, MCP calls and evaluation,
and closes on **limitations rather than the score** — groundedness is a lower
bound, scoring is string matching, n=30 with a ~7-point noise floor, and the
corpus is synthetic and therefore easier than production.

---

## Reproducing every number here

```bash
pytest -q                                          # 140 passed
python scripts/validate_corpus.py                  # 10/10 documents
python scripts/check_no_leak.py                    # PASS
python scripts/build_index.py --debug-chunks       # 140 chunks, none over cap
python evaluation/run_eval.py --provider deepseek  # 93.3%
python evaluation/run_eval.py --provider deepseek --ablation tools
curl https://northwind-hr-agent-fyp7.onrender.com/health
```
