"""Tests for the agent's deterministic parts.

Nothing here calls Groq. The tool loop itself needs a live model and an API key,
so it is exercised by the evaluation harness instead; what is unit-tested is the
logic that decides what the user is shown -- citation validation, deduplication,
trace summaries and payload unwrapping. Those are pure functions, and they are
where the silent failures live.
"""

from __future__ import annotations

import pytest

from app.agent import (
    Citation,
    ToolCall,
    _dedupe,
    _summarise,
    _unwrap,
    find_unsupported_citations,
)


def cite(doc_id: str, number: str, similarity: float = 0.8) -> Citation:
    return Citation(
        doc_id=doc_id,
        section_number=number,
        section_heading="Heading",
        similarity=similarity,
        snippet="text",
    )


# --- citation validation ---------------------------------------------------


def test_citation_backed_by_retrieval_is_supported():
    retrieved = [cite("PTO-01", "4")]
    assert find_unsupported_citations("See PTO-01 §4.", retrieved) == []


def test_citation_never_retrieved_is_flagged():
    retrieved = [cite("PTO-01", "4")]
    assert find_unsupported_citations("See PTO-01 §9.", retrieved) == ["PTO-01 §9"]


def test_subsection_of_a_retrieved_section_is_supported():
    """§2.4 really is inside the §2 text the model was handed."""
    retrieved = [cite("PTO-01", "2")]
    assert find_unsupported_citations("See PTO-01 §2.4.", retrieved) == []


def test_unicode_hyphen_is_still_detected():
    """The regression that made this detector useless.

    Models write "PTO-01" with a non-breaking hyphen (U+2011). An ASCII-only
    pattern matched nothing, so every answer looked perfectly cited.
    """
    retrieved = [cite("PTO-01", "2")]
    answer = "See PTO‑01 §6 for sick leave."
    assert find_unsupported_citations(answer, retrieved) == ["PTO-01 §6"]


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "–", "—"])
def test_every_dash_variant_is_detected(dash):
    answer = f"See REM{dash}01 §3."
    assert find_unsupported_citations(answer, []) == ["REM-01 §3"]


def test_unsupported_citations_are_reported_once():
    answer = "PTO-01 §9 says one thing, and PTO-01 §9 says it again."
    assert find_unsupported_citations(answer, []) == ["PTO-01 §9"]


def test_no_citations_at_all_is_not_an_error():
    assert find_unsupported_citations("I don't know.", []) == []


# --- deduplication ---------------------------------------------------------


def test_dedupe_keeps_the_best_scoring_occurrence():
    deduped = _dedupe([cite("PTO-01", "4", 0.6), cite("PTO-01", "4", 0.9)])
    assert len(deduped) == 1
    assert deduped[0].similarity == 0.9


def test_dedupe_sorts_by_similarity_descending():
    deduped = _dedupe([cite("A-01", "1", 0.5), cite("B-02", "2", 0.9)])
    assert [c.label for c in deduped] == ["B-02 §2", "A-01 §1"]


# --- trace summaries -------------------------------------------------------


def test_failed_tool_summarises_as_failure():
    summary = _summarise("check_pto_balance", {"ok": False, "error": "employee_not_found"})
    assert "failed" in summary and "employee_not_found" in summary


def test_out_of_corpus_search_is_marked_in_the_trace():
    payload = {
        "ok": True,
        "results": [{"citation": "BEN-01 §5"}],
        "likely_out_of_corpus": True,
    }
    assert "below in-corpus range" in _summarise("search_policy_documents", payload)


def test_tool_call_trace_omits_the_raw_result():
    """The trace shown to the user must not dump the whole payload."""
    call = ToolCall(
        name="check_pto_balance",
        arguments={"employee_id": "EMP-101"},
        ok=True,
        duration_ms=5,
        summary="PTO 14.0d",
        result={"secret": "a very large payload"},
    )
    assert "result" not in call.to_dict()
    assert call.to_dict()["summary"] == "PTO 14.0d"


# --- payload unwrapping ----------------------------------------------------


class _Block:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Result:
    def __init__(self, structured=None, text=None):
        self.structuredContent = structured
        self.content = [_Block(text)] if text else []


def test_unwrap_prefers_structured_content():
    assert _unwrap(_Result(structured={"ok": True, "n": 1})) == {"ok": True, "n": 1}


def test_unwrap_unpacks_the_mcpserver_result_wrapper():
    """MCPServer wraps a non-dict return as {"result": ...}."""
    assert _unwrap(_Result(structured={"result": {"ok": True}})) == {"ok": True}


def test_unwrap_falls_back_to_json_in_text_blocks():
    assert _unwrap(_Result(text='{"ok": true}')) == {"ok": True}


def test_unwrap_survives_non_json_text():
    """A tool that returns prose must not crash the turn."""
    assert _unwrap(_Result(text="plain words"))["text"] == "plain words"


# --- rate-limit handling ---------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Please try again in 12.8s.", 12.8),
        ("Please try again in 1m30s.", 90.0),
        # The case that caused a single question to block for 2.26 hours: an
        # hours-only pattern under-read this, and an uncapped sleep honoured it.
        ("Please try again in 2h15m51.6s.", 8151.6),
        ("no duration here", None),
    ],
)
def test_retry_after_parses_every_duration_shape(text, expected):
    from app.agent import _retry_after

    assert _retry_after(Exception(text)) == expected


def test_daily_quota_wait_exceeds_the_cap():
    """A wait this long is an exhausted quota, not a throttle to sleep through."""
    from app.agent import MAX_RETRY_WAIT_SECONDS, _retry_after

    wait = _retry_after(Exception("Please try again in 2h15m51.6s."))
    assert wait > MAX_RETRY_WAIT_SECONDS


# --- provider portability --------------------------------------------------


def test_nullable_union_is_collapsed():
    """`str | None` compiles to anyOf[string, null], which some providers reject.

    OpenRouter's grammar engine refused the whole request with "more than one
    JSON reading of the same emitted value". The parameter is optional via
    `required` either way, so the null branch carries nothing.
    """
    from app.agent import simplify_schema

    schema = {
        "type": "object",
        "properties": {
            "doc_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [],
    }
    assert simplify_schema(schema)["properties"]["doc_id"] == {"type": "string"}


def test_nullable_type_list_is_collapsed():
    from app.agent import simplify_schema

    schema = {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
    assert simplify_schema(schema)["properties"]["x"]["type"] == "string"


def test_a_real_union_is_left_alone():
    """Only NULLABLE unions collapse; a genuine string-or-number must survive."""
    from app.agent import simplify_schema

    branches = [{"type": "string"}, {"type": "number"}]
    schema = {"type": "object", "properties": {"x": {"anyOf": branches}}}
    assert simplify_schema(schema)["properties"]["x"]["anyOf"] == branches


def test_simplify_preserves_descriptions_and_required():
    from app.agent import simplify_schema

    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what to search for"},
            "doc_id": {"anyOf": [{"type": "string"}, {"type": "null"}],
                       "description": "optional filter"},
        },
        "required": ["query"],
    }
    out = simplify_schema(schema)
    assert out["required"] == ["query"]
    assert out["properties"]["query"]["description"] == "what to search for"
    assert out["properties"]["doc_id"]["description"] == "optional filter"


def test_known_providers_are_registered():
    from app.agent import PROVIDERS, resolve_provider

    assert {"groq", "openrouter", "openai"} <= set(PROVIDERS)
    assert resolve_provider("openrouter").base_url.startswith("https://openrouter.ai")


def test_unknown_provider_without_base_url_is_an_error(monkeypatch):
    from app.agent import AgentError, resolve_provider

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(AgentError, match="Unknown provider"):
        resolve_provider("not-a-provider")


def test_any_openai_compatible_endpoint_works_via_env(monkeypatch):
    from app.agent import resolve_provider

    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    provider = resolve_provider("custom")
    assert provider.base_url == "https://example.test/v1"
    assert provider.default_model == "some-model"


def test_employee_id_is_not_read_as_a_citation():
    """"approver EMP-004" parsed as "EMP-00 §4" before the section marker
    was required, so every answer naming an employee looked ungrounded."""
    retrieved = [cite("PTO-01", "2")]
    answer = "Request REQ-2026-0141 for 2 days, approver EMP-004."
    assert find_unsupported_citations(answer, retrieved) == []


def test_unknown_employee_id_is_not_read_as_a_citation():
    assert find_unsupported_citations("No employee with ID EMP-999 exists.", []) == []


def test_section_word_form_is_still_detected():
    assert find_unsupported_citations("See PTO-01 section 9.", []) == ["PTO-01 §9"]


def test_bare_doc_id_without_a_section_is_not_a_citation():
    """"REM-01 covers this" names a document, it does not cite a section."""
    assert find_unsupported_citations("REM-01 covers travel.", []) == []
