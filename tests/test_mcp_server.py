"""End-to-end tests for the MCP server, over a real stdio transport.

These deliberately spawn the server as a subprocess and speak MCP to it rather
than importing the tool functions. Importing would test the Python functions;
it would not test that the tools are *discoverable*, that their schemas are
valid, or that their return values survive JSON serialisation -- and those are
exactly the things that break. A tool whose return value contains a `date`
object passes any in-process test and fails the moment a client calls it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

mcp_client = pytest.importorskip("mcp", reason="MCP SDK not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

# The rubric requires at least five tools; the server ships eight.
EXPECTED_TOOLS = {
    "search_policy_documents",
    "get_policy_section",
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "check_policy_compliance",
    "draft_hr_email",
    "create_mock_hr_ticket",
}

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_server.server"],
    cwd=str(REPO_ROOT),
)


async def _call(session: ClientSession, name: str, arguments: dict) -> dict:
    """Call a tool and return its payload as a dict."""
    result = await session.call_tool(name, arguments)
    if getattr(result, "structuredContent", None):
        payload = result.structuredContent
        # MCPServer wraps a non-dict return in {"result": ...}; ours are dicts.
        return payload.get("result", payload) if isinstance(payload, dict) else payload
    text = "".join(
        block.text for block in result.content if getattr(block, "type", "") == "text"
    )
    return json.loads(text)


@pytest.fixture
async def session():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@pytest.mark.anyio
async def test_tools_are_discoverable_over_stdio(session):
    """The headline requirement: >= 5 tools, listed over a real transport."""
    listed = {tool.name for tool in (await session.list_tools()).tools}
    assert EXPECTED_TOOLS <= listed, f"missing: {EXPECTED_TOOLS - listed}"
    assert len(listed) >= 5


@pytest.mark.anyio
async def test_every_tool_has_a_description_and_schema(session):
    """A tool the model cannot understand is a tool it will not call correctly."""
    for tool in (await session.list_tools()).tools:
        assert tool.description, f"{tool.name} has no description"
        schema = tool.input_schema
        assert schema.get("type") == "object", f"{tool.name} has no object schema"
        assert "properties" in schema, f"{tool.name} exposes no parameters"


@pytest.mark.anyio
async def test_check_pto_balance_known_employee(session):
    payload = await _call(session, "check_pto_balance", {"employee_id": "EMP-101"})
    assert payload["ok"] is True
    assert payload["employee_id"] == "EMP-101"
    assert isinstance(payload["pto_balance_days"], (int, float))
    # Dates must arrive as strings, not have crashed serialisation.
    assert isinstance(payload["as_of"], str)


@pytest.mark.anyio
async def test_unknown_employee_fails_gracefully(session):
    """EMP-999 must come back as data, not as a protocol error."""
    result = await session.call_tool("check_pto_balance", {"employee_id": "EMP-999"})
    assert not result.is_error, "unknown id raised instead of returning a payload"

    payload = await _call(session, "check_pto_balance", {"employee_id": "EMP-999"})
    assert payload["ok"] is False
    assert payload["error"] == "employee_not_found"
    assert payload["known_employee_ids"], "no recovery hint for the agent"


@pytest.mark.anyio
async def test_lookup_employee_profile_derives_from_frozen_clock(session):
    payload = await _call(session, "lookup_employee_profile", {"employee_id": "EMP-102"})
    assert payload["ok"] is True
    # EMP-102 is inside probation as of the frozen TODAY; if this flips, the
    # clock has come unfrozen and every tenure answer in the eval set moves.
    assert payload["derived"]["is_in_probation"] is True
    assert payload["derived"]["can_take_pto"] is False


@pytest.mark.anyio
async def test_contractor_has_no_benefits(session):
    payload = await _call(session, "lookup_benefits_status", {"employee_id": "EMP-106"})
    assert payload["ok"] is True
    assert payload["benefits_eligible"] is False
    assert payload["eligibility_tests"]["is_contractor"] is True


@pytest.mark.anyio
async def test_ticket_creation_requires_confirmation(session):
    """The write gate lives in the tool, so it holds without the system prompt."""
    payload = await _call(
        session,
        "create_mock_hr_ticket",
        {
            "employee_id": "EMP-101",
            "category": "leave",
            "summary": "Test ticket that must not be created",
        },
    )
    assert payload["ok"] is False
    assert payload["error"] == "confirmation_required"
    assert payload["would_create"]["employee_id"] == "EMP-101"

    # And nothing was written.
    tickets = json.loads(
        (REPO_ROOT / "data" / "mock_data" / "tickets.json").read_text(encoding="utf-8")
    )
    assert all(
        t.get("summary") != "Test ticket that must not be created"
        for t in tickets.values()
    )


@pytest.mark.anyio
async def test_draft_email_does_not_send(session):
    payload = await _call(
        session,
        "draft_hr_email",
        {
            "recipient": "manager",
            "subject": "Leave request",
            "key_points": "Requesting three days; citing PTO-01 section 4.",
            "employee_id": "EMP-101",
        },
    )
    assert payload["ok"] is True
    assert payload["sent"] is False
    assert "Subject: Leave request" in payload["draft"]
