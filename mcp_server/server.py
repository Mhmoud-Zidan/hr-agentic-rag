#!/usr/bin/env python3
"""MCP server exposing the Northwind HR tools over stdio.

The agent never imports `app.mock_data` or `app.vector_store` directly: it
speaks MCP to this process. That indirection is the point of the exercise, but
it also buys something real -- the tool schemas below are the entire contract,
so a tool can be re-implemented, moved or backed by a live HRIS without the
agent changing.

Run standalone (the transport the agent uses):

    python -m mcp_server.server

Tool inventory
--------------
retrieval   search_policy_documents, get_policy_section
people      lookup_employee_profile, check_pto_balance, lookup_benefits_status
reasoning   check_policy_compliance
actions     draft_hr_email, create_mock_hr_ticket

ERRORS ARE RETURNED, NOT RAISED. An unknown employee id is an ordinary outcome
of an agent guessing, and the agent can only recover from it if it gets a
readable payload back. Raising would surface as an MCP protocol error, which
reads to the model as "the tool is broken" rather than "that person does not
exist" -- and the difference decides whether it asks a clarifying question or
gives up. `EmployeeNotFoundError` is therefore caught at the boundary and
rendered as {"ok": false, "error": ..., "hint": ...}.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Allow `python mcp_server/server.py` as well as `-m mcp_server.server`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from app import mock_data  # noqa: E402
from app.config import CHROMA_COLLECTION, CHROMA_DIR  # noqa: E402
from app.mock_data import EmployeeNotFoundError  # noqa: E402

SERVER_NAME = "northwind-hr"
SERVER_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = Path(
    os.environ.get("MOCK_DATA_DIR", REPO_ROOT / "data" / "mock_data")
) / "tickets.json"

#: Below this cosine similarity, a hit is treated as "the corpus does not cover
#: this". Calibrated on the retrieval smoke test: in-corpus questions land at
#: 0.68-0.79, while "stock options" -- deliberately absent from the corpus --
#: tops out at 0.63. A threshold inside that gap lets the agent distinguish
#: "no policy exists" from "I retrieved badly", which is the difference between
#: a grounded refusal and a hallucinated benefit.
RELEVANCE_FLOOR = 0.65

mcp = MCPServer(
    name=SERVER_NAME,
    version=SERVER_VERSION,
    instructions=(
        "HR tools for Northwind. Policy questions must be answered from "
        "search_policy_documents or get_policy_section and cited by doc id and "
        "section. Employee-specific facts come from the lookup tools, never "
        "from policy prose. create_mock_hr_ticket writes data and requires "
        "explicit user confirmation."
    ),
)


# ---------------------------------------------------------------------------
# lazy singletons
# ---------------------------------------------------------------------------
# The embedding model is ~130MB and several seconds to load. Building it at
# import time would make tool *discovery* pay for it, so a client that only
# lists tools (CI does exactly this) would look slow or time out.

_store: Any = None


def get_store() -> Any:
    global _store
    if _store is None:
        from app.vector_store import ChromaVectorStore

        _store = ChromaVectorStore(
            persist_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION
        )
    return _store


def _json_safe(value: Any) -> Any:
    """Dates are not JSON-serialisable; everything crossing MCP must be."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _not_found(employee_id: str) -> dict[str, Any]:
    known = sorted(mock_data.load_employees())
    return {
        "ok": False,
        "error": "employee_not_found",
        "employee_id": employee_id,
        "hint": (
            "No such employee. Ask the user to confirm their employee id "
            "rather than guessing."
        ),
        "known_employee_ids": known,
    }


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Semantic search over the Northwind HR policy corpus. Use this for any "
        "question about what the rules are. Returns ranked passages with the "
        "document id and section needed to cite them."
    )
)
def search_policy_documents(
    query: str, top_k: int = 5, doc_id: str | None = None
) -> dict[str, Any]:
    """Search policy documents.

    Args:
        query: A natural-language question, in the user's own words.
        top_k: How many passages to return (1-10).
        doc_id: Optional filter, e.g. "PTO-01", to search within one document.
    """
    top_k = max(1, min(int(top_k), 10))
    where = {"doc_id": doc_id} if doc_id else None

    try:
        hits = get_store().query(query, top_k=top_k, where=where)
    except Exception as exc:  # index missing or unreadable
        return {
            "ok": False,
            "error": "index_unavailable",
            "detail": str(exc),
            "hint": "The policy index is not built. Run scripts/build_index.py.",
        }

    results = [
        {
            "citation": hit.citation,
            "doc_id": hit.metadata.get("doc_id"),
            "doc_title": hit.metadata.get("doc_title"),
            "section_number": hit.metadata.get("section_number"),
            "section_heading": hit.metadata.get("section_heading"),
            "heading_path": hit.heading_path,
            "similarity": round(hit.similarity, 4),
            "text": hit.text,
        }
        for hit in hits
    ]
    best = max((r["similarity"] for r in results), default=0.0)

    return {
        "ok": True,
        "query": query,
        "result_count": len(results),
        "best_similarity": round(best, 4),
        # An explicit flag rather than leaving the model to judge a float it has
        # no calibration for.
        "likely_out_of_corpus": best < RELEVANCE_FLOOR,
        "guidance": (
            "Top similarity is below the in-corpus range; Northwind may have no "
            "policy on this. Say so rather than inferring one."
            if best < RELEVANCE_FLOOR
            else "Cite every passage you rely on by doc id and section."
        ),
        "results": results,
    }


@mcp.tool(
    description=(
        "Fetch a specific policy section verbatim by document id and section "
        "number, e.g. PTO-01 section 4. Use after search_policy_documents when "
        "a passage is cross-referenced and you need its full text."
    )
)
def get_policy_section(doc_id: str, section_number: str) -> dict[str, Any]:
    """Fetch one policy section exactly.

    Args:
        doc_id: Document id, e.g. "BEN-01".
        section_number: Section number as printed, e.g. "9" or "4.2".
    """
    doc_id = doc_id.strip().upper()
    section_number = str(section_number).strip().lstrip("§").strip()

    try:
        store = get_store()
        raw = store._collection.get(  # exact metadata fetch, not a similarity search
            where={
                "$and": [
                    {"doc_id": doc_id},
                    {"section_number": section_number},
                ]
            },
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        return {"ok": False, "error": "index_unavailable", "detail": str(exc)}

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    if not documents:
        return {
            "ok": False,
            "error": "section_not_found",
            "doc_id": doc_id,
            "section_number": section_number,
            "hint": "Check the section number, or use search_policy_documents.",
        }

    # A long section is several chunks; reassemble in document order.
    ordered = sorted(
        zip(documents, metadatas),
        key=lambda pair: int((pair[1] or {}).get("chunk_index", 0)),
    )
    meta = ordered[0][1] or {}
    return {
        "ok": True,
        "citation": f"{doc_id} §{section_number}",
        "doc_id": doc_id,
        "doc_title": meta.get("doc_title"),
        "section_number": section_number,
        "section_heading": meta.get("section_heading"),
        "chunk_count": len(ordered),
        "text": "\n\n".join(text for text, _ in ordered),
    }


# ---------------------------------------------------------------------------
# people data
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Look up an employee's profile: role, department, work location, "
        "employment type, scheduled hours, tenure, probation status and "
        "approval chain. Use before applying any policy that depends on who "
        "the person is. This tool does NOT return leave balances -- call "
        "check_pto_balance for those. Any day figure here is lifetime accrual, "
        "not what the person has left to spend."
    )
)
def lookup_employee_profile(employee_id: str) -> dict[str, Any]:
    """Fetch one employee record with derived tenure and eligibility facts.

    Args:
        employee_id: Employee id, e.g. "EMP-101".
    """
    try:
        emp = mock_data.get_employee(employee_id)
    except EmployeeNotFoundError:
        return _not_found(employee_id)

    manager = None
    if emp.manager_id:
        try:
            mgr = mock_data.get_employee(emp.manager_id)
            manager = {"employee_id": mgr.employee_id, "name": mgr.name, "role": mgr.role}
        except EmployeeNotFoundError:
            manager = {"employee_id": emp.manager_id, "name": None, "role": None}

    return _json_safe(
        {
            "ok": True,
            "employee_id": emp.employee_id,
            "name": emp.name,
            "role": emp.role,
            "department": emp.department,
            "location": emp.location,
            "entity": emp.entity,
            "employment_type": emp.employment_type,
            "status": emp.status,
            "scheduled_hours_per_week": emp.scheduled_hours_per_week,
            "start_date": emp.start_date,
            "manager": manager,
            "as_of": mock_data.TODAY,
            "derived": {
                "tenure_days": emp.tenure_days(),
                "tenure_months": emp.tenure_months(),
                "is_in_probation": emp.is_in_probation(),
                "can_take_pto": emp.can_take_pto(),
                "is_contractor": emp.is_contractor,
                "pto_accrual_rate_days_per_month": emp.accrual_rate_per_month,
                # NAMED AT LENGTH ON PURPOSE. This is gross entitlement earned
                # since the start date; it ignores every day already taken, so
                # it is NOT the person's balance. Shipped as `accrued_pto_days`,
                # it was read as one: the agent told EMP-102 they had 1.67 days
                # when their actual balance was 3.3, and declined partly on a
                # balance that was in fact sufficient. The answer stayed right
                # for other reasons, which is exactly how that survives review.
                "pto_earned_since_start_not_current_balance": emp.accrued_pto_days(),
                "current_balance_source": (
                    "Call check_pto_balance for the spendable balance. The "
                    "figure above is lifetime accrual, not what is left."
                ),
                "meets_benefits_hours_threshold": emp.is_benefits_eligible_by_hours,
                "meets_wellness_hours_threshold": emp.is_wellness_eligible_by_hours,
            },
        }
    )


@mcp.tool(
    description=(
        "Check an employee's PTO and sick-leave balances, carryover, and any "
        "pending requests. Balances are current as of the system date; a "
        "balance alone does not mean leave is approvable -- check notice and "
        "probation rules in PTO-01 too."
    )
)
def check_pto_balance(employee_id: str) -> dict[str, Any]:
    """Fetch leave balances for one employee.

    Args:
        employee_id: Employee id, e.g. "EMP-101".
    """
    try:
        emp = mock_data.get_employee(employee_id)
        balance = mock_data.get_pto_balance(employee_id)
    except EmployeeNotFoundError:
        return _not_found(employee_id)

    return _json_safe(
        {
            "ok": True,
            "employee_id": employee_id,
            "name": emp.name,
            "as_of": mock_data.TODAY,
            "pto_balance_days": balance.pto_balance_days,
            "sick_balance_days": balance.sick_balance_days,
            "carryover_days": balance.carryover_days,
            "carryover_expiry": balance.carryover_expiry,
            "carryover_is_expired": balance.carryover_is_expired(),
            "usable_carryover_days": balance.usable_carryover_days(),
            "pending_request_days": balance.pending_days,
            "pending_requests": [
                {
                    "request_id": r.request_id,
                    "start_date": r.start_date,
                    "end_date": r.end_date,
                    "days": r.days,
                    "status": r.status,
                    "approver_id": r.approver_id,
                }
                for r in balance.pending_requests
            ],
            "eligibility": {
                "can_take_pto": emp.can_take_pto(),
                "is_in_probation": emp.is_in_probation(),
                "tenure_days": emp.tenure_days(),
            },
        }
    )


@mcp.tool(
    description=(
        "Check an employee's benefits enrolment and eligibility: medical, "
        "dental, vision, retirement contribution, FSA and wellness stipend. "
        "Eligibility depends on employment type and scheduled hours."
    )
)
def lookup_benefits_status(employee_id: str) -> dict[str, Any]:
    """Fetch benefits elections and eligibility for one employee.

    Args:
        employee_id: Employee id, e.g. "EMP-103".
    """
    try:
        emp = mock_data.get_employee(employee_id)
        benefits = mock_data.get_benefits(employee_id)
    except EmployeeNotFoundError:
        return _not_found(employee_id)

    return _json_safe(
        {
            "ok": True,
            "employee_id": employee_id,
            "name": emp.name,
            "employment_type": emp.employment_type,
            "scheduled_hours_per_week": emp.scheduled_hours_per_week,
            "benefits_eligible": benefits.benefits_eligible,
            "medical_plan_tier": benefits.medical_plan_tier,
            "dental_plan_tier": benefits.dental_plan_tier,
            "vision_plan_tier": benefits.vision_plan_tier,
            "retirement_401k_contribution_percent": (
                benefits.retirement_401k_contribution_percent
            ),
            "dependent_care_fsa_annual": benefits.dependent_care_fsa_annual,
            "wellness_stipend_enrolled": benefits.wellness_stipend_enrolled,
            "has_health_coverage": benefits.has_health_coverage,
            "eligibility_tests": {
                "is_contractor": emp.is_contractor,
                "meets_benefits_hours_threshold": emp.is_benefits_eligible_by_hours,
                "meets_wellness_hours_threshold": emp.is_wellness_eligible_by_hours,
            },
            "note": (
                "Cite the eligibility rule from BEN-01; these fields are this "
                "person's enrolment, not the policy."
            ),
        }
    )


# ---------------------------------------------------------------------------
# reasoning
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Assemble the facts needed to judge whether a specific employee's "
        "request complies with policy: their profile, balances, and the most "
        "relevant policy passages. Returns evidence, NOT a decision -- you must "
        "apply the rules and state the reasoning yourself."
    )
)
def check_policy_compliance(
    employee_id: str, request_description: str, top_k: int = 4
) -> dict[str, Any]:
    """Gather profile plus policy evidence for one request.

    Args:
        employee_id: The employee making the request, e.g. "EMP-102".
        request_description: What they are asking to do, in plain language.
        top_k: How many policy passages to include (1-8).
    """
    profile = lookup_employee_profile(employee_id)
    if not profile.get("ok"):
        return profile

    balances = check_pto_balance(employee_id)
    policy = search_policy_documents(
        request_description, top_k=max(1, min(int(top_k), 8))
    )

    return {
        "ok": True,
        "employee_id": employee_id,
        "request": request_description,
        "as_of": profile["as_of"],
        "profile": profile,
        "balances": balances if balances.get("ok") else None,
        "policy_evidence": policy.get("results", []),
        "likely_out_of_corpus": policy.get("likely_out_of_corpus", False),
        "instruction": (
            "Check every applicable condition separately -- eligibility, notice "
            "period, probation, balance, approver -- and name the one that "
            "decides the outcome. Do not approve or deny on balance alone."
        ),
    }


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Draft an HR email for the user to review and send. This only composes "
        "text; it sends nothing. Ground the body in policy and cite sections."
    )
)
def draft_hr_email(
    recipient: str, subject: str, key_points: str, employee_id: str | None = None
) -> dict[str, Any]:
    """Compose a draft email.

    Args:
        recipient: Who it is addressed to, e.g. "manager" or "HR Business Partner".
        subject: Subject line.
        key_points: The points to cover, including any policy citations.
        employee_id: Optional, to personalise the signature.
    """
    sender = "the employee"
    if employee_id:
        try:
            sender = mock_data.get_employee(employee_id).name
        except EmployeeNotFoundError:
            pass

    points = [p.strip() for p in key_points.replace(";", "\n").splitlines() if p.strip()]
    body = "\n".join(f"- {p}" for p in points) if points else key_points.strip()

    draft = (
        f"To: {recipient}\n"
        f"Subject: {subject}\n\n"
        f"Hello,\n\n{body}\n\n"
        f"Please let me know if you need anything further.\n\n"
        f"Best regards,\n{sender}"
    )
    return {
        "ok": True,
        "draft": draft,
        "sent": False,
        "note": "Draft only. Show it to the user; nothing has been sent.",
    }


@mcp.tool(
    description=(
        "Create an HR ticket. THIS WRITES DATA. Call it only after the user has "
        "explicitly agreed to raising a ticket in this turn, and pass "
        "confirmed=true. If they have not agreed, ask them first instead of "
        "calling this."
    )
)
def create_mock_hr_ticket(
    employee_id: str,
    category: str,
    summary: str,
    confirmed: bool = False,
    priority: str = "normal",
) -> dict[str, Any]:
    """Create a ticket, only with explicit confirmation.

    Args:
        employee_id: Who the ticket is for, e.g. "EMP-104".
        category: One of leave, benefits, payroll, equipment, conduct, other.
        summary: One-line description of the issue.
        confirmed: Must be true. False returns a confirmation prompt instead.
        priority: low, normal or high.
    """
    # The gate is here, in the tool, not only in the prompt. A guardrail that
    # lives solely in the system prompt is one jailbreak away from being gone;
    # this one holds even if the model decides to skip asking.
    if not confirmed:
        return {
            "ok": False,
            "error": "confirmation_required",
            "would_create": {
                "employee_id": employee_id,
                "category": category,
                "summary": summary,
                "priority": priority,
            },
            "hint": (
                "Show the user exactly what would be created and ask them to "
                "confirm. Only then call again with confirmed=true."
            ),
        }

    try:
        emp = mock_data.get_employee(employee_id)
    except EmployeeNotFoundError:
        return _not_found(employee_id)

    try:
        tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        tickets = {}
    if not isinstance(tickets, dict):
        tickets = {}

    ticket_id = f"HR-{len(tickets) + 1001}"
    ticket = {
        "ticket_id": ticket_id,
        "employee_id": emp.employee_id,
        "employee_name": emp.name,
        "category": category,
        "summary": summary,
        "priority": priority if priority in {"low", "normal", "high"} else "normal",
        "status": "open",
        "created_date": mock_data.TODAY.isoformat(),
        "assigned_to": "People Operations",
    }
    tickets[ticket_id] = ticket

    try:
        TICKETS_PATH.write_text(
            json.dumps(tickets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return {"ok": False, "error": "write_failed", "detail": str(exc)}

    return {
        "ok": True,
        "created": True,
        "ticket": ticket,
        "note": f"Ticket {ticket_id} created and assigned to People Operations.",
    }


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
