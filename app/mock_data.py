"""Typed loaders for the mock HR records in data/mock_data/.

Everything time-dependent is computed from `TODAY`, never `date.today()`.
CORPUS_SPEC.md §5 freezes "today" to 2026-09-17 precisely because accrual,
probation and notice math have to be reproducible: a floating clock would make
gold answers drift silently, and EMP-102's probation case would quietly stop
being a probation case once enough real time passed.

Override with the MOCK_TODAY environment variable (ISO date) for what-if runs.

Thresholds used in derived values are the canonical ones from CORPUS_SPEC.md §3
and are defined once, here, so that eval code never re-types them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "BENEFITS_HOURS_THRESHOLD",
    "DATA_DIR",
    "Employee",
    "EmployeeNotFoundError",
    "MockDataError",
    "PendingRequest",
    "PtoBalance",
    "BenefitsElection",
    "PROBATION_DAYS",
    "PTO_USABLE_FROM_DAY",
    "TODAY",
    "WELLNESS_HOURS_THRESHOLD",
    "get_benefits",
    "get_employee",
    "get_pto_balance",
    "load_benefits",
    "load_employees",
    "load_pto_balances",
    "load_tickets",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MOCK_DATA_DIR", REPO_ROOT / "data" / "mock_data"))

#: Frozen "today". Every derived value below is relative to this, not the clock.
TODAY: date = date.fromisoformat(os.environ.get("MOCK_TODAY", "2026-09-17"))

# --- canonical thresholds (CORPUS_SPEC.md §3) ------------------------------
PROBATION_DAYS = 90  # ONB-01
PTO_USABLE_FROM_DAY = 91  # PTO-01: accrues during probation, usable from day 91
FULL_TIME_HOURS = 40
FULL_TIME_MONTHLY_ACCRUAL = 1.67  # PTO-01: days/month, 20 days/year
BENEFITS_HOURS_THRESHOLD = 30  # BEN-01: medical/dental/vision
WELLNESS_HOURS_THRESHOLD = 20  # BEN-01: wellness stipend
CONTRACTOR_TYPE = "Contractor"


class MockDataError(RuntimeError):
    """A mock data file is missing or malformed."""


class EmployeeNotFoundError(KeyError):
    """No employee with the given id exists.

    Deliberately explicit rather than a bare KeyError: CORPUS_SPEC.md §5 uses
    EMP-999 as the missing-id failure test, and the agent needs to distinguish
    "this person does not exist" from "this lookup blew up".
    """

    def __init__(self, employee_id: str):
        self.employee_id = employee_id
        super().__init__(employee_id)

    def __str__(self) -> str:  # KeyError would otherwise quote the message
        return f"no employee with id {self.employee_id!r}"


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Employee:
    employee_id: str
    name: str
    role: str
    department: str
    location: dict[str, Any]
    employment_type: str
    scheduled_hours_per_week: int
    start_date: date
    manager_id: str | None
    entity: str
    status: str

    # -- derived, all relative to TODAY --------------------------------------

    @property
    def is_contractor(self) -> bool:
        return self.employment_type == CONTRACTOR_TYPE

    def tenure_days(self, as_of: date = TODAY) -> int:
        """Days since the start date. Day 1 is the first day of employment."""
        return (as_of - self.start_date).days + 1

    def tenure_months(self, as_of: date = TODAY) -> int:
        """Whole months completed, used for accrual."""
        months = (as_of.year - self.start_date.year) * 12 + (
            as_of.month - self.start_date.month
        )
        if as_of.day < self.start_date.day:
            months -= 1
        return max(months, 0)

    def is_in_probation(self, as_of: date = TODAY) -> bool:
        """True through day 90. Contractors have no probation period."""
        if self.is_contractor:
            return False
        return self.tenure_days(as_of) <= PROBATION_DAYS

    def can_take_pto(self, as_of: date = TODAY) -> bool:
        """PTO-01: accrues from day 1, not usable before day 91."""
        if self.is_contractor:
            return False
        return self.tenure_days(as_of) >= PTO_USABLE_FROM_DAY

    @property
    def accrual_rate_per_month(self) -> float:
        """PTO-01: 1.67 days/month full-time, pro-rated by scheduled hours."""
        if self.is_contractor:
            return 0.0
        ratio = self.scheduled_hours_per_week / FULL_TIME_HOURS
        return round(FULL_TIME_MONTHLY_ACCRUAL * ratio, 4)

    def accrued_pto_days(self, as_of: date = TODAY) -> float:
        """PTO accrued since the start date. Not the same as current balance.

        The balance in pto_balances.json reflects days already taken; this is the
        gross figure, useful for checking a new hire's entitlement.
        """
        return round(self.accrual_rate_per_month * self.tenure_months(as_of), 2)

    @property
    def is_benefits_eligible_by_hours(self) -> bool:
        """BEN-01 hours test only. Contractors are ineligible regardless."""
        if self.is_contractor:
            return False
        return self.scheduled_hours_per_week >= BENEFITS_HOURS_THRESHOLD

    @property
    def is_wellness_eligible_by_hours(self) -> bool:
        if self.is_contractor:
            return False
        return self.scheduled_hours_per_week >= WELLNESS_HOURS_THRESHOLD

    @property
    def has_manager(self) -> bool:
        """False for the CEO; the approval chain terminates there by design."""
        return self.manager_id is not None


@dataclass(frozen=True)
class PendingRequest:
    request_id: str
    start_date: date
    end_date: date
    days: float
    submitted_date: date
    status: str
    approver_id: str | None


@dataclass(frozen=True)
class PtoBalance:
    employee_id: str
    pto_balance_days: float
    sick_balance_days: int
    carryover_days: float
    carryover_expiry: date
    pending_requests: tuple[PendingRequest, ...] = field(default=())

    @property
    def pending_days(self) -> float:
        return round(sum(r.days for r in self.pending_requests), 2)

    def carryover_is_expired(self, as_of: date = TODAY) -> bool:
        return as_of > self.carryover_expiry

    def usable_carryover_days(self, as_of: date = TODAY) -> float:
        return 0.0 if self.carryover_is_expired(as_of) else self.carryover_days


@dataclass(frozen=True)
class BenefitsElection:
    employee_id: str
    benefits_eligible: bool
    medical_plan_tier: str | None
    dental_plan_tier: str | None
    vision_plan_tier: str | None
    retirement_401k_contribution_percent: float
    dependent_care_fsa_annual: int
    wellness_stipend_enrolled: bool

    @property
    def has_health_coverage(self) -> bool:
        return any(
            (self.medical_plan_tier, self.dental_plan_tier, self.vision_plan_tier)
        )

    @property
    def has_any_benefit(self) -> bool:
        return (
            self.has_health_coverage
            or self.retirement_401k_contribution_percent > 0
            or self.dependent_care_fsa_annual > 0
            or self.wellness_stipend_enrolled
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _read_json(filename: str) -> dict[str, Any]:
    path = DATA_DIR / filename
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise MockDataError(f"missing mock data file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MockDataError(f"malformed JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MockDataError(f"{path} must contain a JSON object keyed by employee id")
    return payload


@lru_cache(maxsize=1)
def load_employees() -> dict[str, Employee]:
    raw = _read_json("employees.json")
    return {
        emp_id: Employee(
            employee_id=rec["employee_id"],
            name=rec["name"],
            role=rec["role"],
            department=rec["department"],
            location=dict(rec["location"]),
            employment_type=rec["employment_type"],
            scheduled_hours_per_week=int(rec["scheduled_hours_per_week"]),
            start_date=date.fromisoformat(rec["start_date"]),
            manager_id=rec.get("manager_id"),
            entity=rec["entity"],
            status=rec["status"],
        )
        for emp_id, rec in raw.items()
    }


@lru_cache(maxsize=1)
def load_pto_balances() -> dict[str, PtoBalance]:
    raw = _read_json("pto_balances.json")
    return {
        emp_id: PtoBalance(
            employee_id=rec["employee_id"],
            pto_balance_days=float(rec["pto_balance_days"]),
            sick_balance_days=int(rec["sick_balance_days"]),
            carryover_days=float(rec["carryover_days"]),
            carryover_expiry=date.fromisoformat(rec["carryover_expiry"]),
            pending_requests=tuple(
                PendingRequest(
                    request_id=r["request_id"],
                    start_date=date.fromisoformat(r["start_date"]),
                    end_date=date.fromisoformat(r["end_date"]),
                    days=float(r["days"]),
                    submitted_date=date.fromisoformat(r["submitted_date"]),
                    status=r["status"],
                    approver_id=r.get("approver_id"),
                )
                for r in rec.get("pending_requests", [])
            ),
        )
        for emp_id, rec in raw.items()
    }


@lru_cache(maxsize=1)
def load_benefits() -> dict[str, BenefitsElection]:
    raw = _read_json("benefits_elections.json")
    return {
        emp_id: BenefitsElection(
            employee_id=rec["employee_id"],
            benefits_eligible=bool(rec["benefits_eligible"]),
            medical_plan_tier=rec.get("medical_plan_tier"),
            dental_plan_tier=rec.get("dental_plan_tier"),
            vision_plan_tier=rec.get("vision_plan_tier"),
            retirement_401k_contribution_percent=float(
                rec["retirement_401k_contribution_percent"]
            ),
            dependent_care_fsa_annual=int(rec["dependent_care_fsa_annual"]),
            wellness_stipend_enrolled=bool(rec["wellness_stipend_enrolled"]),
        )
        for emp_id, rec in raw.items()
    }


def load_tickets() -> dict[str, Any]:
    """Not cached: the mock action tool writes this file during a run."""
    return _read_json("tickets.json")


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------


def get_employee(employee_id: str) -> Employee:
    try:
        return load_employees()[employee_id]
    except KeyError:
        raise EmployeeNotFoundError(employee_id) from None


def get_pto_balance(employee_id: str) -> PtoBalance:
    """Raises EmployeeNotFoundError for an unknown id, not a bare KeyError."""
    balances = load_pto_balances()
    if employee_id not in balances:
        if employee_id not in load_employees():
            raise EmployeeNotFoundError(employee_id)
        raise MockDataError(f"{employee_id} has no PTO balance record")
    return balances[employee_id]


def get_benefits(employee_id: str) -> BenefitsElection:
    """Raises EmployeeNotFoundError for an unknown id, not a bare KeyError."""
    elections = load_benefits()
    if employee_id not in elections:
        if employee_id not in load_employees():
            raise EmployeeNotFoundError(employee_id)
        raise MockDataError(f"{employee_id} has no benefits record")
    return elections[employee_id]
