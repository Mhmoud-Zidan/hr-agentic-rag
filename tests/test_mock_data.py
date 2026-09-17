"""Tests for app.mock_data.

Each roster record exists to unlock a specific evaluation case (CORPUS_SPEC.md
§5). These tests pin the property that makes each one useful, so a data edit
that quietly breaks an eval case fails here instead of in the eval.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.mock_data import (
    BENEFITS_HOURS_THRESHOLD,
    PROBATION_DAYS,
    PTO_USABLE_FROM_DAY,
    TODAY,
    WELLNESS_HOURS_THRESHOLD,
    EmployeeNotFoundError,
    get_benefits,
    get_employee,
    get_pto_balance,
    load_benefits,
    load_employees,
    load_pto_balances,
    load_tickets,
)

ROSTER = [
    "EMP-001",
    "EMP-004",
    "EMP-005",
    "EMP-006",
    "EMP-101",
    "EMP-102",
    "EMP-103",
    "EMP-104",
    "EMP-105",
    "EMP-106",
]


# --------------------------------------------------------------------------
# required assertions
# --------------------------------------------------------------------------


def test_all_ten_employees_load():
    employees = load_employees()
    assert len(employees) == 10
    assert sorted(employees) == ROSTER
    for emp_id, emp in employees.items():
        assert emp.employee_id == emp_id, "record key must match its employee_id"


def test_unknown_id_raises_employee_not_found():
    with pytest.raises(EmployeeNotFoundError) as excinfo:
        get_employee("EMP-999")
    assert excinfo.value.employee_id == "EMP-999"
    assert "EMP-999" in str(excinfo.value)


def test_unknown_id_raises_on_every_lookup():
    """The missing-id path must be consistent, not just on get_employee."""
    for lookup in (get_employee, get_pto_balance, get_benefits):
        with pytest.raises(EmployeeNotFoundError):
            lookup("EMP-999")


def test_emp_001_has_null_manager():
    ceo = get_employee("EMP-001")
    assert ceo.manager_id is None
    assert ceo.has_manager is False


def test_emp_102_is_inside_probation_as_of_today():
    omar = get_employee("EMP-102")
    assert omar.is_in_probation(TODAY) is True
    assert omar.tenure_days(TODAY) <= PROBATION_DAYS
    # The point of this record: PTO has accrued but cannot be taken.
    assert omar.can_take_pto(TODAY) is False
    assert get_pto_balance("EMP-102").pto_balance_days > 0
    assert omar.accrued_pto_days(TODAY) > 0


def test_emp_103_is_below_the_benefits_hours_threshold():
    priya = get_employee("EMP-103")
    assert priya.scheduled_hours_per_week < BENEFITS_HOURS_THRESHOLD
    assert priya.is_benefits_eligible_by_hours is False
    # ...but above the wellness threshold: one record, two different answers.
    assert priya.scheduled_hours_per_week >= WELLNESS_HOURS_THRESHOLD
    assert priya.is_wellness_eligible_by_hours is True

    elections = get_benefits("EMP-103")
    assert elections.benefits_eligible is False
    assert elections.has_health_coverage is False
    assert elections.retirement_401k_contribution_percent > 0
    assert elections.wellness_stipend_enrolled is True


def test_emp_106_is_a_contractor_with_no_benefits():
    marcus = get_employee("EMP-106")
    assert marcus.is_contractor is True
    assert marcus.is_benefits_eligible_by_hours is False
    assert marcus.is_wellness_eligible_by_hours is False
    assert marcus.can_take_pto(TODAY) is False

    elections = get_benefits("EMP-106")
    assert elections.benefits_eligible is False
    assert elections.has_any_benefit is False

    balance = get_pto_balance("EMP-106")
    assert balance.pto_balance_days == 0.0
    assert balance.sick_balance_days == 0


# --------------------------------------------------------------------------
# frozen clock
# --------------------------------------------------------------------------


def test_today_is_frozen_to_the_spec_date():
    assert TODAY == date(2026, 9, 17)


def test_derived_values_do_not_use_the_real_clock():
    """Passing an explicit date must change the answer; a hardcoded today would not."""
    omar = get_employee("EMP-102")
    assert omar.is_in_probation(date(2026, 8, 1)) is True
    assert omar.is_in_probation(date(2027, 1, 1)) is False
    assert omar.can_take_pto(date(2027, 1, 1)) is True


def test_probation_boundary_is_day_90_to_91():
    omar = get_employee("EMP-102")
    day_90 = omar.start_date.toordinal() + PROBATION_DAYS - 1
    day_91 = omar.start_date.toordinal() + PTO_USABLE_FROM_DAY - 1

    assert omar.tenure_days(date.fromordinal(day_90)) == PROBATION_DAYS
    assert omar.is_in_probation(date.fromordinal(day_90)) is True
    assert omar.can_take_pto(date.fromordinal(day_90)) is False

    assert omar.tenure_days(date.fromordinal(day_91)) == PTO_USABLE_FROM_DAY
    assert omar.is_in_probation(date.fromordinal(day_91)) is False
    assert omar.can_take_pto(date.fromordinal(day_91)) is True


def test_tenured_employees_are_not_in_probation():
    for emp_id in ("EMP-101", "EMP-104", "EMP-105", "EMP-001"):
        assert get_employee(emp_id).is_in_probation(TODAY) is False
        assert get_employee(emp_id).can_take_pto(TODAY) is True


# --------------------------------------------------------------------------
# accrual
# --------------------------------------------------------------------------


def test_part_time_accrual_is_prorated():
    full_time = get_employee("EMP-101")
    part_time = get_employee("EMP-103")
    assert full_time.accrual_rate_per_month == pytest.approx(1.67)
    # 25/40 of the full-time rate.
    assert part_time.accrual_rate_per_month == pytest.approx(1.67 * 25 / 40, abs=1e-4)
    assert part_time.accrual_rate_per_month < full_time.accrual_rate_per_month


def test_contractors_accrue_nothing():
    marcus = get_employee("EMP-106")
    assert marcus.accrual_rate_per_month == 0.0
    assert marcus.accrued_pto_days(TODAY) == 0.0


# --------------------------------------------------------------------------
# balances and pending requests
# --------------------------------------------------------------------------


def test_every_employee_has_a_balance_and_an_elections_record():
    assert sorted(load_pto_balances()) == ROSTER
    assert sorted(load_benefits()) == ROSTER


def test_pending_requests_exist_for_the_two_designated_records():
    balances = load_pto_balances()
    with_pending = {e for e, b in balances.items() if b.pending_requests}
    assert with_pending == {"EMP-101", "EMP-105"}
    for emp_id in with_pending:
        request = balances[emp_id].pending_requests[0]
        assert request.status == "pending"
        assert request.end_date >= request.start_date
        assert request.submitted_date < request.start_date
        assert request.days > 0
        assert balances[emp_id].pending_days == request.days


def test_pending_requests_do_not_exceed_the_balance():
    """A pending request larger than the balance would be an invalid fixture."""
    for emp_id, balance in load_pto_balances().items():
        assert balance.pending_days <= balance.pto_balance_days, emp_id


def test_carryover_expiry_is_in_the_past_so_nothing_is_usable():
    for balance in load_pto_balances().values():
        assert balance.carryover_expiry == date(2026, 3, 31)
        assert balance.carryover_is_expired(TODAY) is True
        assert balance.usable_carryover_days(TODAY) == 0.0


def test_manager_ids_resolve_to_real_employees():
    employees = load_employees()
    for emp in employees.values():
        if emp.manager_id is not None:
            assert emp.manager_id in employees, emp.employee_id


def test_dublin_staff_are_on_the_irish_entity():
    for emp_id in ("EMP-105", "EMP-006"):
        emp = get_employee(emp_id)
        assert emp.location["country"] == "Ireland"
        assert emp.entity == "Northwind Systems Ltd."
    for emp_id in ("EMP-101", "EMP-102", "EMP-001"):
        assert get_employee(emp_id).entity == "Northwind Systems Inc."


def test_tickets_starts_empty():
    assert load_tickets() == {}
