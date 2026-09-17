# Corpus Specification — Northwind Systems Inc.

**Purpose.** This is the single source of truth for the policy corpus, the mock data, and
the evaluation gold answers. Every generated document must conform to the numbers in
§3. If a document contradicts §3, the document is wrong.

Hand this file to the code-generation tool as context for *every* corpus document you
generate. Do not generate documents independently of it.

---

## 1. Fictional company profile

| Field | Value |
|---|---|
| Name | Northwind Systems Inc. |
| Industry | B2B software |
| HQ | Austin, TX, USA |
| Other US offices | San Francisco, CA; New York, NY |
| International entity | Northwind Systems Ltd. (Dublin, Ireland) |
| Headcount | ~240 |
| Fiscal year | Jan 1 – Dec 31 |
| Policy version | v4.2, effective 2026-01-01 |

All documents carry the same version and effective date in their front matter.

---

## 2. Document map

Ten documents. Target 30–120 pages total, so aim for **3–12 pages each**. Two formats
are required by the rubric, so generate the last two as `.html`.

| Doc ID | Filename | Format | Target pages |
|---|---|---|---|
| PTO-01 | `pto_and_leave_policy.md` | md | 10 |
| REM-01 | `remote_and_hybrid_work_policy.md` | md | 10 |
| TAX-01 | `work_location_and_tax_compliance.md` | md | 6 |
| EXP-01 | `expense_and_reimbursement_policy.md` | md | 10 |
| BEN-01 | `benefits_and_healthcare_policy.md` | md | 12 |
| SEC-01 | `data_security_and_acceptable_use.md` | md | 10 |
| EQP-01 | `equipment_and_asset_policy.md` | md | 6 |
| ONB-01 | `onboarding_and_probation.md` | md | 6 |
| CON-01 | `workplace_conduct_and_grievance.html` | html | 8 |
| APR-01 | `approvals_and_escalation_matrix.html` | html | 4 |

### Structural rules for every document

- H1 = document title, containing the doc ID in parentheses. Example:
  `# PTO and Leave Policy (PTO-01)`
- Front matter block at top: `doc_id`, `title`, `version`, `effective_date`, `owner`.
- H2 sections are numbered: `## 3. Carryover and Expiration`.
- H3 subsections where needed. **No H2 section may exceed ~1,200 words**, so that
  heading-aware chunking produces chunks that fit the embedding window.
- Every H2 section must be independently meaningful. A chunk retrieved alone should
  still answer a question. Avoid "as described above."
- Cross-references use doc IDs and section numbers: `see REM-01 §4`.

---

## 3. Canonical facts

**These numbers are binding.** Every document, every mock record, and every gold answer
must agree with this table.

### PTO and leave (PTO-01)

| Fact | Value |
|---|---|
| Accrual, full-time | 1.67 days/month (20 days/year) |
| Accrual, part-time | Pro-rated by scheduled hours |
| Accrual starts | First day of employment |
| PTO usable during probation | No. Accrues but cannot be taken before day 91 |
| Notice for 1–2 days | 3 business days |
| Notice for 3–5 days | 10 business days |
| Notice for 6+ days | 20 business days |
| Approver | Direct manager |
| Second approver if 10+ consecutive days | Department head |
| Carryover cap | 5 days, expiring March 31 |
| Payout on termination | Yes, capped at 10 days |
| Sick leave | Separate, 8 days/year, no accrual, no carryover |
| Bereavement | 5 days immediate family, 2 days extended |
| Parental leave | 16 weeks primary, 6 weeks secondary, after 12 months tenure |
| Unpaid leave | Requires HR + department head, max 90 days |

### Remote and hybrid work (REM-01)

| Fact | Value |
|---|---|
| Default model | Hybrid, 2 days/week in office for employees within 50 mi of an office |
| Fully remote | By exception, requires VP approval |
| Domestic out-of-state work, no approval | Up to 10 consecutive business days |
| Domestic out-of-state work, manager approval | 11–30 consecutive business days |
| Domestic out-of-state work, >30 days | Requires HR + Tax review, see TAX-01 |
| International work, no approval | **Never permitted** |
| International work, ≤14 calendar days | Manager + HR notification, Security review |
| International work, 15–30 calendar days | Manager + HR + Tax approval |
| International work, >30 calendar days | **Prohibited** unless transferred to a local entity |
| Approved international countries | Ireland, Canada, UK only (existing entity or treaty coverage) |
| Minimum notice for any international request | 30 calendar days |
| Core overlap hours | 10:00–15:00 US Central, regardless of location |

### Work location and tax (TAX-01)

| Fact | Value |
|---|---|
| US state nexus trigger | 30 cumulative days in a calendar year in a non-registered state |
| Registered states | TX, CA, NY, WA, CO |
| Permanent establishment risk | Any international presence >30 days in a 12-month period |
| Tax equalization | Not offered |
| Employee responsibility | Report location changes >5 business days via HR portal |

### Expenses (EXP-01)

| Fact | Value |
|---|---|
| Submission deadline | 45 days from expense date |
| Receipt required above | $25 |
| Manager approval required above | $250 |
| Finance approval required above | $2,000 |
| Home office stipend | $750 one-time, after 90 days tenure |
| Home office annual refresh | $200/year, from year 2 |
| Office chair | Covered under home office stipend, not separately |
| Laptop | Company-provided, **never** an employee expense, see EQP-01 |
| Meal per diem, domestic | $65/day |
| Meal per diem, international | $90/day |
| Airfare | Economy; premium economy permitted for flights >6 hours |
| Hotel cap | $250/night domestic, $300/night international |
| Personal travel combined with business | Permitted; employee pays incremental cost |
| Non-reimbursable | Alcohol, fines, personal entertainment, spouse travel |
| Reimbursement cycle | Twice monthly, 15th and last business day |

### Benefits (BEN-01)

| Fact | Value |
|---|---|
| Medical/dental/vision eligibility | Employees scheduled **≥30 hours/week** |
| Waiting period | First of the month following 30 days of employment |
| Open enrollment | November 1–21, effective January 1 |
| Qualifying life events | 30 days to change elections |
| 401(k) eligibility | Day 1 |
| 401(k) match | 100% of first 3%, 50% of next 2% |
| 401(k) vesting | Immediate |
| Contractors | Not eligible for any benefit |
| Part-time <30 hrs | 401(k) only, no health coverage |
| Dependent care FSA | $5,000/year |
| Wellness stipend | $50/month, all employees ≥20 hrs/week |
| Ireland entity | Separate benefits schedule, see BEN-01 §9 |

### Data security (SEC-01)

| Fact | Value |
|---|---|
| Data classes | Public, Internal, Confidential, Restricted |
| Customer PII | Restricted |
| Restricted data outside US/EU | Prohibited |
| VPN | Required for Confidential and Restricted access |
| Personal devices | Permitted for email/chat only, with MDM enrollment |
| Public Wi-Fi | Permitted with VPN only |
| Travel to high-risk countries | Loaner device required, Security team issues |
| Incident reporting | Within 2 hours of discovery |
| MFA | Mandatory, hardware key for Restricted access |

### Equipment (EQP-01)

| Fact | Value |
|---|---|
| Laptop refresh cycle | 36 months |
| Early replacement | Requires manager + IT approval, documented failure |
| Monitor | One 27" provided, second by manager approval |
| Peripherals | Via home office stipend, EXP-01 |
| Return on termination | Within 5 business days |
| Lost/stolen | Report within 24 hours, Security incident |

### Onboarding and probation (ONB-01)

| Fact | Value |
|---|---|
| Probation period | 90 days |
| Probation review | Day 30, day 60, day 90 |
| Extension | Up to 30 additional days, one time |
| Benefits during probation | Active, per BEN-01 waiting period |
| PTO during probation | Accrues, not usable until day 91 |

### Conduct and grievance (CON-01)

| Fact | Value |
|---|---|
| Reporting channels | Manager, HR Business Partner, anonymous ethics line |
| HR acknowledgment SLA | 2 business days |
| Investigation target | 15 business days |
| Anti-retaliation | Absolute |
| Mandatory escalation topics | Harassment, discrimination, safety, financial impropriety |

### Approvals matrix (APR-01)

Consolidated table of every approval threshold above, cross-referencing each source doc.
This document exists deliberately so that multi-document questions have a second hop.

---

## 4. Deliberate design points

These are placed on purpose. Keep them.

1. **Chair question is a trap.** EXP-01 says an office chair falls under the home office
   stipend and is not separately reimbursable, and the stipend requires 90 days tenure.
   Correct answers must combine EXP-01 + ONB-01.
2. **Laptop question is a trap.** Laptops are never an expense; they're EQP-01 assets on a
   36-month cycle. Tests whether the agent routes to the right document.
3. **Spain is not on the approved list**, and 6 weeks exceeds the 30-day international
   ceiling twice over. The correct answer for demo task 1 is a grounded *refusal with
   alternatives*, not an approval. This is much stronger on camera than a yes.
4. **PTO notice tiers interact with balance.** A 3-day request needs 10 business days'
   notice, so "next week" fails the notice rule even when the balance is sufficient.
5. **The 30 hrs/week benefits threshold** plus a 25-hour part-time employee creates a
   clean eligibility question requiring both mock data and BEN-01.
6. **Nothing in the corpus covers**: stock options, sabbaticals, pet insurance, tuition
   reimbursement, relocation. These are the out-of-scope refusal cases. Do not add them.

---

## 5. Mock roster (`mock_data/employees.json`)

Designed so each record unlocks a specific evaluation case.

| ID | Name | Role | Location | Type | Hrs/wk | Start date | Manager | PTO bal | Sick bal |
|---|---|---|---|---|---|---|---|---|---|
| EMP-101 | Jane Doe | Senior Software Engineer | San Francisco, CA | Full-Time | 40 | 2022-03-14 | EMP-004 | 14.0 | 6 |
| EMP-102 | Omar Haddad | Support Specialist | Austin, TX | Full-Time | 40 | 2026-07-20 | EMP-005 | 3.3 | 8 |
| EMP-103 | Priya Raman | Marketing Associate | Austin, TX | Part-Time | 25 | 2024-01-08 | EMP-005 | 8.5 | 5 |
| EMP-104 | Tom Becker | Data Analyst | Remote, CO | Full-Time | 40 | 2023-09-01 | EMP-004 | 1.5 | 8 |
| EMP-105 | Aoife Byrne | Account Executive | Dublin, IE | Full-Time | 40 | 2021-05-17 | EMP-006 | 19.0 | 8 |
| EMP-106 | Marcus Hale | Contractor, QA | New York, NY | Contractor | 40 | 2025-11-03 | EMP-004 | 0.0 | 0 |
| EMP-004 | Sarah Lin | VP Engineering | Austin, TX | Full-Time | 40 | 2019-02-11 | EMP-001 | 11.0 | 8 |
| EMP-005 | David Okafor | Director, GTM | Austin, TX | Full-Time | 40 | 2020-06-22 | EMP-001 | 7.0 | 8 |
| EMP-006 | Niamh Kelly | Director, EMEA | Dublin, IE | Full-Time | 40 | 2020-10-05 | EMP-001 | 15.5 | 8 |
| EMP-001 | Rachel Stone | CEO | Austin, TX | Full-Time | 40 | 2018-01-02 | `null` | 4.0 | 8 |

**What each record is for:**

- **EMP-101** — both demo tasks. Sufficient balance, insufficient notice, ineligible country.
- **EMP-102** — inside the 90-day probation window as of the demo date. PTO accrued but
  unusable. Tests the ONB-01 × PTO-01 interaction.
- **EMP-103** — 25 hrs/week. Below the 30-hour benefits threshold, above the 20-hour
  wellness threshold. Two different answers from one record.
- **EMP-104** — 1.5 days balance. Any multi-day request must be denied on balance.
  Also remote in CO, a registered state, so no nexus issue.
- **EMP-105** — Ireland entity. Different benefits schedule, and "working remotely from
  the US" inverts the usual direction of the international question.
- **EMP-106** — contractor. Ineligible for everything. Escalation case.
- **EMP-001** — `manager_id: null`. Tests graceful handling when the approval chain
  terminates.
- **EMP-999** — does not exist. Used for the missing-ID failure test. Do not create it.

Record shape:

```json
{
  "EMP-101": {
    "employee_id": "EMP-101",
    "name": "Jane Doe",
    "role": "Senior Software Engineer",
    "department": "Engineering",
    "location": { "city": "San Francisco", "state": "CA", "country": "USA" },
    "employment_type": "Full-Time",
    "scheduled_hours_per_week": 40,
    "start_date": "2022-03-14",
    "manager_id": "EMP-004",
    "entity": "Northwind Systems Inc.",
    "status": "Active"
  }
}
```

Companion files: `pto_balances.json` (balance, pending requests, carryover, expiry),
`benefits_elections.json`, `tickets.json` (starts as `{}`, written by the mock action).

**Freeze "today" to `2026-09-17`** in an env var. Accrual, probation, and notice math
must be reproducible, and a floating `date.today()` will silently break your gold answers.

---

## 6. Generation instructions

### Binding authoring rules

These two are not style preferences. Both protect the evaluation, and both fail
silently in the corpus while showing up as bogus retrieval or bogus agent metrics.

1. **Worked examples must use values that appear nowhere in `data/mock_data/`.**
   An example built on a roster balance plants the gold answer in a retrievable
   chunk: the agent then pattern-matches a paragraph instead of reasoning across
   balance, notice tier and calendar. That collapses the tools-on/tools-off
   ablation and makes workflow-completion numbers meaningless. Check the roster,
   `pto_balances.json` and `benefits_elections.json` before choosing any
   illustrative figure.

2. **Never restate a threshold defined in another document by number.**
   Cross-reference it by doc ID and section instead: "eligibility is defined in
   BEN-01 §2", never "eligibility requires 30 hours/week". A document states only
   the numbers §3 assigns to it. A restated threshold makes the borrowing document
   compete with the owning one for that query, and the resulting misroute looks
   like a chunking bug rather than an authoring one.

### Verification

Generate one document at a time, passing this file as context each time. After each one,
verify:

1. Every number appears in §3 with the same value.
2. No H2 section exceeds ~1,200 words.
3. Front matter present and correct.
4. Cross-references point at doc IDs that exist.
5. Nothing from the §4.6 out-of-scope list has crept in.
6. No worked-example value collides with anything in `data/mock_data/`.
7. No number owned by another doc in §3 has been restated instead of cross-referenced.

Write a `scripts/validate_corpus.py` that checks 2, 3, and 4 mechanically, and wire it
into CI. A corpus validator is cheap and it is the kind of thing that reads as
engineering discipline in the design document.
