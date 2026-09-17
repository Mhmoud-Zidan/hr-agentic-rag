---
doc_id: SEC-01
title: Data Security and Acceptable Use
version: v4.2
effective_date: 2026-01-01
owner: Security
---

# Data Security and Acceptable Use (SEC-01)

## 1. Purpose, Scope, and Definitions

This policy defines how Northwind Systems Inc. classifies its data, what each class
requires, and the rules for devices, networks, authentication, and incident reporting. It
is the authoritative source for every data-handling requirement in the Northwind policy
library.

**Who this policy covers.** Everyone with access to Northwind systems or data: employees
of both entities, and contractors. Contractors are inside the scope of this policy even
though they are outside most others, because access, not employment status, is what
creates the risk.

**What it covers.** Every device used for Northwind work, every network used to reach
Northwind systems, and every piece of Northwind data wherever it is held.

### 1.1 The principle

Security obligations attach to the **data**, not to the place. Working from an office,
from home, or from another country changes nothing about what a given class of data
requires. REM-01 decides where an employee may work; this policy decides what they must
do to work securely once they are there.

### 1.2 Definitions

**Managed device.** A device provisioned or enrolled by Northwind IT, subject to
Northwind's security configuration and capable of being remotely wiped.

**MDM.** Mobile device management: the enrolment that makes a personal device manageable
for the limited purposes described in §5.2.

**VPN.** The Northwind virtual private network, which establishes an encrypted connection
to Northwind systems.

**MFA.** Multi-factor authentication: a second factor in addition to a password.

**Hardware key.** A physical security key used as an MFA factor. Required for Restricted
access under §7.2.

**Incident.** Any event that may have exposed Northwind data or systems to unauthorised
access, including a lost device, a suspected phishing click, an unexpected access prompt,
or a misdirected message containing sensitive data.

### 1.3 Related documents

- EQP-01 — equipment and assets: what hardware is issued, the refresh cycle, and the
  reporting timeline for lost or stolen equipment.
- REM-01 — remote and hybrid work: where employees may work, including the approval
  routes that require a Security review.
- CON-01 — workplace conduct and grievance, including the escalation route for
  deliberate misuse.
- APR-01 — the consolidated approvals and escalation matrix.
- ONB-01 — onboarding, including security setup for new joiners.

---

## 2. Data Classification

Northwind classifies all data into four classes: **Public, Internal, Confidential, and
Restricted**.

### 2.1 The four classes

**Public.** Information Northwind has published or is content to publish: marketing
material, public documentation, job postings. No handling restrictions.

**Internal.** The default class for ordinary business information: project plans, team
documents, internal announcements, most routine correspondence. Not for external
disclosure, but not sensitive enough to need special handling.

**Confidential.** Information whose disclosure would harm Northwind or a third party:
contracts, financial records, unreleased product plans, employee records, security
documentation.

**Restricted.** The most sensitive class, requiring the strongest controls.

### 2.2 Customer PII is Restricted

**Customer personally identifiable information is Restricted.** This is stated as a rule
rather than a judgement so that nobody has to decide case by case: if it identifies a
customer's people, it is Restricted.

Restricted data carries the strictest controls in this policy: VPN required for access
(§4.1), hardware-key MFA (§7.2), and the geographic restriction in §3.

### 2.3 Classifying in practice

If a document's class is unclear, treat it as Confidential until told otherwise, and ask
the data owner or Security. Over-classifying briefly is a minor inconvenience;
under-classifying can be an incident.

Classification follows the data, not the container. Extracting customer PII into a
spreadsheet does not make it Internal because spreadsheets usually are — the extract is
Restricted, and so is any summary that still identifies individuals.

### 2.4 Aggregation

Separate pieces of Internal information can become Confidential or Restricted when
combined, where the combination identifies people or reveals something neither piece does
alone. Anyone assembling a dataset should classify the result rather than the inputs.

---

## 3. Where Data May Be Held and Accessed

**Restricted data may not be accessed from outside the United States or the European
Union.** This is a prohibition, not a threshold.

### 3.1 What the restriction means in practice

An employee whose role involves Restricted data cannot do that part of their job from a
location outside the US or EU, even where the location itself is approved for work.

This has a direct consequence for international travel. REM-01 permits work from Ireland,
Canada, and the United Kingdom. Ireland is in the EU; Canada and the United Kingdom are
not. An employee approved under REM-01 to work from Canada or the UK still cannot access
Restricted data while there.

**A location approval is not a data-access approval.** An employee whose role touches
Restricted data should confirm with Security, before requesting an arrangement under
REM-01, that they will be able to do their job from the destination. Discovering the
restriction after arriving means being unable to work.

### 3.2 Confidential and Internal data

Confidential and Internal data carry no geographic restriction, but do carry the VPN
requirement in §4.1 for Confidential.

### 3.3 Storage

Northwind data is held in Northwind-managed systems. Copying it to personal cloud
storage, personal email, or personal devices outside the terms of §5 is prohibited at
every classification above Public, whatever the intention.

### 3.4 Third parties and AI tools

Pasting Northwind data into an external service — including a third-party AI assistant,
a translation site, a file converter, or a code-sharing site — is a disclosure to that
service. For Confidential and Restricted data it is prohibited. For Internal data, use
only services Northwind has approved; the current list is maintained by Security in the
HR portal.

Where a tool is provided and managed by Northwind, its approved classification level is
stated in that list. Where a tool is not on the list, it has not been assessed, and not
having been assessed means not approved.

---

## 4. Networks

### 4.1 VPN

**The VPN is required for access to Confidential and Restricted data**, from any location
including a Northwind office.

The requirement follows the data class, not the place. An employee accessing Internal
data from home needs no VPN; the same employee opening a Confidential contract does, even
at their desk in the office.

### 4.2 Public Wi-Fi

**Public Wi-Fi may be used only with the VPN active.**

This covers hotels, cafés, airports, conference venues, trains, and shared workspaces.
The VPN must be connected before any Northwind system is accessed, not after.

Where the VPN cannot connect on a particular network, the answer is to use a different
network — a phone hotspot is usually the simplest — rather than to proceed without it. A
network that blocks VPN connections is not a reason to work around the VPN.

### 4.3 Home networks

A home network is not a public network, but it is not a managed one either. Employees
should set a strong unique password on their home router and keep its firmware current.
The VPN requirement in §4.1 applies at home exactly as elsewhere.

### 4.4 Tethering and hotspots

A phone hotspot is preferable to an unknown public network and is the recommended
fallback while travelling. The VPN requirement still applies for Confidential and
Restricted access.

---

## 5. Devices

### 5.1 Company devices

Northwind work is done on company-provided, managed devices. Provisioning, the refresh
cycle, and return are governed by EQP-01.

Company devices must run the security configuration IT applies, must not have that
configuration disabled or circumvented, and must be kept current with updates. Installing
software from outside the approved channels is prohibited on managed devices.

### 5.2 Personal devices

**Personal devices may be used for email and chat only, and only with MDM enrolment.**

This is a narrow permission and its two limits both matter:

- **Email and chat only.** A personal device may not be used to access Northwind systems
  beyond these — not source code, not customer systems, not administrative consoles, and
  not file stores beyond what email and chat surface.
- **MDM enrolment required.** The device must be enrolled before access is granted.
  Enrolment lets Northwind enforce a passcode, require encryption, and remove Northwind
  data from the device if it is lost or when the employee leaves.

MDM enrolment gives Northwind the ability to remove **Northwind** data. It is not a
general right to inspect an employee's personal device, and Security does not read
personal content on enrolled devices.

An employee who does not want to enrol a personal device simply does not use it for
Northwind work. That is a legitimate choice and has no consequence beyond needing the
company device to hand.

### 5.3 Shared and public computers

Northwind systems must never be accessed from a shared or public computer — a hotel
business centre, a library, a client's machine, a family computer. There is no
configuration that makes this acceptable and no exception for urgency.

### 5.4 Travel to high-risk countries

**Travel to a high-risk country requires a loaner device, issued by the Security team.**

Security maintains the list of countries in this category and reviews it regularly.
Employees do not assess this themselves — the destination is checked as part of the
Security review that REM-01 requires for every international request.

The employee's usual laptop does not travel; a loaner is issued for the trip and returned
afterwards. Because issuing a device takes time, this is one of the reasons REM-01 sets a
minimum notice period for international requests. A request that leaves no time to issue
a loaner cannot be approved.

### 5.5 Physical security

Devices should not be left unattended in public, should be locked whenever an employee
steps away, and should not be checked into hold luggage. Screen privacy matters in shared
spaces: an aeroplane tray table is not a good place to review Confidential material.

---

## 6. Acceptable Use

### 6.1 What Northwind systems are for

Northwind systems and accounts are provided for Northwind work. Incidental personal use —
a personal appointment in a calendar, a quick message — is acceptable and expected.
Sustained personal use, use for a separate business, or use in a way that consumes
significant resources is not.

### 6.2 Prohibited use

Northwind systems must not be used to access, store, or distribute unlawful material, to
harass or intimidate anyone, to attempt unauthorised access to any system inside or
outside Northwind, or to circumvent a security control.

Attempting to bypass a control — disabling MDM, avoiding the VPN, sharing an MFA factor,
using an unapproved service to move data — is prohibited even where the intent is to get
work done. Employees who find a control genuinely blocking legitimate work should tell
Security so the control can be fixed; working around it silently is the problem.

### 6.3 Accounts and credentials

Accounts are personal and must not be shared. Credentials must not be reused across
Northwind and personal services, must not be written down in an accessible place, and
must not be sent over email or chat. Northwind provides a password manager, and employees
should use it.

An employee must never share an MFA factor or approve an MFA prompt they did not initiate.
An unexpected prompt is a signal that someone else has the password, and it is an incident
under §8.

### 6.4 Monitoring

Northwind monitors its systems for security purposes and logs access to Confidential and
Restricted data. Monitoring is proportionate, is directed at protecting data rather than
at supervising individuals, and does not extend to personal content on MDM-enrolled
personal devices.

### 6.5 Leaving Northwind

On the last day of employment, access is removed and Northwind data is removed from
enrolled personal devices. Employees must not retain copies of Northwind data of any
classification, including work they personally produced. Equipment return is governed by
EQP-01.

---

## 7. Authentication

### 7.1 MFA is mandatory

**Multi-factor authentication is mandatory** on every Northwind account, with no
exceptions and no opt-out.

### 7.2 Hardware keys for Restricted access

**Access to Restricted data requires a hardware key** as the MFA factor. App-based codes
and push approvals are not sufficient for this class.

Hardware keys are issued by IT to employees whose roles require Restricted access. A lost
key is reported to Security immediately and is treated as an incident under §8.

### 7.3 Phishing

Most real incidents begin with a convincing message. Employees should treat unexpected
requests for credentials, urgent payment instructions, and unexpected MFA prompts with
suspicion regardless of who the message appears to be from, and should verify through a
separate channel before acting.

Reporting a suspected phishing message is always correct, including when it turns out to
be genuine. Nobody at Northwind is criticised for reporting a message that was legitimate,
and employees who click something they should not have are expected to report it
immediately — the outcome of a fast report is almost always containable.

---

## 8. Incident Reporting

**Security incidents must be reported within 2 hours of discovery.**

### 8.1 What to report

Report a lost or stolen device; a suspected or confirmed unauthorised access; a phishing
message that was acted on; data sent to the wrong recipient; data pasted into an
unapproved external service; a lost hardware key; an unexpected MFA prompt; and anything
else that might have exposed Northwind data or systems.

If unsure whether something is an incident, report it. The 2-hour clock runs from
discovery, and deciding whether it qualifies is Security's job.

### 8.2 The 2-hour clock

The clock runs from **discovery**, not from the event. An employee who realises on Tuesday
that something happened on Friday reports within 2 hours of Tuesday's realisation.

The window is short because containment options narrow quickly. It applies at all hours
and in all timezones; Security maintains an out-of-hours contact route, and employees
travelling should know it before they leave, per REM-01 §9.4.

### 8.3 How to report

Contact Security through the channel published in the HR portal. Report first and
investigate afterwards — an employee should not spend the 2 hours establishing what
happened before telling anyone.

A lost or stolen device is reported both to Security under this section and under the
equipment reporting timeline in EQP-01.

### 8.4 No blame for reporting

An employee who reports an incident promptly, including one they caused, is doing exactly
what this policy asks. Northwind does not discipline employees for honest mistakes
reported in good time. The behaviour that is treated seriously is concealing an incident
or failing to report one, which is a conduct matter under CON-01.

This is the most important sentence in this policy: the reporting culture is worth more
than any individual control, and an employee who is afraid to report is a bigger risk than
the mistake they made.

---

## 9. Roles, Responsibilities, and Document Control

### 9.1 Employees and contractors

Everyone with access classifies data correctly, uses the VPN where §4.1 requires it, uses
public Wi-Fi only with the VPN, keeps personal-device use within §5.2, never accesses
Northwind systems from a shared computer, uses MFA, and reports incidents within 2 hours
of discovery.

### 9.2 Managers

Managers ensure their teams know these rules, do not ask anyone to work around a control,
support an employee who reports an incident, and raise blocked legitimate work with
Security rather than tolerating a workaround.

### 9.3 Security

Security owns this policy, maintains the data classification scheme and the approved
service list, maintains the high-risk country list and issues loaner devices, conducts
the Security review REM-01 requires for international requests, issues hardware keys,
operates the incident response process, and maintains an out-of-hours contact route.

### 9.4 IT

IT provisions and manages devices, administers MDM enrolment, applies security
configuration, and removes access on the last day of employment.

### 9.5 Out of scope

This policy covers data security and acceptable use. It does not cover which equipment is
issued (EQP-01), where an employee may work (REM-01), or employment matters arising from
misuse (CON-01). Where a question is not answered here or in another Northwind policy, the
correct answer is that the policy library does not cover it and the question should go to
Security.

### 9.6 Version and review

This is version v4.2, effective 2026-01-01. It supersedes all prior versions of the
Northwind Data Security and Acceptable Use policy. It is owned by Security and is reviewed
annually, and the high-risk country list and approved service list are reviewed more
frequently. A consolidated view of every approval and escalation threshold in this policy
is maintained in APR-01.
