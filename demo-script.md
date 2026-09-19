# Demo script — 9 minutes

**Before you hit record**

1. Open https://northwind-hr-agent-fyp7.onrender.com/health a minute early. Free
   instances sleep after ~15 min idle and the first request costs ~20 s.
2. Set the provider selector to **deepseek** (~2-3 s answers, under a cent for
   the whole demo). On a free provider you will be filming a progress spinner.
3. Have these open in tabs: the app, the GitHub repo, the latest green CI run,
   and `design-and-evaluation.md`.
4. Reset the tickets file if you have been testing: `data/mock_data/tickets.json`
   should read `{}` so the ticket you create on camera is `HR-1001`.

---

## 0:00 — What it is (45 s)

> "This is an HR assistant for a fictional company, Northwind. It answers policy
> questions from a ten-document corpus and acts on employee data through eight
> tools exposed over the Model Context Protocol. Everything it says is cited,
> and every tool call is visible."

Show the README architecture diagram. One sentence on the split:

> "The agent holds no tool definitions of its own — it reads the schemas from
> the MCP server at connect time, so the server is the single source of truth
> and the two cannot drift apart."

## 0:45 — It is really MCP (30 s)

Open `/health`:

```
{"status":"ok","mcp_connected":true,"tool_count":8,"tools":[...]}
```

> "Eight tools, discovered over stdio from a separate process. The health check
> returns 503 if that subprocess is down, so a broken instance fails its
> platform health check instead of reporting itself healthy."

---

## 1:15 — TASK 1: the international work request (2 min 15 s)

Type: **"I'm EMP-101. Can I work from Spain for six weeks?"**

While it runs, say what it is doing. When it lands, expand **Tool calls**:

> "Two tools. It searched the policy corpus, then looked up who EMP-101 actually
> is — because the answer depends on their entity and location."

Then expand **Sources** and make the key point:

> "This is a refusal, and it is grounded. Spain is not on the approved-country
> list, and six weeks exceeds the thirty-day cap — two independent blockers, each
> cited to a section you can open and check. It also offers alternatives rather
> than just saying no."

**What to emphasise:** the hard part is not retrieving REM-01. It is that the
model must apply *two separate rules* and not approve on the strength of the one
it happens to retrieve first.

---

## 3:30 — TASK 2: leave request into a ticket (2 min 30 s)

This is the multi-step one. Three messages in sequence.

**Message 1:** "I'm EMP-102 and I'd like to take 3 days off next week."

Expected: declined. Expand the trace and land this point:

> "Look at *why* it declined. EMP-102 has 3.3 days of PTO, so the balance is
> sufficient — the request fails on probation, day 60 of 91, and on notice,
> because a three-day request needs ten business days. An assistant that only
> checked the balance would have approved this."

**Message 2:** "Then please raise an HR ticket to ask about unpaid leave."

Expected: it shows what it *would* create and asks for confirmation.

> "It has not written anything. It is asking first."

**Message 3:** "Yes, go ahead."

Expected: ticket created, trace shows `create_mock_hr_ticket`.

> "That confirmation gate lives in the MCP tool, not in the system prompt. The
> tool refuses without `confirmed=true`, so the guardrail holds even if the model
> skips asking. A guardrail that exists only in a prompt is one jailbreak away
> from being gone."

---

## 6:00 — Guardrails (1 min)

Two quick ones, no need to dwell.

**"Do I get stock options?"**

> "Northwind has no stock option policy. It says so instead of inventing a
> plausible one. Search returns a flag when the best match falls below the
> in-corpus similarity range — real questions land at 0.68 to 0.79, this one at
> 0.63."

**"I'm EMP-999, how many PTO days do I have?"**

> "Unknown employee. The tool returns that as *data*, not as an error, so the
> agent can ask a clarifying question rather than crash or invent a balance."

If a warning banner appears on any answer, point at it:

> "That flags a citation the agent used but never actually retrieved. I found
> the agent citing sections it had only seen cross-referenced — a correct answer
> with a fabricated citation, which is the combination that survives review. Now
> every response is checked."

---

## 7:00 — Evaluation (1 min 15 s)

Show the table in `design-and-evaluation.md`.

> "Thirty questions across seven behaviours — straightforward, multi-document,
> tool-requiring, ambiguous, out-of-scope, escalation and action safety.
> 93.3% overall. Groundedness 96.7, tool selection and workflow completion 100,
> action safety 100. Median latency under three seconds."

Then the ablation — **this is the most important slide**:

> "Same model, same prompts, tools removed: 86.7% drops to 26.7%. Sixty points.
> And latency more than halves — without tools it is faster and wrong.
>
> That gap is only possible because the answers are not recoverable from
> pre-training. That is a property of the corpus: no worked example in any policy
> document uses a value from the employee data, enforced by a script in CI. If
> examples had been built on real roster balances, the model could pattern-match
> a paragraph and the ablation would show nothing."

## 8:15 — Engineering (45 s)

Show the green CI run:

> "Three jobs: tests, corpus validation and leakage gates, a Docker build, and a
> standalone MCP tool-discovery check. Deploy only runs if all of it is green.
> The container pins Python 3.12 and builds the search index at image build
> time, so a cold start loads the model rather than downloading it, and a broken
> corpus fails the build instead of production."

## 9:00 — Close (30 s)

Be honest; it reads as strength:

> "Known limits: groundedness detects fabricated citations but not uncited false
> claims, so it is a lower bound. The corpus is synthetic and internally
> consistent, which makes retrieval easier than production. And n=30 with a
> roughly seven-point noise floor, so I would not read small differences as real."

---

## If something goes wrong on camera

- **Slow / spinner:** you are on a free provider. Switch the selector to
  deepseek and re-ask.
- **429:** free-tier quota. Switch to deepseek.
- **Cold start:** load `/health` and wait 20 s.
- **Total outage:** run locally — `python -m app.web` — the demo is identical.

## What NOT to do

- Do not read the answers aloud; the viewer can read. Narrate the **trace**.
- Do not skip the ablation. It is the one number that proves the retrieval layer
  is doing work rather than decorating a model that already knew the answer.
- Do not claim 100% anywhere. Two evaluation questions still fail and both are
  written up.
