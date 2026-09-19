# Design and evaluation

## 1. What this system is

An HR assistant that answers policy questions from a ten-document corpus and
acts on employee data through eight tools exposed over MCP. It cites what it
used, shows every tool call, refuses questions the corpus does not cover, and
asks before writing.

```
browser ──▶ FastAPI ──▶ Agent (tool-calling loop) ──MCP/stdio──▶ 8 tools
                                                                  ├─ Chroma (140 chunks)
                                                                  └─ mock HR data (10 employees)
```

## 2. Design decisions that mattered

### The corpus is generated against a spec, and gates enforce it

`data/CORPUS_SPEC.md` §3 is a table of canonical facts. Every document, every
mock record and every gold answer must agree with it. Two scripts check this
mechanically, because the failure mode of generated content is not obvious
nonsense — it is **plausible drift**, a document that reads perfectly and
quietly contradicts another one.

Two authoring rules are binding:

1. **No worked example may use a value from `data/mock_data/`.** An example
   built on a roster balance plants the gold answer inside a retrievable chunk.
   The agent then pattern-matches a paragraph instead of reasoning, and the
   tools-on/off ablation collapses because the answer is in the corpus either
   way. `scripts/check_no_leak.py` fails the build on this. **§7 shows this rule
   paying for itself.**

2. **Never restate another document's threshold by number.** Cross-reference by
   document ID. APR-01 is the deliberate exception — it exists so multi-document
   questions have a second hop, and says so in its own §1.1.

### H3 became the retrieval unit, not H2

The design intent was H2-per-chunk. The spec's ~1,200-**word** section limit is
roughly 3.5× the 480-**token** cap, so H2 sections do not fit. Greedy packing of
consecutive H3 leaves, flushing at the H2 boundary, brought PTO-01 from 60
chunks at median 148 tokens to 30 at median 346. Corpus-wide: **140 chunks over
94 sections, median 378, p95 464, max 479** — none over cap, none under the
80-token floor.

### Tokens are counted with the embedding model's own tokenizer

Never whitespace. `all-MiniLM-L6-v2` ships with truncation set to 128 tokens, so
`len(encode(text).ids)` saturates and every cap assertion passes vacuously while
real chunks run long. A test now asserts truncation is disabled.

### The BGE query prefix is applied manually

BGE is asymmetric: queries need an instruction prefix that passages do not.
Measured rather than assumed — `cos(query_embed(q), embed(q)) = 1.000000`
proves fastembed does **not** apply it. `app/embeddings.py` records the numbers
and applies the prefix in `embed_query` only.

### Errors are returned from tools, not raised

An unknown employee ID comes back as
`{"ok": false, "error": "employee_not_found", "known_employee_ids": [...]}`.
Raised, it reaches the model as a protocol error — "the tool is broken" rather
than "that person does not exist" — and that distinction decides whether the
agent asks a clarifying question or gives up. Q19 confirms it asks.

### The write gate lives in the tool, not the prompt

`create_mock_hr_ticket` returns `confirmation_required` unless `confirmed=true`.
A guardrail that exists only in a system prompt is one jailbreak away from being
gone. §7 shows why this placement matters.

### The trace records operations, not reasoning

Tool name, arguments, result summary, duration. Not chain-of-thought. A
reasoning transcript is not an audit log — it is unfaithful to the actual
computation, and presenting it as the basis for a leave decision would mislead.

## 3. Results

**30 questions, DeepSeek `deepseek-chat`, 2026-09-20.** Full report:
`evaluation/results/eval_baseline_20260919T205805Z.json`.

| Metric | Rate | n |
|---|---|---|
| Overall pass rate | **93.3%** | 30 |
| Groundedness | 96.7% | 30 |
| Citation accuracy | 94.4% | 18 |
| Tool selection | 100% | 22 |
| Workflow completion | 100% | 30 |
| Escalation accuracy | 100% | 2 |
| Action safety | 100% | 9 |

| Category | Passed |
|---|---|
| straightforward | 8/8 |
| multi_doc | 5/5 |
| tool_required | 5/7 |
| ambiguous | 3/3 |
| out_of_scope | 3/3 |
| escalation | 2/2 |
| action_safety | 2/2 |

**Latency:** p50 2,936 ms · p95 4,419 ms · max 6,054 ms
**Cost:** 188,736 tokens ≈ $0.013 per full run.

Tool usage across the run — 45 calls:
`search_policy_documents` 35 · `check_pto_balance` 4 ·
`lookup_employee_profile` 3 · `lookup_benefits_status` 2 ·
`get_policy_section` 1.

### What each metric is actually worth

These are automatic checks, not a human rater.

- **Groundedness** detects citations that retrieval never returned. It is a
  **lower bound** on hallucination: it cannot detect an uncited false claim.
- **Citation accuracy** is checked at document level, so a right document with a
  wrong section still passes.
- **Workflow completion** is string matching with alternatives — a proxy for
  "did it actually answer".
- **Action safety** is the one that matters most. A single failure here is the
  difference between an assistant and an incident.

## 4. Ablation: tools on vs off

Same model, same prompts, same questions; the tool list emptied.

| Metric | Tools ON | Tools OFF | Δ |
|---|---|---|---|
| Overall pass rate | 86.7% | 26.7% | **−60.0** |
| Workflow completion | 96.7% | 26.7% | −70.0 |
| Escalation accuracy | 100% | 50% | −50.0 |
| Action safety | 100% | 100% | **0** |
| Latency p50 | 2,803 ms | 1,067 ms | −1,736 ms |

Three things this shows.

**The retrieval layer is doing the work.** A 60-point drop is only possible
because the answers are not recoverable from pre-training. That is a property of
the corpus, not luck: because no worked example contains a roster value, there
is nothing to pattern-match. Authoring rule 1 is what makes this ablation
informative rather than flat.

**Without tools the model is faster and wrong.** Latency more than halves while
pass rate falls by two thirds. Anyone optimising this system on latency alone
would make it worse.

**Action safety is unchanged at 100%.** With no tools there is nothing to write,
so the guardrail is not being carried by the prompt. That is the intended
consequence of putting the gate in the tool.

Note the tools-ON arm scored 86.7% here against 93.3% in the baseline run. Same
configuration — this is run-to-run variance at `temperature=0.1`, and it is the
honest size of the noise floor on a 30-question set. Differences smaller than
about 7 points should not be read as real.

## 5. Failure analysis

Two failures remain, both genuine.

**Q16 — EMP-106 benefits.** The agent correctly reported that a contractor has
no benefits, using tool data, but stated the eligibility rule without citing
BEN-01. Correct answer, missing provenance.

**Q17 — EMP-102, three days next week.** Correctly declined on probation *and*
notice — the trap the question was built for — but cited PTO-01 §2.4 and §6.2,
neither of which was retrieved. This is the residual of the dominant failure
mode below.

### The dominant failure mode, and what fixed it

Before the final prompt revision, **5 of 7 failures were the same thing**: the
agent reads a cross-reference inside a retrieved passage and cites it as though
it had read that section. The answer is typically *correct* and the citation
fabricated, which is the most dangerous combination — it reads as well-sourced
and survives a casual check.

Adding an explicit instruction — cite only sections returned by your own search
this turn; fetch with `get_policy_section` or describe the rule without a
section number — moved:

| | Before | After |
|---|---|---|
| Groundedness | 83.3% | 96.7% |
| Tool selection | 95.5% | 100% |
| Overall | 76.7% | 93.3% |

### One failure worth naming separately

In an earlier run the agent wrote *"I searched Northwind's HR policy corpus...
there's no policy covering it"* for a question about suspected expense fraud.
The trace shows **it called no tools at all**, and CON-01 explicitly covers
financial impropriety as a mandatory escalation topic. It asserted an action it
had not taken and drew a false conclusion from the imagined result.

This is worse than a wrong answer, because the fabricated process is what makes
it credible. The prompt now forbids claiming a search that did not happen, and
the trace makes the claim checkable — but the general problem is not solved by a
prompt, and a production system should verify such claims against the trace
programmatically.

## 6. What the evaluation infrastructure got wrong

Recorded because the harness was wrong more often than the agent, and a metric
nobody audits is worse than no metric.

**Two full runs were contaminated by rate limits.** Questions the provider
refused were scored as *wrong answers*, so the metrics measured the provider's
quota, not the agent. A run reported 56.7% when the honest figure over the
questions that actually ran was 77.3%. Fixed: refused questions are now marked
`errored`, excluded from every metric, and the summary prints `>> INCOMPLETE
RUN` with the missing IDs.

**The citation checker matched employee IDs.** `EMP-004` parsed as "EMP-00 §4"
and `REQ-2026-0141` as "REQ-20 §26", so **any answer naming an employee was
scored ungrounded**. Fixed by requiring an explicit section marker.

**Before that, the same checker matched nothing at all.** The first version used
an ASCII hyphen; the model writes `PTO‑01` with U+2011. It silently reported
zero violations, which is indistinguishable from a clean result.

**A forbidden-phrase check fired inside a negation.** "Should I ignore it?" →
"No — don't just ignore it" is the *correct* answer, and it was scored as the
wrong one.

**Three gold answers were wrong.** Q06 matched the literal `"5 day"` against a
correct answer saying "5 PTO days"; Q11 demanded a notice period the question
never asked for; Q23 required a literal "?" when answering both readings is also
correct.

All were the grader failing, not the agent, and all are visible in git history.
Changing an evaluation after seeing results is exactly how scores get fixed, so
each change is recorded here and in `ai-tooling.md`.

## 7. Known limitations

- **Groundedness is a lower bound.** It cannot detect an uncited false claim.
- **String matching is a proxy** for whether a question was answered.
- **n=30, single run per configuration.** The noise floor is ~7 points (§4).
  Differences below that are not results.
- **The corpus is synthetic** and internally consistent by construction. Real HR
  corpora contradict themselves; this one cannot, so retrieval here is easier
  than in production.
- **`char_start`/`char_end` locate the parent section, not the chunk**, so a
  citation cannot yet be resolved to an exact offset.
- **Carryover is inert** — every roster member has 0.0 days expiring before the
  frozen clock, so no question exercises carryover reasoning.
- **One shared MCP session, serialised** behind a lock. Correct, not concurrent.
