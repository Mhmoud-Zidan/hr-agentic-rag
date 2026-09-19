# AI tooling

How AI tooling was used to build this project, and — more usefully — where it
was wrong and how that was caught.

## What was used

**Claude (Claude Code)** for the whole build: corpus authoring, the ingestion
pipeline, the MCP server, the agent, the web layer, the Docker and CI setup, and
the evaluation harness. Work proceeded in reviewed increments rather than one
generation pass, with tests and mechanical gates written alongside each piece.

**Groq** (`openai/gpt-oss-120b`) as the runtime LLM inside the product itself.

## The working pattern

The division that held up was: **the human owned the specification and the
adversarial review; the model owned the implementation.**

`data/CORPUS_SPEC.md` was written first and treated as binding. Every generated
policy document had to agree with its §3 canonical-facts table, and two scripts
enforced that rather than trusting the generation:

- `scripts/validate_corpus.py` — structure, metadata, scope
- `scripts/check_no_leak.py` — eval-set leakage

This matters because the failure mode of generated content is not obvious
nonsense. It is **plausible drift**: a document that reads perfectly and quietly
contradicts another one. A spec plus a checker catches that; reading the output
does not.

## Where the model was wrong

These are the substantive errors. All were caught, but they illustrate what to
distrust.

### Leaking the evaluation set into the corpus

The first generated policy document included worked examples using balances of
**14.0, 3.3 and 1.5 days** — the exact balances of three roster employees. The
prose was perfect; "an employee with 14.0 days" reads like good policy writing.

But it plants the gold answer inside a retrievable chunk. The agent then
pattern-matches a paragraph instead of reasoning across balance, notice tier and
calendar, and the tools-on/tools-off ablation collapses because the answer is in
the corpus either way.

Caught by human review, not by the model. It became binding authoring rule 1 and
is now enforced by `check_no_leak.py`.

### The checker for that bug was itself broken — twice

Worth recording, because "we added a check" is not the same as "the check works".

1. The first version suppressed by **value**, not spelling. Run against the
   original leak it *passed*, because §3 contains "14" and the roster balance is
   "14.0". Rewritten to suppress by exact spelling.
2. Its HTML masking ran in the wrong order — tags were stripped **last**, so
   `<h2>4. Expense Approvals</h2>` never matched the heading pattern and the `4`
   was reported as a leak. Reordered so markup is stripped first.

Both were found by deliberately testing the checker against a known-bad input.
A gate that has never failed has not been shown to work.

### Tokenizer truncation made every assertion vacuous

The chunker was initially built against `all-MiniLM-L6-v2`, whose tokenizer ships
with truncation set to 128 tokens. `len(encode(text).ids)` therefore saturates at
128, so every "chunk is under the cap" assertion passed while real chunks ran
long. Fixed with an explicit `no_truncation()`, and pinned by a test that asserts
truncation is off.

This is the archetype: **the code was wrong in a way that made the tests agree
with it.**

### A fabricated citation inside a correct answer

Asked whether an employee could work from Spain for six weeks, the agent produced
the right refusal for the right reasons — and cited `REM-01 §6` and `§7.1`, neither
of which had been retrieved. It had read cross-references inside a retrieved
section and repeated the pointers as sources.

The answer was right and the citation was invented, which is the most dangerous
combination, because it reads as well-sourced and survives a casual check. Now
detected on every response via `unsupported_citations`.

### The detector for *that* bug silently matched nothing

The first citation regex used an ASCII hyphen. The model writes `PTO‑01` with a
non-breaking hyphen (U+2011), so the pattern matched **nothing** and every answer
looked perfectly cited. A detector that never fires is indistinguishable from a
clean result. There is now a parametrised test over five dash variants.

### Confusing accrual with balance

`lookup_employee_profile` returned a field called `accrued_pto_days` — lifetime
accrual, ignoring days already taken. The model read it as a spendable balance and
told an employee they had **1.67 days** when their actual balance was **3.3**, and
declined their request partly on a balance that was in fact sufficient.

The decline was still correct on other grounds, so the right answer masked a wrong
sub-reason. Fixed by renaming the field to
`pto_earned_since_start_not_current_balance` — a name that cannot be misread beats
a comment the model never sees.

### Encoding, repeatedly

`requirements.txt` was generated as **UTF-16 with a BOM** by a PowerShell
`pip freeze`, unreadable by pip on Linux, and pinned Windows-only packages. It was
rewritten — and the rewrite was *also* UTF-16, because the fix used PowerShell
redirection again. It was reported as fixed after a check that only tested for a
BOM, which a BOM-less UTF-16 file passes.

The same trap then hit `.env`, where `python-dotenv` crashed outright.

Lesson: verify the property you care about, not a proxy for it. The check should
have decoded the file, not looked for a marker.

## What AI tooling was good and bad at

**Good:** boilerplate with many small correctness details (parsers, dataclasses,
CLI plumbing); long structured documents conforming to a spec; test scaffolding;
explaining an unfamiliar API surface (the MCP 2.x SDK, where `FastMCP` had been
renamed to `MCPServer` and every client model moved to snake_case, so most
published examples were wrong).

**Bad, or needing supervision:** anything where being wrong is invisible.
Truncation defaults, encodings, a regex that matches nothing, a similarity
threshold, a field name that misleads a downstream consumer. In every case the
code ran and produced confident output.

The through-line: AI tooling accelerates writing code enormously and does not
reduce the need to decide what "correct" means and test for it adversarially.
Most of the real work in this project was specification and verification.

## Honest limitations

- `evaluation/run_eval.py` grades by string matching, not by a human or an
  LLM judge. It is deliberately strict and transparent, but "workflow completion"
  means "the required facts appear in the text", which is a proxy.
- `groundedness` detects citations that were never retrieved. It cannot detect an
  uncited false claim, so it is a **lower bound** on hallucination.
- The corpus is synthetic and internally consistent by construction, which makes
  it easier than a real one — real HR corpora contradict themselves.
