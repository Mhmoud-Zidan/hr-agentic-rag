# Northwind HR Agentic RAG

An HR assistant that answers policy questions from a document corpus and acts on
employee data through tools exposed over the **Model Context Protocol**. It cites
what it used, shows every tool call it made, refuses questions the corpus does not
cover, and asks before writing anything.

```
                    ┌─────────────────────────────────────────┐
  browser ─────────▶│  FastAPI   POST /chat   GET /health     │
                    └────────────────────┬────────────────────┘
                                         │
                            ┌────────────▼────────────┐
                            │  Agent (app/agent.py)   │
                            │  Groq tool-calling loop │
                            └────────────┬────────────┘
                                         │  MCP over stdio
                            ┌────────────▼────────────┐
                            │  MCP server, 8 tools    │
                            └───┬─────────────────┬───┘
                                │                 │
                   ┌────────────▼──────┐   ┌──────▼────────────┐
                   │ Chroma + bge-small│   │ mock HR data      │
                   │ 140 chunks        │   │ 10 employees      │
                   └───────────────────┘   └───────────────────┘
```

The agent holds no tool definitions of its own. It reads the schemas from the live
MCP session at connect time, so the server is the single source of truth and the
two cannot drift apart.

## Quick start

```bash
git clone https://github.com/Mhmoud-Zidan/hr-agentic-rag
cd hr-agentic-rag
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                 # then add your Groq API key
python scripts/build_index.py
python -m app.web                                    # http://localhost:8000
```

> **Windows note.** Write `.env` with an editor, not `>` or `Out-File`. PowerShell
> defaults to UTF-16, and `python-dotenv` fails outright on a UTF-16 file. The same
> trap silently corrupted `requirements.txt` during development.

With Docker, which is what deploys:

```bash
docker build -t northwind-hr .
docker run -p 8000:8000 --env-file .env northwind-hr
```

The image builds the index at **build** time, so a cold start loads the embedding
model rather than downloading it, and a broken corpus fails the build instead of
production.

## The corpus

Ten documents, ~52 pages, in two formats — eight Markdown and two HTML — so the
parser is exercised on both. `data/CORPUS_SPEC.md` is the single source of truth:
every document, every mock record and every gold answer must agree with its §3
table.

| ID | Document | ID | Document |
|---|---|---|---|
| PTO-01 | PTO and leave | SEC-01 | Data security and acceptable use |
| REM-01 | Remote and hybrid work | EQP-01 | Equipment and assets |
| TAX-01 | Work location and tax | ONB-01 | Onboarding and probation |
| EXP-01 | Expenses and reimbursement | CON-01 | Workplace conduct *(HTML)* |
| BEN-01 | Benefits and healthcare | APR-01 | Approvals matrix *(HTML)* |

Two rules bind every document, and both are enforced mechanically:

1. **Worked examples must use values that appear nowhere in `data/mock_data/`.**
   A policy example built on a roster balance plants the gold answer inside a
   retrievable chunk. The agent then pattern-matches a paragraph instead of
   reasoning, and the tools-on/tools-off ablation collapses because the answer is
   in the corpus either way. `scripts/check_no_leak.py` fails the build on this.

2. **Never restate another document's threshold by number** — cross-reference by
   document ID and section. APR-01 is the deliberate exception; it exists to give
   multi-document questions a second hop, and it says so in its own §1.1.

```bash
python scripts/validate_corpus.py   # structure, metadata, scope
python scripts/check_no_leak.py     # eval-set leakage
```

## Retrieval

Heading-aware chunking with H2 as the retrieval unit and H3 as the fallback,
greedy packing of consecutive leaves, and a sliding window only for leaves that
exceed the cap on their own. Tokens are counted with **the embedding model's own
tokenizer**, never whitespace — `all-MiniLM-L6-v2` ships truncating at 128 tokens,
which makes every cap assertion pass vacuously while real chunks run long.

Embeddings are `BAAI/bge-small-en-v1.5`. BGE is asymmetric, so queries need an
instruction prefix that passages do not. `app/embeddings.py` records the measured
finding that **fastembed does not apply it**, with the cosine numbers, and applies
it manually in `embed_query`.

Current index: **140 chunks over 94 sections**, median 378 tokens, p95 464, max
479 against a 480 cap. Similarity is always `1 - cosine_distance`; the raw distance
never leaves the store.

## Tools

| Tool | Purpose |
|---|---|
| `search_policy_documents` | Semantic search; flags likely-out-of-corpus results |
| `get_policy_section` | Exact fetch by document ID and section |
| `lookup_employee_profile` | Role, hours, tenure, probation, approval chain |
| `check_pto_balance` | Leave balances, carryover, pending requests |
| `lookup_benefits_status` | Enrolment and eligibility tests |
| `check_policy_compliance` | Assembles profile + balances + policy evidence |
| `draft_hr_email` | Composes text; sends nothing |
| `create_mock_hr_ticket` | **Writes data. Refuses without explicit confirmation.** |

Tool errors are **returned, not raised**. An unknown employee ID comes back as
`{"ok": false, "error": "employee_not_found", "known_employee_ids": [...]}`.
Raised, it would reach the model as a protocol error — "the tool is broken" rather
than "that person does not exist" — and that distinction decides whether the agent
asks a clarifying question or gives up.

## Guardrails

- **Write confirmation** lives in the tool, not the prompt. `create_mock_hr_ticket`
  returns `confirmation_required` unless `confirmed=true`, so it holds even if the
  model skips asking. A guardrail that exists only in a system prompt is one
  jailbreak away from being gone.
- **Out-of-corpus refusal.** Search returns `likely_out_of_corpus` when the top
  similarity falls below 0.65 — calibrated on the gap between in-corpus questions
  (0.68–0.79) and absent topics like stock options (0.63).
- **Citation validation.** Every response carries `unsupported_citations`:
  references that appear in the prose but were never retrieved. This caught the
  agent citing `REM-01 §6` after reading only a cross-reference to it — a correct
  answer with a fabricated citation, which is the combination that survives review.
- **Frozen clock.** `MOCK_TODAY` defaults to `2026-09-17`. Tenure, probation and
  accrual are computed from it, never `date.today()`, so gold answers stay valid.

## Testing and evaluation

```bash
pytest                                        # 119 tests, no API key needed
python evaluation/run_eval.py                 # 30 questions, 7 metrics
python evaluation/run_eval.py --ablation tools
```

See [design-and-evaluation.md](design-and-evaluation.md) for results and what
each metric is actually worth.

## Layout

```
app/          parsers, chunker, embeddings, vector_store, agent, web
mcp_server/   MCP server (named mcp_server/, not mcp/, so it cannot
              shadow the installed SDK under pythonpath=["."])
data/         CORPUS_SPEC.md, policies/, mock_data/
scripts/      build_index, validate_corpus, check_no_leak
evaluation/   questions.json, run_eval.py, results/
tests/        119 tests
```

## Documentation

- [design-and-evaluation.md](design-and-evaluation.md) — architecture decisions,
  evaluation results, failure analysis
- [ai-tooling.md](ai-tooling.md) — how AI tooling was used to build this
- [deployed.md](deployed.md) — live URL, deployment notes, known limits
