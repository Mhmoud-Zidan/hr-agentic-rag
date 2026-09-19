# Deployment

**Live URL:** https://northwind-hr-agent-fyp7.onrender.com

| | |
|---|---|
| Platform | Render, Docker runtime, free plan |
| Region | Frankfurt |
| Image | `python:3.12-slim`, ~1.12 GB |
| Health check | `GET /health` |
| Source | `master` at [Mhmoud-Zidan/hr-agentic-rag](https://github.com/Mhmoud-Zidan/hr-agentic-rag) |
| Config | `render.yaml` (committed, so the service is reproducible) |

## Verified on the live instance

```
$ curl https://northwind-hr-agent-fyp7.onrender.com/health
{"status":"ok","model":"openai/gpt-oss-120b","groq_key_present":true,
 "mcp_connected":true,"tool_count":8,
 "tools":["search_policy_documents","get_policy_section",
          "lookup_employee_profile","check_pto_balance",
          "lookup_benefits_status","check_policy_compliance",
          "draft_hr_email","create_mock_hr_ticket"]}
```

| Request | Result | Latency |
|---|---|---|
| `/health` (cold) | 200, MCP connected, 8 tools | 23.2 s |
| "EMP-104, how many PTO days left?" | "1.5 PTO days" — correct, via `check_pto_balance` | 1.8 s |
| "EMP-101, work from Spain for six weeks?" | Grounded refusal: not an approved country (REM-01 §6) **and** over the 30-day cap (§5.3). No unsupported citations. | 23.5 s |

## How it deploys

The Dockerfile builds the Chroma index **at image build time** and bakes the
embedding model into the image. Two consequences worth stating:

- A cold start is a model *load*, not a *download*. Verified with
  `docker run --network none`: retrieval returns all 140 chunks with no network
  at all.
- A broken corpus fails the **build**, not production. `scripts/build_index.py`
  runs during the image build, so a document that no longer parses stops the
  deploy.

CI must pass before a deploy is triggered. `.github/workflows/ci.yml` runs tests,
both corpus gates, an MCP tool-discovery check and a full Docker build; only then
does the `deploy` job fire.

## Configuration

Set in the Render dashboard, never committed:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | **Required.** Without it the app boots and `/health` returns 503. |
| `GROQ_MODEL` | Optional. Defaults to `openai/gpt-oss-120b`. |
| `MOCK_TODAY` | Frozen clock, `2026-09-17`. Changing it invalidates the gold answers. |

To have CI trigger deploys, add Render's Deploy Hook URL as the GitHub secret
`RENDER_DEPLOY_HOOK_URL`. Without it the deploy job skips cleanly rather than
failing the run.

## Known limitations

**Cold start ~20-25 s.** The free plan sleeps after about 15 minutes idle. The
first request then pays for container start plus embedding-model load. Warm
requests are 1.8 s for a data lookup and 20-25 s for a multi-step policy
question. *If you are demoing this, hit `/health` a minute beforehand.*

**The LLM provider rate-limits before this app does.** Groq's free tier allows
8,000 tokens per minute and 200,000 per day. A tool-augmented question costs
roughly 2,700-4,300 tokens, so the daily budget is about 45-70 questions. When
it is exhausted, `/chat` returns **429** with a plain explanation — not 500,
because a provider quota is not a fault in this application and reporting it as
one sends the reader hunting for a bug that does not exist.

The per-minute limit also explains the latency profile: much of a 20-second
policy answer is waiting for token budget, not generation.

**Single shared agent, serialised.** One MCP stdio session is one pipe, so
requests are serialised behind a lock. Correct but not concurrent; a production
version would pool sessions. Adequate at demo scale, and the alternative —
constructing an agent per request — would make every cold request pay the
subprocess spawn and model load.

**Ticket writes do not persist across deploys.** `create_mock_hr_ticket` writes
to `data/mock_data/tickets.json` inside the container's filesystem, which is
ephemeral. A redeploy resets it. That is intentional for a demo with mock data;
real persistence would need a volume or a database.
