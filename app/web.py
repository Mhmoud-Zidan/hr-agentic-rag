"""FastAPI front end for the HR agent.

Endpoints
---------
GET  /          the chat page
POST /chat      answer + citations + tool-call trace
GET  /health    app and MCP connectivity, for the platform health check

On the agent's lifetime
-----------------------
One `HRAgent` is created at startup and shared by every request. That is not an
optimisation detail -- connecting spawns the MCP subprocess, and the first
search loads a ~130MB embedding model. Per-request construction would put that
on the critical path of every question and, on a free-tier instance, turn each
cold request into a timeout.

The cost is that the agent must be concurrency-safe. One MCP stdio session is a
single duplex pipe: two requests interleaving calls on it would race, so a lock
serialises turns. For a demo-scale app that is the right trade; a production
version would hold a pool of sessions.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import DEFAULT_MODEL, AgentError, HRAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"

load_dotenv(REPO_ROOT / ".env")

MAX_QUESTION_CHARS = 2000
#: A turn can legitimately take ~40s (several tool calls plus generation on a
#: free tier). Past that something is wrong, and a caller waiting forever is
#: worse than a clear timeout.
TURN_TIMEOUT_SECONDS = float(os.environ.get("TURN_TIMEOUT_SECONDS", "90"))


class Turn(BaseModel):
    """One prior message, replayed so follow-ups like "yes, go ahead" work."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class AppState:
    """Holds the single shared agent and the lock that serialises access."""

    agent: HRAgent | None = None
    lock: asyncio.Lock = asyncio.Lock()
    startup_error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect the agent at startup; degrade rather than refuse to boot.

    A failure here (no API key, MCP subprocess won't start) is recorded and
    surfaced by /health instead of crashing the process. A container that exits
    on boot gives the operator a restart loop and no diagnosis; one that starts
    and reports "mcp_connected: false" tells them exactly what is wrong.
    """
    try:
        agent = HRAgent()
        await agent.connect()
        state.agent = agent
    except Exception as exc:
        state.startup_error = f"{type(exc).__name__}: {exc}"

    try:
        yield
    finally:
        if state.agent is not None:
            await state.agent.aclose()
            state.agent = None


app = FastAPI(
    title="Northwind HR Assistant",
    description="Agentic HR assistant: policy RAG over MCP-exposed tools.",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> Any:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return JSONResponse({"detail": "UI not found; POST /chat still works."}, 404)
    return FileResponse(page)


@app.get("/health")
async def health() -> JSONResponse:
    """App and dependency status.

    Returns 503 when the agent could not connect, so the platform's health
    check actually fails instead of reporting a broken instance as healthy.
    """
    payload: dict[str, Any] = {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "groq_key_present": bool(os.environ.get("GROQ_API_KEY")),
    }

    if state.agent is None:
        payload["status"] = "degraded"
        payload["mcp_connected"] = False
        payload["error"] = state.startup_error or "agent not initialised"
        return JSONResponse(payload, status_code=503)

    try:
        payload.update(await state.agent.health())
    except Exception as exc:
        payload["status"] = "degraded"
        payload["mcp_connected"] = False
        payload["error"] = str(exc)
        return JSONResponse(payload, status_code=503)

    if not payload.get("mcp_connected"):
        payload["status"] = "degraded"
        return JSONResponse(payload, status_code=503)

    return JSONResponse(payload)


@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    """Answer one question, with citations and the tool-call trace."""
    if state.agent is None:
        raise HTTPException(
            status_code=503,
            detail=f"Agent unavailable: {state.startup_error or 'not initialised'}",
        )

    history = [turn.model_dump() for turn in request.history]

    try:
        # One session, one pipe: serialise turns rather than interleave them.
        async with state.lock:
            response = await asyncio.wait_for(
                state.agent.ask(request.message, history=history),
                timeout=TURN_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"The agent did not finish within {TURN_TIMEOUT_SECONDS:.0f}s.",
        )
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return JSONResponse(response.to_dict())


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.web:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=bool(os.environ.get("RELOAD")),
    )


if __name__ == "__main__":
    main()
