# syntax=docker/dockerfile:1

# Python 3.12, NOT the 3.14 used for local development.
#
# chromadb, onnxruntime and fastembed ship compiled wheels that lag new CPython
# releases. On 3.14-linux a missing wheel means pip falls back to building from
# source, or fails outright -- and the first time you would find out is during
# a deploy, which is the worst possible moment. 3.12 is the newest version all
# three publish wheels for.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the model cache on a path that exists in the image and is writable
    # by the runtime user. The default (~/.cache) is not reliably either.
    HF_HOME=/opt/models \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    CHROMA_DIR=/app/chroma_db

WORKDIR /app

# Dependencies first, as their own layer: source edits then rebuild in seconds
# instead of re-resolving and re-downloading ~400MB of wheels every time.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/
COPY mcp_server/ ./mcp_server/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY static/ ./static/

# Build the index AT IMAGE BUILD TIME, not on first request.
#
# This does two things that matter on a free-tier instance. It bakes the ~130MB
# embedding model into the image, so a cold start is a model LOAD rather than a
# model DOWNLOAD; and it means the first user question does not pay for parsing,
# chunking and embedding the whole corpus. It also fails the BUILD if the corpus
# is broken, which is far better than discovering it in production.
RUN mkdir -p /opt/models && \
    python scripts/build_index.py --persist-dir "$CHROMA_DIR" && \
    python -c "from app.embeddings import get_embedder; get_embedder()"

# Everything the runtime needs is now in the image. Set AFTER the build steps
# above -- setting it earlier would block the very downloads that populate the
# cache. Without this the hub is contacted on first use just to check for
# updates, which adds latency and makes a cold start depend on the network for
# a model that is already on disk. Verified with `docker run --network none`.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Run as non-root. The app writes only to data/mock_data/tickets.json, so that
# path and the model cache are the only ones needing ownership.
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app /opt/models
USER appuser

EXPOSE 8000

# Render (and most PaaS) inject $PORT, so the port must be expanded by a shell
# -- a plain exec-form CMD would pass "${PORT:-8000}" through as a literal.
# `exec` then replaces the shell with uvicorn, so uvicorn is PID 1 and receives
# SIGTERM directly. Without it the shell holds PID 1, swallows the signal, and
# every deploy ends in a 10-second SIGKILL instead of a clean shutdown.
CMD ["sh", "-c", "exec uvicorn app.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
