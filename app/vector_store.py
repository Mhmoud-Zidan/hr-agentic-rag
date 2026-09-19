"""Vector store abstraction and a Chroma-backed implementation.

The Protocol exists so retrieval code never imports chromadb directly, and so a
BM25 or hybrid store can be dropped in for the ablation without touching callers.

Two invariants this module enforces:

  * Cosine space. The collection is created with metadata={"hnsw:space": "cosine"}.
    Chroma's default is L2, and on normalized vectors L2 ranks the same but the
    numbers are not comparable across stores and thresholds tuned on one are
    meaningless on the other.
  * Callers only ever see `similarity = 1 - distance`, never a raw distance.
    Chroma returns a DISTANCE (lower is better). Handing that to an LLM prompt or
    a threshold check as if it were a score inverts the ranking silently.

Embeddings are computed here via app.embeddings so that passages get bare text
and queries get the BGE instruction prefix. Chroma's own embedding function is
never used -- it would default to MiniLM and apply no prefix at all.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from app.chunker import Chunk
from app.config import CHROMA_COLLECTION, CHROMA_DIR
from app.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, get_embedder

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_PERSIST_DIR",
    "ChromaVectorStore",
    "SearchResult",
    "VectorStore",
]

# Both from app.config so the writer and every reader resolve to one path.
DEFAULT_PERSIST_DIR = CHROMA_DIR
DEFAULT_COLLECTION = CHROMA_COLLECTION

# Chroma needs this at creation time; changing it later requires a rebuild.
COSINE_SPACE = {"hnsw:space": "cosine"}


@dataclass(frozen=True)
class SearchResult:
    """One hit. `similarity` is cosine similarity in [-1, 1]; higher is better."""

    chunk_id: str
    text: str
    similarity: float
    metadata: dict[str, Any]

    @property
    def heading_path(self) -> str:
        return str(self.metadata.get("heading_path", ""))

    @property
    def citation(self) -> str:
        doc_id = self.metadata.get("doc_id", "?")
        number = self.metadata.get("section_number", "?")
        return f"{doc_id} §{number}"


@runtime_checkable
class VectorStore(Protocol):
    """The surface retrieval code is allowed to depend on."""

    def add(self, chunks: Sequence[Chunk]) -> int:
        """Index chunks. Returns the number written. Idempotent on chunk_id."""

    def query(
        self,
        text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search by text. Results carry similarity, never distance."""

    def count(self) -> int:
        """Number of indexed chunks."""

    def reset(self) -> None:
        """Drop everything in the collection."""


class ChromaVectorStore:
    """Chroma PersistentClient implementation of VectorStore."""

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedder: Embedder | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._embedder = embedder or get_embedder(model_name)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata=COSINE_SPACE,
            # We supply vectors ourselves; Chroma's default EF is the wrong model
            # and would silently embed queries without the BGE prefix.
            embedding_function=None,
        )

    # -- writes ------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate chunk_ids in this batch: {dupes}")

        embeddings = self._embedder.embed_passages([c.text for c in chunks])
        # upsert, not add: re-indexing an unchanged corpus must not duplicate rows.
        self._collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in chunks],
        )
        return len(chunks)

    # -- reads -------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        vector = self._embedder.embed_query(text)
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, max(self.count(), 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=document or "",
                    # Chroma cosine distance = 1 - cosine similarity.
                    similarity=1.0 - float(distance),
                    metadata=dict(metadata or {}),
                )
            )
        return results

    def count(self) -> int:
        return int(self._collection.count())

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Delete and recreate the collection, preserving the cosine setting."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=COSINE_SPACE,
            embedding_function=None,
        )

    def destroy(self) -> None:
        """Remove the on-disk store entirely. Tests and rebuilds only."""
        try:
            self._client.reset()
        except Exception:
            pass
        shutil.rmtree(self.persist_dir, ignore_errors=True)

    @property
    def space(self) -> str:
        """The distance space actually configured on the live collection."""
        metadata = self._collection.metadata or {}
        return str(metadata.get("hnsw:space", "l2"))
