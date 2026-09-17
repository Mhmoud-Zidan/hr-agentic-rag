"""Embedding wrapper around fastembed / BAAI/bge-small-en-v1.5.

FINDING: fastembed does NOT apply the BGE query instruction prefix.
=================================================================

BGE v1.5 retrieval models are trained asymmetrically. Passages are embedded
bare; queries are supposed to be embedded with the instruction prefix
"Represent this sentence for searching relevant passages: " prepended. The
model card is explicit about this. Skipping it costs a few points of
retrieval accuracy, and the failure is completely silent -- you get plausible
neighbours, just worse ones.

`fastembed.TextEmbedding` exposes `query_embed()` and `passage_embed()`,
whose names strongly imply they handle this. For dense text models they do
not. Verified two ways against fastembed 0.8.0:

1. Source. `TextEmbeddingBase.query_embed` (text/text_embedding_base.py:46)
   is `yield from self.embed([query])` -- a plain passthrough, with a comment
   noting that prefixing is left to model-specific subclasses. No dense BGE
   subclass overrides it; only ColBERT, BM25/BM42/miniCOIL and the multitask
   (jina) models do. Grepping the whole package for the literal prefix string
   returns nothing: it is not in fastembed at all.

2. Numerically, on BAAI/bge-small-en-v1.5 with the query
   "How many days of PTO carry over?":

       cos(query_embed(q),  embed(q))          = 1.000000
       cos(query_embed(q),  embed(PREFIX + q)) = 0.973882
       cos(embed(q),        embed(PREFIX + q)) = 0.973882

   `query_embed` is bit-identical to `embed`. The prefix is a real,
   measurable change to the vector that fastembed never makes.

CONSEQUENCE: this module applies the prefix itself, in `embed_query` only.
`embed_passages` deliberately does not prefix -- prefixing both sides is the
other half of this mistake and re-introduces the same mismatch.

MEASURED IMPACT (be honest about this): on the corpus as it stands -- one
document, 60 chunks -- an 8-query probe with known gold sections showed the
prefix making things very slightly WORSE, not better:

    mean rank of first correct-section chunk: 1.12 bare -> 1.25 prefixed
    MRR    0.9375 bare -> 0.8750 prefixed
    hit@1  7/8    bare -> 6/8    prefixed

That is one query moving from rank 1 to rank 2, which is noise at ceiling: with
a single document and no competing sources, nearly every query already retrieves
its own section first. The probe is too easy to separate the two conditions.

The prefix is kept anyway, for two reasons. It is the documented usage on the
BGE v1.5 model card, so it is what the asymmetric training expects; and the
condition that actually distinguishes it -- many documents, near-duplicate
passages competing across sources -- does not exist yet. Re-run the probe once
the full ten-document corpus is built. If the prefix is still not earning its
place on a harder eval set, set `query_prefix=""` on the Embedder and record
that as a measured decision rather than an assumption in either direction.

Vectors come back L2-normalized (verified: ||v|| = 1.0), so cosine distance
lands in [0, 2] and `similarity = 1 - distance` is exact. app.vector_store
relies on that.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

from app.config import EMBEDDING_MODEL as DEFAULT_EMBEDDING_MODEL

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "QUERY_PREFIX",
    "Embedder",
    "embed_passages",
    "embed_query",
    "get_embedder",
]

# DEFAULT_EMBEDDING_MODEL is re-exported from app.config, which is the single
# source for it. Do not read the environment variable here.

# The instruction prefix from the BGE v1.5 model card. Queries only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBEDDING_DIM = 384  # bge-small-en-v1.5

DEFAULT_BATCH_SIZE = 32


class Embedder:
    """Passage/query asymmetry made explicit, because fastembed does not do it."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        query_prefix: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        # Only BGE-family models take this prefix; an explicit "" disables it.
        self.query_prefix = (
            QUERY_PREFIX if query_prefix is None and "bge" in model_name.lower() else (query_prefix or "")
        )
        self._model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: Sequence[str] | Iterable[str]) -> list[list[float]]:
        """Embed documents for indexing. No prefix -- BGE passages are bare."""
        texts = list(texts)
        if not texts:
            return []
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("embed_passages expects strings")
        vectors = self._model.embed(texts, batch_size=self.batch_size)
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, with the BGE instruction prefix applied here."""
        if not isinstance(text, str):
            raise TypeError("embed_query expects a single string")
        prefixed = f"{self.query_prefix}{text}"
        # NOTE: self._model.embed, not query_embed -- see the module docstring.
        # query_embed is a passthrough and would drop the prefix work above.
        vector = next(iter(self._model.embed([prefixed])))
        return [float(x) for x in vector]

    def embed_queries(self, texts: Sequence[str] | Iterable[str]) -> list[list[float]]:
        """Batch form of embed_query, same prefixing."""
        texts = list(texts)
        if not texts:
            return []
        prefixed = [f"{self.query_prefix}{t}" for t in texts]
        vectors = self._model.embed(prefixed, batch_size=self.batch_size)
        return [[float(x) for x in v] for v in vectors]


@lru_cache(maxsize=2)
def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    """Cached. Model load and ONNX session creation are the expensive parts."""
    return Embedder(model_name)


def embed_passages(texts: Sequence[str] | Iterable[str]) -> list[list[float]]:
    """Embed passages for indexing with the default model."""
    return get_embedder().embed_passages(texts)


def embed_query(text: str) -> list[float]:
    """Embed a query for search with the default model, prefix included."""
    return get_embedder().embed_query(text)
