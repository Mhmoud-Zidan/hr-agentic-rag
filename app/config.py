"""Single source of truth for model configuration.

Every module that needs to know which embedding model is in play imports
`EMBEDDING_MODEL` from here. Nothing else defines it, and nothing else reads
the environment variable. Two modules each with their own default is how you
end up chunking against one tokenizer and embedding with another -- a mismatch
that produces no error, no warning, and quietly wrong token budgets.

`resolve_input_window()` replaces what used to be a hardcoded dict of model
name -> max tokens. That table was stale by construction: it only knew the
models someone had thought to add, and it silently returned None (i.e. "no
limit, skip the check") for anything else.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_MODEL",
    "CHROMA_DIR",
    "CHROMA_COLLECTION",
    "resolve_input_window",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: The embedding model for the whole pipeline: chunk token counting, passage
#: embedding and query embedding all follow this one string.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

#: Where the built index lives. Writer (scripts/build_index.py) and readers
#: (the MCP server, the agent) must agree on this: a reader pointed at the wrong
#: directory does not fail, it silently opens an empty collection and every
#: search returns nothing. That failure looks like a bad retriever, not a bad
#: path, which is why it is defined once here rather than defaulted per module.
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", REPO_ROOT / "chroma_db"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "northwind_policies")


# Where a model's real input window is published, most authoritative first.
#
# IMPORTANT: the loaded `tokenizers.Tokenizer` object cannot answer this on its
# own, which is why this function consults the model repo rather than just the
# tokenizer handed to it. Measured on the two models in play:
#
#                                     bge-small-en-v1.5   all-MiniLM-L6-v2
#   tokenizer.truncation.max_length         None                128
#   config.json max_position_embeddings      512                512
#   tokenizer_config.json model_max_length   512                512
#   sentence_bert_config.json max_seq_length 512                256   <- correct
#
# MiniLM's true window is 256. Its tokenizer advertises 128 (a truncation
# default, not a model limit) and its config advertises 512 (the positional
# embedding table, which sentence-transformers caps below). Only
# sentence_bert_config.json is right for both. The others are kept as fallbacks
# for models that do not ship it, ordered so the least trustworthy is last.
_WINDOW_SOURCES: tuple[tuple[str, str], ...] = (
    ("sentence_bert_config.json", "max_seq_length"),
    ("tokenizer_config.json", "model_max_length"),
    ("config.json", "max_position_embeddings"),
)

# tokenizer_config.json sometimes carries a sentinel meaning "unbounded".
_IMPLAUSIBLE_WINDOW = 100_000


def _read_hub_json(model_name: str, filename: str) -> dict[str, Any] | None:
    """Fetch one JSON file from the model repo. Cached locally by huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, filename)
    except Exception:
        return None

    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


@lru_cache(maxsize=8)
def resolve_input_window(model_name: str = EMBEDDING_MODEL) -> int | None:
    """The model's real maximum sequence length in tokens, derived at runtime.

    Returns None only when every source fails -- typically offline with nothing
    cached. Callers must treat None as "unknown", never as "unlimited".
    """
    for filename, key in _WINDOW_SOURCES:
        payload = _read_hub_json(model_name, filename)
        if not payload:
            continue
        value = payload.get(key)
        if isinstance(value, int) and 0 < value < _IMPLAUSIBLE_WINDOW:
            return value
    return None


def window_from_tokenizer(tokenizer: Any) -> int | None:
    """Last-resort window hint off a live tokenizer object.

    Only meaningful when the tokenizer ships a truncation config, and even then
    it reports a truncation default rather than the model's window -- MiniLM
    says 128 against a real 256. Used only when `resolve_input_window` returns
    None, and deliberately not used to override it.
    """
    truncation = getattr(tokenizer, "truncation", None)
    if isinstance(truncation, dict):
        value = truncation.get("max_length")
        if isinstance(value, int) and 0 < value < _IMPLAUSIBLE_WINDOW:
            return value
    return None
