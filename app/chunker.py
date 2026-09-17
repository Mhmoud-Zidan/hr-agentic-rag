"""Heading-aware chunker for the Northwind policy corpus.

Input: `list[Section]` from app.parsers. Output: `list[Chunk]`, ready to embed.

Policy:
  * H2 is the primary unit. A section that fits the cap becomes exactly one chunk.
  * H3 is the fallback. Only an oversized H2 is split at its numbered subsections.
  * A sliding window is used only when a single leaf still exceeds the cap.
  * Leaves below the minimum merge into the next sibling within the same section.

Every chunk's text is `heading_path + "\\n\\n" + body`, and that full string is what
gets measured and what must be embedded. Measuring the body alone would let the
breadcrumb push a chunk past the model's window at embed time.

Token counts come from the embedding model's own tokenizer. Whitespace splitting is
never used, not even as a fallback: if the tokenizer cannot be loaded the chunker
raises, because a wrong count here silently corrupts every downstream guarantee.

Fully deterministic: same Sections in, byte-identical Chunks out. No randomness, no
dict-ordering dependence, no clock, no network at chunk time.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from app.config import EMBEDDING_MODEL, resolve_input_window, window_from_tokenizer
from app.parsers import Section

__all__ = [
    "Chunk",
    "ChunkConfig",
    "SPLIT_REASONS",
    "DEFAULT_CONFIG",
    "Tokenizer",
    "chunk_section",
    "chunk_sections",
    "get_tokenizer",
]

# The tokenizer that decides what a "token" is here MUST be the one belonging to
# the model that will embed these chunks. app.config owns that choice for the
# whole pipeline; nothing in this module defines or reads it independently.
# Its real input window is derived at runtime (config.resolve_input_window), not
# looked up in a table -- a table only knows the models someone remembered to add.
BREADCRUMB_SEPARATOR = "\n\n"

# Why a chunk exists, for build-time diagnostics. Diagnostic only: these are NOT
# part of the Chroma metadata record, so a reason can never change what is
# indexed or how a chunk is retrieved.
REASON_H2_WHOLE = "h2_whole"  # the H2 fit the cap; one section, one chunk
REASON_H3_SPLIT = "h3_split"  # oversized H2, one leaf that filled a chunk alone
REASON_PACKED = "packed"  # several consecutive H3 leaves packed up to the cap
REASON_WINDOW_SPLIT = "window_split"  # a single leaf still overflowed; windowed
REASON_MERGED = "merged"  # below the floor, folded into a sibling

SPLIT_REASONS = (
    REASON_H2_WHOLE,
    REASON_H3_SPLIT,
    REASON_PACKED,
    REASON_WINDOW_SPLIT,
    REASON_MERGED,
)

# Chroma metadata values must be primitives, so the list of H3 headings a packed
# chunk covers is stored as a delimited string rather than a list.
SUB_HEADING_SEPARATOR = " | "

# A leaf boundary: a numbered H3, in either source format.
# Markdown keeps the marker ("### 3.1 Notice periods"); the HTML parser flattens
# the <h3> to plain text ("3.1 Detail"). One pattern covers both so that the two
# formats split at the same places.
_H3_BOUNDARY = re.compile(r"^(?:#{3,6}\s+)?(?P<number>\d+\.\d+(?:\.\d+)*)\.?\s+\S")

# Sentence-ish boundary used to build windows without ever decoding token ids
# back to text (WordPiece decoding is lossy and would mangle the citation text).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9(\[*_`\"'-])|\n{2,}")


class TokenizerUnavailable(RuntimeError):
    """Raised when the embedding model's tokenizer cannot be loaded."""


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 400
    max_tokens: int = 480
    overlap_tokens: int = 60
    min_tokens: int = 80
    embedding_model: str = EMBEDDING_MODEL

    def __post_init__(self) -> None:
        if not 0 < self.target_tokens <= self.max_tokens:
            raise ValueError("target_tokens must be > 0 and <= max_tokens")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be >= 0 and < target_tokens")
        if not 0 <= self.min_tokens <= self.target_tokens:
            raise ValueError("min_tokens must be >= 0 and <= target_tokens")


DEFAULT_CONFIG = ChunkConfig()


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit: breadcrumb + body, plus the metadata a citation needs."""

    chunk_id: str
    text: str
    token_count: int
    # --- metadata ---
    doc_id: str
    doc_title: str
    section_number: str
    section_heading: str
    heading_path: str
    source_filename: str
    source_format: str
    chunk_index: int
    char_start: int
    char_end: int
    #: Diagnostic only, deliberately absent from metadata(). See SPLIT_REASONS.
    split_reason: str = REASON_H2_WHOLE
    #: The H3 headings this chunk covers, in document order. Empty when the
    #: section has no numbered subsections.
    sub_headings: tuple[str, ...] = ()
    #: How many leaves were packed into this chunk. Not derivable from
    #: sub_headings: a section's intro prose is a leaf with no H3 heading.
    leaf_count: int = 1

    def metadata(self) -> dict[str, str | int]:
        """Flat, primitive-only mapping — what Chroma accepts as a metadata record."""
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section_number": self.section_number,
            "section_heading": self.section_heading,
            "heading_path": self.heading_path,
            "source_filename": self.source_filename,
            "source_format": self.source_format,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            # Delimited, not a list: Chroma rejects non-primitive metadata.
            "sub_headings": SUB_HEADING_SEPARATOR.join(self.sub_headings),
        }


class Tokenizer:
    """Thin wrapper over the embedding model's tokenizer.

    Truncation is explicitly disabled. Some tokenizers ship with a truncation
    default baked in -- all-MiniLM-L6-v2 sets 128 -- and with it left in place
    `len(encode(text).ids)` saturates at that number, so every cap assertion
    passes vacuously while real chunks run long. bge-small-en-v1.5 ships no
    truncation config at all, so this is a property of the tokenizer rather than
    of any one model, and is disabled unconditionally.

    Special tokens are counted because the model spends its window on them.
    `input_window` is the model's real maximum sequence length, derived at
    runtime by app.config rather than read from a table.
    """

    def __init__(self, model_name: str):
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TokenizerUnavailable(
                "the `tokenizers` package is required to count tokens; "
                "whitespace splitting is not an acceptable substitute"
            ) from exc

        try:
            self._tok = HFTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise TokenizerUnavailable(
                f"could not load the tokenizer for {model_name!r}: {exc}. "
                "Chunking refuses to fall back to whitespace counting."
            ) from exc

        # Read any truncation hint before clearing it; it is the last-resort
        # fallback for the input window.
        truncation_hint = window_from_tokenizer(self._tok)

        self._tok.no_truncation()
        self._tok.no_padding()
        self.model_name = model_name
        #: Real maximum sequence length, or None when it cannot be determined.
        #: None means UNKNOWN, never unlimited -- callers must not skip cap
        #: checks on it silently.
        self.input_window = resolve_input_window(model_name) or truncation_hint

    def count(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str = EMBEDDING_MODEL) -> Tokenizer:
    """Cached per model name. Loading is the only slow part of chunking."""
    return Tokenizer(model_name)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


@dataclass
class _Leaf:
    """One H3 subsection (or a whole section that has none), before packing."""

    text: str
    tokens: int  # measured WITH the breadcrumb, as the chunk will be
    reason: str = REASON_H3_SPLIT
    heading: str | None = None


@dataclass
class _Piece:
    """A chunk body after packing, carrying what it covers and how it got there."""

    text: str
    reason: str
    headings: tuple[str, ...] = ()
    leaves: int = 1


def _decorate(heading_path: str, body: str) -> str:
    return f"{heading_path}{BREADCRUMB_SEPARATOR}{body}".strip()


def _split_into_leaves(text: str) -> list[str]:
    """Split a section body at numbered H3 boundaries. Returns [text] if there are none."""
    lines = text.splitlines()
    boundaries = [i for i, line in enumerate(lines) if _H3_BOUNDARY.match(line)]
    if not boundaries:
        return [text]

    cuts = boundaries if boundaries[0] == 0 else [0, *boundaries]
    leaves: list[str] = []
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else len(lines)
        piece = "\n".join(lines[start:end]).strip()
        if piece:
            leaves.append(piece)
    return leaves


def _leaf_heading(text: str) -> str | None:
    """The H3 heading a leaf opens with, cleaned of its Markdown marker."""
    first = text.splitlines()[0] if text else ""
    if not _H3_BOUNDARY.match(first):
        return None
    return first.lstrip("#").strip()


def _pack_leaves(
    leaves: list[_Leaf], heading_path: str, tok: Tokenizer, cfg: ChunkConfig
) -> list[_Piece]:
    """Greedily pack consecutive H3 leaves up to the hard cap.

    Without this, every leaf became its own chunk and chunk size was dictated by
    however long the author's subsections happened to be -- on PTO-01 that meant
    a median of 148 tokens against a 400 target, because `target_tokens` was only
    ever consulted when windowing an oversized leaf.

    Rules:
      * accumulate consecutive leaves into a buffer;
      * flush when adding the next leaf would exceed the cap;
      * always flush at the H2 boundary, so a chunk never spans two sections
        (the breadcrumb names one H2, and it must be true);
      * a leaf already over the cap bypasses packing entirely and is handed to
        window splitting, exactly as before.

    The budget is measured on the DECORATED text. Packing to the cap on the body
    alone would push every packed chunk over the cap once the breadcrumb is
    prepended, which is the bug this comment exists to prevent.
    """
    groups: list[list[_Leaf]] = []
    oversized: set[int] = set()  # indices of groups that bypassed packing
    buffer: list[_Leaf] = []

    def group_tokens(group: list[_Leaf]) -> int:
        return tok.count(_decorate(heading_path, "\n\n".join(l.text for l in group)))

    def flush() -> None:
        if buffer:
            groups.append(list(buffer))
            buffer.clear()

    for leaf in leaves:
        if leaf.tokens > cfg.max_tokens:
            # Oversized on its own: nothing can pack with it. Flush what we have
            # so document order is preserved, then pass it through for windowing.
            flush()
            oversized.add(len(groups))
            groups.append([leaf])
            continue

        if buffer:
            trial = "\n\n".join([*(l.text for l in buffer), leaf.text])
            if tok.count(_decorate(heading_path, trial)) > cfg.max_tokens:
                flush()

        buffer.append(leaf)

    flush()

    # Rebalance a stranded tail. Greedy packing fills early chunks to the cap,
    # which can leave the section's last group holding one short subsection that
    # no longer fits anywhere -- PTO-01 §5.4 lands at 71 tokens this way. Pull
    # leaves back from the previous group while both sides stay legal.
    if len(groups) >= 2 and cfg.min_tokens > 0:
        last = len(groups) - 1
        prev = last - 1
        while (
            last not in oversized
            and prev not in oversized
            and len(groups[prev]) > 1
            and group_tokens(groups[last]) < cfg.min_tokens
        ):
            candidate_prev = groups[prev][:-1]
            candidate_last = [groups[prev][-1], *groups[last]]
            if (
                group_tokens(candidate_last) > cfg.max_tokens
                or group_tokens(candidate_prev) < cfg.min_tokens
            ):
                break
            groups[prev], groups[last] = candidate_prev, candidate_last

    packed: list[_Piece] = []
    for index, group in enumerate(groups):
        text = "\n\n".join(leaf.text for leaf in group)
        headings = tuple(leaf.heading for leaf in group if leaf.heading)
        if index in oversized:
            reason = REASON_WINDOW_SPLIT
        elif len(group) > 1:
            reason = REASON_PACKED
        else:
            reason = group[0].reason
        packed.append(_Piece(text, reason, headings, len(group)))
    return packed


def _split_sentences(text: str) -> list[str]:
    """Split into sentence-ish units, preserving the original characters."""
    units = [u.strip() for u in _SENTENCE_SPLIT.split(text) if u and u.strip()]
    return units or ([text.strip()] if text.strip() else [])


def _window(
    body: str,
    heading_path: str,
    tok: Tokenizer,
    cfg: ChunkConfig,
) -> list[str]:
    """Sliding window over a single oversized leaf, with `overlap_tokens` of carry-back.

    Windows are built from whole sentences rather than raw token ids: decoding
    WordPiece ids back to text is lossy (casing and punctuation are destroyed),
    and these strings are shown to users as citations.
    """
    units = _split_sentences(body)
    if not units:
        return []

    # Any single unit that alone blows the cap is split on whitespace runs as a
    # last resort. Rare, but it must terminate rather than emit an oversized chunk.
    budget = cfg.max_tokens - tok.count(_decorate(heading_path, "")) - 1
    normalized: list[str] = []
    for unit in units:
        if tok.count(unit) <= budget:
            normalized.append(unit)
            continue
        words = unit.split()
        buf: list[str] = []
        for word in words:
            trial = " ".join([*buf, word])
            if buf and tok.count(trial) > budget:
                normalized.append(" ".join(buf))
                buf = [word]
            else:
                buf.append(word)
        if buf:
            normalized.append(" ".join(buf))

    counts = [tok.count(u) for u in normalized]

    windows: list[str] = []
    i = 0
    n = len(normalized)
    while i < n:
        selected: list[int] = []
        running = 0
        j = i
        while j < n:
            candidate = running + counts[j]
            decorated = tok.count(_decorate(heading_path, " ".join(normalized[i : j + 1])))
            if selected and (candidate > cfg.target_tokens or decorated > cfg.max_tokens):
                break
            selected.append(j)
            running = candidate
            j += 1

        windows.append(" ".join(normalized[i : selected[-1] + 1]))

        if selected[-1] + 1 >= n:
            break

        # Step back far enough to carry ~overlap_tokens of context forward.
        back = 0
        k = selected[-1]
        while k > i and back + counts[k] <= cfg.overlap_tokens:
            back += counts[k]
            k -= 1
        next_i = max(k + 1, i + 1)  # always make progress
        i = next_i

    return windows


def _enforce_floor(
    bodies: list[_Piece], heading_path: str, tok: Tokenizer, cfg: ChunkConfig
) -> list[_Piece]:
    """Final floor pass over emitted bodies, after packing and windowing.

    Packing already absorbs most short subsections, but it cannot catch a tiny
    leaf stranded next to an oversized one: that neighbour bypasses packing, so
    the tiny leaf flushes alone. Once the oversized leaf has been windowed, the
    tail window usually has room. Runs last for that reason.
    """
    if len(bodies) <= 1 or cfg.min_tokens <= 0:
        return bodies

    result = list(bodies)
    i = 0
    while i < len(result):
        if tok.count(_decorate(heading_path, result[i].text)) >= cfg.min_tokens:
            i += 1
            continue

        merged = False
        for neighbour in (i + 1, i - 1):  # next sibling first, then previous
            if not 0 <= neighbour < len(result):
                continue
            lo, hi = sorted((i, neighbour))
            combined = f"{result[lo].text}\n\n{result[hi].text}"
            if tok.count(_decorate(heading_path, combined)) <= cfg.max_tokens:
                result[lo : hi + 1] = [
                    _Piece(
                        combined,
                        REASON_MERGED,
                        result[lo].headings + result[hi].headings,
                        result[lo].leaves + result[hi].leaves,
                    )
                ]
                i = max(lo - 1, 0)
                merged = True
                break

        if not merged:
            # Nothing it can merge with without breaking the cap. Keeping a short
            # chunk is better than dropping text or emitting an oversized one.
            i += 1

    return result


def _chunk_id(doc_id: str, heading_path: str, index: int) -> str:
    return hashlib.sha1(f"{doc_id}{heading_path}{index}".encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def chunk_section(
    section: Section,
    config: ChunkConfig = DEFAULT_CONFIG,
    tokenizer: Tokenizer | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """Chunk one Section. `start_index` continues a document-scoped chunk_index."""
    tok = tokenizer or get_tokenizer(config.embedding_model)
    heading_path = section.heading_path
    body = section.text.strip()

    if not body:
        return []

    whole = _decorate(heading_path, body)
    whole_tokens = tok.count(whole)

    if whole_tokens <= config.max_tokens:
        # The whole section fits. It still covers its H3s, so record them.
        bodies = [
            _Piece(
                body,
                REASON_H2_WHOLE,
                tuple(
                    h
                    for h in (_leaf_heading(p) for p in _split_into_leaves(body))
                    if h
                ),
            )
        ]
    else:
        pieces = _split_into_leaves(body)
        # A section with no H3s yields one leaf covering the whole body; nothing
        # was cut at a heading, so it is not an h3_split.
        leaf_reason = REASON_H3_SPLIT if len(pieces) > 1 else REASON_WINDOW_SPLIT
        leaves = [
            _Leaf(
                p,
                tok.count(_decorate(heading_path, p)),
                leaf_reason,
                _leaf_heading(p),
            )
            for p in pieces
        ]

        # Pack before windowing: packing decides chunk size, windowing only
        # rescues leaves that are oversized on their own.
        bodies = []
        for piece in _pack_leaves(leaves, heading_path, tok, config):
            if tok.count(_decorate(heading_path, piece.text)) <= config.max_tokens:
                bodies.append(piece)
            else:
                bodies.extend(
                    _Piece(w, REASON_WINDOW_SPLIT, piece.headings, piece.leaves)
                    for w in _window(piece.text, heading_path, tok, config)
                )

        bodies = _enforce_floor(bodies, heading_path, tok, config)

    chunks: list[Chunk] = []
    for offset, piece in enumerate(bodies):
        index = start_index + offset
        text = _decorate(heading_path, piece.text)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(section.doc_id, heading_path, index),
                text=text,
                token_count=tok.count(text),
                doc_id=section.doc_id,
                doc_title=section.doc_title,
                section_number=section.number,
                section_heading=section.heading,
                heading_path=heading_path,
                source_filename=section.source_filename,
                source_format=section.source_format,
                chunk_index=index,
                # Offsets locate the parent SECTION in the source file. The parser
                # normalizes section text, so a sub-section chunk cannot be mapped
                # back to exact source characters without re-reading the file.
                char_start=section.char_start,
                char_end=section.char_end,
                split_reason=piece.reason,
                sub_headings=piece.headings,
                leaf_count=piece.leaves,
            )
        )
    return chunks


def chunk_sections(
    sections: Iterable[Section],
    config: ChunkConfig = DEFAULT_CONFIG,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Chunk a corpus. `chunk_index` restarts at 0 for each document."""
    tok = tokenizer or get_tokenizer(config.embedding_model)

    if tok.input_window and config.max_tokens > tok.input_window:
        warnings.warn(
            f"max_tokens={config.max_tokens} exceeds the {tok.model_name} input window "
            f"of {tok.input_window} tokens; chunks at the cap will be truncated at "
            f"embed time and their tails will be unsearchable.",
            RuntimeWarning,
            stacklevel=2,
        )

    chunks: list[Chunk] = []
    per_doc: dict[str, int] = {}
    for section in sections:
        index = per_doc.get(section.doc_id, 0)
        produced = chunk_section(section, config, tok, start_index=index)
        per_doc[section.doc_id] = index + len(produced)
        chunks.extend(produced)
    return chunks
