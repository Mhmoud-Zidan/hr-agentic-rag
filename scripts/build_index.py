#!/usr/bin/env python3
"""Build the Chroma index from the policy corpus.

Parse data/policies/ -> chunk -> embed in batches -> write to ./chroma_db.

Idempotent: the collection is reset at the start of every run, so a rebuild
always reflects the corpus exactly as it is on disk. Chunk ids are deterministic
(sha1 of doc_id + heading_path + chunk_index), so two runs over an unchanged
corpus produce a byte-identical index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --batch-size 64 --persist-dir ./chroma_db
    python scripts/build_index.py --dry-run     # stats only, no embedding
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.chunker import (  # noqa: E402
    DEFAULT_CONFIG,
    SPLIT_REASONS,
    Chunk,
    ChunkConfig,
    chunk_sections,
    get_tokenizer,
)
from app.config import EMBEDDING_MODEL, resolve_input_window  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.parsers import Section, parse_corpus  # noqa: E402
from app.vector_store import ChromaVectorStore  # noqa: E402

DEFAULT_POLICY_DIR = REPO_ROOT / "data" / "policies"
DEFAULT_PERSIST_DIR = REPO_ROOT / "chroma_db"
DEFAULT_BATCH_SIZE = 64


def percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile. No numpy dependency, no interpolation ambiguity."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
    return ordered[rank - 1]


def _rule(char: str = "-", width: int = 68) -> str:
    return char * width


def report_corpus(sections: list[Section], chunks: list[Chunk], config: ChunkConfig) -> int:
    """Print the build report. Returns the number of chunks at the hard cap."""
    docs = {s.doc_id: s.doc_title for s in sections}
    tokens = [c.token_count for c in chunks]

    print(_rule("="))
    print("CORPUS")
    print(_rule("="))
    print(f"  documents : {len(docs)}")
    print(f"  sections  : {len(sections)}")
    print(f"  chunks    : {len(chunks)}")

    if not chunks:
        print("\n  nothing to index.")
        return 0

    print()
    print(_rule("="))
    print("TOKEN DISTRIBUTION")
    print(_rule("="))
    print(f"  min    : {min(tokens)}")
    print(f"  median : {int(statistics.median(tokens))}")
    print(f"  p95    : {percentile(tokens, 95)}")
    print(f"  max    : {max(tokens)}")
    print(f"  mean   : {int(statistics.fmean(tokens))}")
    print(f"  target {config.target_tokens} | hard cap {config.max_tokens} | floor {config.min_tokens}")

    # Anything within a hair of the cap is worth eyeballing: it means the
    # splitter had no good boundary and packed a window right up to the limit.
    at_cap = [c for c in chunks if c.token_count >= config.max_tokens - 10]
    over_cap = [c for c in chunks if c.token_count > config.max_tokens]
    under_floor = [c for c in chunks if c.token_count < config.min_tokens]

    print()
    print(_rule("="))
    print("CAP CHECK")
    print(_rule("="))
    if over_cap:
        print(f"  !! {len(over_cap)} chunk(s) EXCEED the hard cap of {config.max_tokens}:")
        for c in over_cap:
            print(f"     {c.token_count:>4}  {c.chunk_id}  {c.heading_path}")
    else:
        print(f"  no chunk exceeds the hard cap of {config.max_tokens}.")

    if at_cap:
        print(f"  {len(at_cap)} chunk(s) within 10 tokens of the cap:")
        for c in at_cap:
            print(f"     {c.token_count:>4}  {c.chunk_id}  {c.heading_path}")
    else:
        print(f"  no chunk is within 10 tokens of the cap.")

    if under_floor:
        print(f"  {len(under_floor)} chunk(s) below the {config.min_tokens}-token floor:")
        for c in under_floor:
            print(f"     {c.token_count:>4}  {c.chunk_id}  {c.heading_path}")

    print()
    print(_rule("="))
    print("PER-DOCUMENT")
    print(_rule("="))
    per_doc_chunks = Counter(c.doc_id for c in chunks)
    per_doc_sections = Counter(s.doc_id for s in sections)
    per_doc_tokens: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        per_doc_tokens[c.doc_id].append(c.token_count)

    print(f"  {'doc':<9} {'sections':>8} {'chunks':>7} {'tokens':>8}  {'fmt':<9} title")
    print(f"  {_rule('-', 62)}")
    fmt_by_doc = {s.doc_id: s.source_format for s in sections}
    for doc_id in sorted(per_doc_chunks):
        print(
            f"  {doc_id:<9} {per_doc_sections[doc_id]:>8} {per_doc_chunks[doc_id]:>7} "
            f"{sum(per_doc_tokens[doc_id]):>8}  {fmt_by_doc.get(doc_id, '?'):<9} {docs.get(doc_id, '')}"
        )

    return len(at_cap)


def assert_models_agree(chunk_config: ChunkConfig, embed_model: str) -> None:
    """Fail loudly if chunking and embedding are not using the same model.

    Token budgets are computed with the chunker's tokenizer and then spent by
    the embedder. If those are two different models, every chunk is measured
    against the wrong vocabulary: nothing raises, nothing looks wrong in the
    output, and chunks silently overflow the real window at embed time. This
    assertion is permanent on purpose -- it is the only place the mismatch is
    cheap to catch.
    """
    chunk_model = chunk_config.embedding_model
    if chunk_model != embed_model:
        raise SystemExit(
            "\nFATAL: model mismatch between chunking and embedding.\n"
            f"  chunker tokenizer : {chunk_model}\n"
            f"  embedder model    : {embed_model}\n"
            "Token counts would be measured with one model's vocabulary and spent "
            "by another's. Set EMBEDDING_MODEL (app/config.py) so both agree, or "
            "pass a --model that matches the chunker's tokenizer."
        )


def check_cap_against_window(chunk_config: ChunkConfig, model_name: str) -> int | None:
    """Warn when the hard cap exceeds the model's real input window.

    The window is derived at runtime from the model itself, not from a table.
    """
    window = resolve_input_window(model_name)
    if window is None:
        print(
            f"  window     : UNKNOWN for {model_name} -- could not derive a max "
            "sequence length. The cap check is NOT being enforced; verify manually."
        )
        return None

    print(f"  window     : {window} tokens (derived)")
    if chunk_config.max_tokens > window:
        print(
            f"  !! WARNING: hard cap {chunk_config.max_tokens} exceeds the "
            f"{window}-token input window of {model_name}.\n"
            "     Chunks above the window are truncated at embed time and their "
            "tails become unsearchable."
        )
    return window


def report_chunks(chunks: list[Chunk], config: ChunkConfig) -> None:
    """Per-chunk debug dump plus a histogram of why each chunk exists.

    The split reason is the fastest way to read a chunking regression: a corpus
    that is mostly `window_split` means the splitter is not finding heading
    boundaries, and a corpus that is mostly `merged` means the floor is too high
    for the document's subsection sizes.
    """
    print()
    print(_rule("="))
    print("CHUNK DETAIL")
    print(_rule("="))
    print(f"  {'idx':>4} {'tokens':>7} {'lvs':>4}  {'split reason':<13} heading_path")
    print(f"  {_rule('-', 64)}")

    current_doc: str | None = None
    for c in chunks:
        if c.doc_id != current_doc:
            if current_doc is not None:
                print()
            current_doc = c.doc_id
        flag = "!" if c.token_count > config.max_tokens else " "
        print(
            f" {flag}{c.chunk_index:>4} {c.token_count:>7} {c.leaf_count:>4}  "
            f"{c.split_reason:<13} {c.heading_path}"
        )
        if c.leaf_count > 1:
            print(f"         covers: {', '.join(c.sub_headings)}")

    counts = Counter(c.split_reason for c in chunks)
    total = len(chunks)

    print()
    print(_rule("="))
    print("SPLIT REASON HISTOGRAM")
    print(_rule("="))
    width = 34
    for reason in SPLIT_REASONS:
        n = counts.get(reason, 0)
        share = n / total if total else 0.0
        bar = "#" * int(round(share * width))
        print(f"  {reason:<13} {n:>4}  {share * 100:>5.1f}%  {bar}")

    unknown = set(counts) - set(SPLIT_REASONS)
    for reason in sorted(unknown):
        print(f"  {reason:<13} {counts[reason]:>4}  (unrecognised reason)")

    # Anything still under the floor survived both merge passes, meaning it had
    # no neighbour it could join without breaking the cap. Worth seeing by name.
    under_floor = [c for c in chunks if c.token_count < config.min_tokens]
    print()
    print(f"  under the {config.min_tokens}-token floor : {len(under_floor)}")
    for c in under_floor:
        print(f"     {c.token_count:>4}  idx {c.chunk_index:<4} {c.heading_path}")

    # Leaves per chunk says whether packing is doing anything: 1.0 means every
    # H3 subsection still became its own chunk.
    leaves = [c.leaf_count for c in chunks]
    leaf_hist = Counter(leaves)
    print()
    print(f"  leaves per chunk : {statistics.fmean(leaves):.2f} mean, "
          f"{max(leaves)} max, {sum(leaves)} leaves total")
    for count in sorted(leaf_hist):
        n = leaf_hist[count]
        print(f"     {count} leaf/chunk : {n:>4}  {'#' * int(round(n / total * 34))}")

    # Mean chunks per section says whether heading-aware splitting is doing any
    # work at all: 1.0 means every section fit whole.
    by_section = Counter((c.doc_id, c.section_number) for c in chunks)
    if by_section:
        mean = total / len(by_section)
        worst = max(by_section.items(), key=lambda kv: kv[1])
        print()
        print(f"  sections chunked : {len(by_section)}")
        print(f"  chunks/section   : {mean:.1f} mean, {worst[1]} max "
              f"({worst[0][0]} §{worst[0][1]})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Chroma index from data/policies/.")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default="northwind_policies")
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--debug-chunks",
        action="store_true",
        help="print every chunk with its token count and split reason, plus a histogram",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse, chunk and report, but do not embed or write to Chroma",
    )
    args = parser.parse_args(argv)

    if not args.policy_dir.is_dir():
        print(f"error: no such directory: {args.policy_dir}", file=sys.stderr)
        return 1

    started = time.perf_counter()

    print(f"\nparsing   {args.policy_dir} ...")
    sections = parse_corpus(args.policy_dir)
    if not sections:
        print("error: no documents parsed; nothing to index.", file=sys.stderr)
        return 1

    print(f"chunking  target={DEFAULT_CONFIG.target_tokens} cap={DEFAULT_CONFIG.max_tokens} "
          f"overlap={DEFAULT_CONFIG.overlap_tokens} floor={DEFAULT_CONFIG.min_tokens} ...")
    chunks = chunk_sections(sections, DEFAULT_CONFIG)

    print()
    report_corpus(sections, chunks, DEFAULT_CONFIG)

    if args.debug_chunks:
        report_chunks(chunks, DEFAULT_CONFIG)

    if args.dry_run:
        print(f"\ndry run: nothing written. ({time.perf_counter() - started:.1f}s)")
        return 0

    # Duplicate ids would mean two chunks silently collapsing into one row.
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        print(f"\nerror: duplicate chunk_ids: {dupes}", file=sys.stderr)
        return 1

    print()
    print(_rule("="))
    print("INDEXING")
    print(_rule("="))
    print(f"  model      : {args.model}")
    print(f"  persist to : {args.persist_dir}")
    print(f"  collection : {args.collection}")

    # Permanent guard: the tokenizer that measured these chunks and the model
    # about to embed them must be the same string. See assert_models_agree.
    tokenizer = get_tokenizer(DEFAULT_CONFIG.embedding_model)
    assert_models_agree(DEFAULT_CONFIG, args.model)
    print(f"  tokenizer  : {tokenizer.model_name} (matches embedder)")
    check_cap_against_window(DEFAULT_CONFIG, args.model)

    embedder = Embedder(args.model, batch_size=args.batch_size)
    store = ChromaVectorStore(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedder=embedder,
    )

    # Idempotence: wipe the collection so a rebuild can never leave behind rows
    # for chunks that no longer exist. Deterministic ids make the rest a no-op.
    store.reset()
    print(f"  space      : {store.space}")
    print(f"  reset      : collection emptied (count={store.count()})")
    print()

    written = 0
    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    for batch_no, start in enumerate(range(0, len(chunks), args.batch_size), start=1):
        batch = chunks[start : start + args.batch_size]
        t0 = time.perf_counter()
        written += store.add(batch)
        print(
            f"  batch {batch_no}/{total_batches}: {len(batch):>3} chunks "
            f"({time.perf_counter() - t0:.1f}s)  total={written}"
        )

    final = store.count()
    elapsed = time.perf_counter() - started

    print()
    print(_rule("="))
    print("DONE")
    print(_rule("="))
    print(f"  chunks written   : {written}")
    print(f"  collection count : {final}")
    print(f"  elapsed          : {elapsed:.1f}s")

    if final != len(chunks):
        print(
            f"\nerror: collection holds {final} rows but {len(chunks)} chunks were produced.",
            file=sys.stderr,
        )
        return 1

    print(f"\nindex ready at {args.persist_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
