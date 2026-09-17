"""Tests for app.chunker.

The three required guarantees — determinism, the hard cap, and the breadcrumb —
plus the rules that make those guarantees mean something: real tokenizer counts,
H2-primary/H3-fallback, windowing only on oversized leaves, and the minimum.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from app.chunker import (
    BREADCRUMB_SEPARATOR,
    REASON_PACKED,
    SUB_HEADING_SEPARATOR,
    DEFAULT_CONFIG,
    Chunk,
    ChunkConfig,
    chunk_section,
    chunk_sections,
    get_tokenizer,
)
from app.parsers import Section, parse_markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
PTO_PATH = REPO_ROOT / "data" / "policies" / "pto_and_leave_policy.md"

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer(DEFAULT_CONFIG.embedding_model)


@pytest.fixture(scope="module")
def sections() -> list[Section]:
    return parse_markdown(PTO_PATH)


@pytest.fixture(scope="module")
def chunks(sections) -> list[Chunk]:
    return chunk_sections(sections)


def _section(text: str, *, number: str = "1", heading: str = "Test Section") -> Section:
    return Section(
        doc_id="PTO-01",
        doc_title="PTO and Leave Policy",
        number=number,
        heading=heading,
        heading_path=f"PTO and Leave Policy (PTO-01) > {number}. {heading}",
        text=text,
        source_filename="pto_and_leave_policy.md",
        source_format="markdown",
        char_start=0,
        char_end=len(text),
    )


# ---------------------------------------------------------------------------
# required: determinism
# ---------------------------------------------------------------------------


def test_determinism_across_two_runs(sections):
    first = chunk_sections(sections)
    second = chunk_sections(sections)

    assert len(first) == len(second)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
    assert [c.token_count for c in first] == [c.token_count for c in second]
    assert [c.metadata() for c in first] == [c.metadata() for c in second]


def test_determinism_from_a_fresh_parse():
    """Re-parsing from disk must not perturb chunk boundaries or ids."""
    a = chunk_sections(parse_markdown(PTO_PATH))
    b = chunk_sections(parse_markdown(PTO_PATH))
    assert [(c.chunk_id, c.text) for c in a] == [(c.chunk_id, c.text) for c in b]


def test_chunk_ids_are_unique_and_match_the_documented_formula(chunks):
    import hashlib

    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        expected = hashlib.sha1(
            f"{c.doc_id}{c.heading_path}{c.chunk_index}".encode("utf-8")
        ).hexdigest()[:16]
        assert c.chunk_id == expected
        assert len(c.chunk_id) == 16


# ---------------------------------------------------------------------------
# required: hard cap
# ---------------------------------------------------------------------------


def test_no_chunk_exceeds_the_cap(chunks, tok):
    for c in chunks:
        assert c.token_count <= DEFAULT_CONFIG.max_tokens, (
            f"{c.chunk_id} §{c.section_number} is {c.token_count} tokens"
        )


def test_reported_token_count_matches_a_fresh_tokenization(chunks, tok):
    """token_count must describe the text that will actually be embedded."""
    for c in chunks:
        assert c.token_count == tok.count(c.text)


def test_cap_holds_on_pathological_input():
    """One unbroken 5,000-word paragraph: no sentence boundaries to split on."""
    section = _section("word " * 5000)
    produced = chunk_section(section)
    assert produced
    for c in produced:
        assert c.token_count <= DEFAULT_CONFIG.max_tokens


def test_tokenizer_is_not_whitespace_counting(tok):
    """WordPiece splits 'PTO' and '(PTO-01)'; a whitespace count would not."""
    text = "PTO and Leave Policy (PTO-01) > 4. Carryover and Expiration"
    assert tok.count(text) > len(text.split())


def test_tokenizer_truncation_is_disabled(tok):
    """MiniLM ships truncating at 128; if that leaked in, every cap test is vacuous."""
    assert tok.count("word " * 600) > 500


# ---------------------------------------------------------------------------
# required: breadcrumb
# ---------------------------------------------------------------------------


def test_every_chunk_begins_with_its_breadcrumb(chunks):
    for c in chunks:
        assert c.text.startswith(c.heading_path + BREADCRUMB_SEPARATOR), (
            f"{c.chunk_id} does not open with its heading_path"
        )


def test_breadcrumb_is_followed_by_a_blank_line_then_body(chunks):
    for c in chunks:
        remainder = c.text[len(c.heading_path) :]
        assert remainder.startswith("\n\n")
        assert remainder[2:].strip(), "chunk has a breadcrumb but no body"


def test_breadcrumb_appears_exactly_once(chunks):
    for c in chunks:
        assert c.text.count(c.heading_path) == 1


def test_breadcrumb_is_inside_the_measured_text(tok):
    """The cap must be enforced on breadcrumb+body, not body alone."""
    section = _section("word " * 5000)
    for c in chunk_section(section):
        assert tok.count(c.text) == c.token_count
        assert c.token_count > tok.count(c.text[len(c.heading_path) :])


# ---------------------------------------------------------------------------
# chunking policy
# ---------------------------------------------------------------------------


def test_small_section_stays_whole_as_one_chunk():
    section = _section("A short policy statement about carryover. It fits easily.")
    produced = chunk_section(section)
    assert len(produced) == 1
    assert produced[0].chunk_index == 0
    assert produced[0].text.endswith("It fits easily.")


def test_h2_is_primary_no_split_when_it_fits(sections, tok):
    """Any section that fits the cap whole must emit exactly one chunk."""
    for s in sections:
        whole = tok.count(f"{s.heading_path}\n\n{s.text.strip()}")
        if whole <= DEFAULT_CONFIG.max_tokens:
            assert len(chunk_section(s)) == 1


def test_h3_fallback_splits_an_oversized_section_at_subsections(tok):
    body = "\n\n".join(
        f"### 1.{i} Subsection {i}\n\n" + ("Policy sentence number %d. " % i) * 60
        for i in range(1, 5)
    )
    produced = chunk_section(_section(body))
    assert len(produced) >= 2
    # Each chunk should open on a subsection heading, i.e. H3 boundaries were used.
    bodies = [c.text[len(c.heading_path) + 2 :] for c in produced]
    assert sum(b.startswith("### 1.") for b in bodies) >= 2


def test_window_only_used_when_a_leaf_exceeds_the_cap(tok):
    """A section with no H3s that overflows falls through to windowing."""
    body = " ".join(f"Sentence number {i} about leave policy." for i in range(200))
    produced = chunk_section(_section(body))
    assert len(produced) > 1
    for c in produced:
        assert c.token_count <= DEFAULT_CONFIG.max_tokens


def test_windows_overlap(tok):
    body = " ".join(f"Sentence number {i} about leave policy." for i in range(200))
    produced = chunk_section(_section(body))
    first_body = produced[0].text[len(produced[0].heading_path) + 2 :]
    second_body = produced[1].text[len(produced[1].heading_path) + 2 :]
    tail = first_body.split()[-8:]
    assert " ".join(tail) in second_body, "consecutive windows share no text"


def test_no_overlap_between_sibling_leaf_chunks():
    """Overlap applies to window splits only, never to H3 leaf boundaries."""
    body = "\n\n".join(
        f"### 2.{i} Heading {i}\n\nUnique marker phrase alpha{i}. " + ("Filler text. " * 40)
        for i in range(1, 4)
    )
    produced = chunk_section(_section(body, number="2"))
    markers = ["alpha1", "alpha2", "alpha3"]
    for marker in markers:
        hits = sum(marker in c.text for c in produced)
        assert hits == 1, f"{marker} appears in {hits} chunks; leaves must not overlap"


def test_minimum_token_floor_merges_small_leaves():
    """A tiny trailing subsection must not survive as its own chunk."""
    big = "### 3.1 Long subsection\n\n" + ("Substantial policy sentence. " * 120)
    tiny = "### 3.2 Tiny\n\nSee APR-01."
    produced = chunk_section(_section(f"{big}\n\n{tiny}", number="3"))
    assert all(
        c.token_count >= DEFAULT_CONFIG.min_tokens for c in produced
    ), [c.token_count for c in produced]
    assert any("3.2 Tiny" in c.text for c in produced), "the small leaf was dropped"


def test_real_corpus_respects_the_floor(chunks):
    below = [(c.chunk_id, c.token_count) for c in chunks if c.token_count < DEFAULT_CONFIG.min_tokens]
    assert not below, f"chunks under the {DEFAULT_CONFIG.min_tokens}-token floor: {below}"


def test_empty_section_produces_nothing():
    assert chunk_section(_section("   \n\n  ")) == []


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_metadata_carries_every_required_key(chunks):
    required = {
        "doc_id",
        "doc_title",
        "section_number",
        "section_heading",
        "heading_path",
        "source_filename",
        "source_format",
        "chunk_index",
        "char_start",
        "char_end",
        "sub_headings",
    }
    for c in chunks:
        md = c.metadata()
        assert set(md) == required
        assert all(isinstance(v, (str, int)) for v in md.values())


def test_metadata_is_inherited_from_the_parent_section(sections, chunks):
    by_path = {s.heading_path: s for s in sections}
    for c in chunks:
        s = by_path[c.heading_path]
        assert (c.doc_id, c.doc_title) == (s.doc_id, s.doc_title)
        assert (c.section_number, c.section_heading) == (s.number, s.heading)
        assert (c.source_filename, c.source_format) == (s.source_filename, s.source_format)
        assert (c.char_start, c.char_end) == (s.char_start, s.char_end)


def test_chunk_index_is_document_scoped_and_contiguous(chunks):
    per_doc: dict[str, list[int]] = {}
    for c in chunks:
        per_doc.setdefault(c.doc_id, []).append(c.chunk_index)
    for doc_id, indices in per_doc.items():
        assert indices == list(range(len(indices))), doc_id


def test_all_source_text_is_preserved_across_chunks(sections):
    """Chunking must not silently drop a subsection."""
    produced = chunk_sections(sections)
    joined = " ".join(" ".join(c.text.split()) for c in produced)
    for probe in (
        "1.67 days per month",
        "cannot be taken before day 91",
        "expiring March 31" if "expiring March 31" in joined else "expire on March 31",
        "capped at 10 days",
        "8 days of sick leave",
        "maximum of 90 days",
    ):
        assert probe in joined, f"lost from the corpus: {probe!r}"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_defaults_match_the_spec():
    assert (
        DEFAULT_CONFIG.target_tokens,
        DEFAULT_CONFIG.max_tokens,
        DEFAULT_CONFIG.overlap_tokens,
        DEFAULT_CONFIG.min_tokens,
    ) == (400, 480, 60, 80)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens": 0},
        {"target_tokens": 500, "max_tokens": 480},
        {"overlap_tokens": 400},
        {"overlap_tokens": -1},
        {"min_tokens": 500},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ChunkConfig(**kwargs)


def test_tighter_cap_still_holds(sections):
    cfg = ChunkConfig(target_tokens=200, max_tokens=240, overlap_tokens=40, min_tokens=50)
    for c in chunk_sections(sections, cfg):
        assert c.token_count <= cfg.max_tokens


# ---------------------------------------------------------------------------
# greedy packing
# ---------------------------------------------------------------------------


def test_no_chunk_spans_two_h2_sections(chunks, sections):
    """The breadcrumb names one H2. Packing must never make that a lie."""
    by_path = {s.heading_path: s for s in sections}
    for c in chunks:
        assert c.heading_path in by_path
        section = by_path[c.heading_path]
        body = c.text[len(c.heading_path) + 2 :]

        # No second H2 heading smuggled into the body.
        assert "\n## " not in body
        assert not body.startswith("## ")

        # Every sub-heading it claims belongs to this section's numbering.
        for heading in c.sub_headings:
            assert heading.split()[0].startswith(f"{section.number}."), (
                f"{c.chunk_id} claims {heading!r} but sits under §{section.number}"
            )


def test_median_token_count_is_within_20_percent_of_target(chunks):
    median = statistics.median(c.token_count for c in chunks)
    target = DEFAULT_CONFIG.target_tokens
    drift = abs(median - target) / target
    assert drift <= 0.20, (
        f"median {median} is {drift * 100:.1f}% from the {target}-token target; "
        "packing is not filling chunks"
    )


def test_packing_actually_happened(chunks):
    """Guards against the packer silently becoming a no-op."""
    packed = [c for c in chunks if c.split_reason == REASON_PACKED]
    assert packed, "no chunk was produced by packing"
    assert statistics.fmean(c.leaf_count for c in chunks) > 1.5


def test_packed_chunks_list_their_sub_headings(chunks):
    for c in chunks:
        if c.split_reason == REASON_PACKED:
            assert c.leaf_count > 1
            # A section's intro prose is a leaf with no H3 heading, so a packed
            # chunk can cover fewer headings than it holds leaves -- but never more.
            assert 1 <= len(c.sub_headings) <= c.leaf_count
            # Every listed heading appears in the body it claims to cover.
            for heading in c.sub_headings:
                assert heading in c.text


def test_sub_headings_serialise_as_a_primitive(chunks):
    for c in chunks:
        value = c.metadata()["sub_headings"]
        assert isinstance(value, str)
        if c.sub_headings:
            assert value.split(SUB_HEADING_SEPARATOR) == list(c.sub_headings)
        else:
            assert value == ""


def test_packing_never_exceeds_the_cap_or_drops_below_the_floor(chunks):
    for c in chunks:
        assert c.token_count <= DEFAULT_CONFIG.max_tokens
        assert c.token_count >= DEFAULT_CONFIG.min_tokens


def test_packing_preserves_document_order_and_loses_no_subsection(sections, chunks):
    """Every H3 heading in the corpus is covered by exactly one chunk."""
    import re

    expected: list[str] = []
    for s in sections:
        for line in s.text.splitlines():
            if line.startswith("### "):
                expected.append(line.lstrip("#").strip())

    covered = [h for c in chunks for h in c.sub_headings]
    assert covered == expected, "sub-headings lost, duplicated or reordered"


def test_oversized_leaf_still_bypasses_packing():
    """A leaf over the cap goes to windowing, not into a packed chunk."""
    huge = "### 1.1 Huge\n\n" + ("Policy sentence about leave. " * 200)
    small = "### 1.2 Small\n\n" + ("Short note. " * 30)
    produced = chunk_section(_section(f"{huge}\n\n{small}"))
    reasons = {c.split_reason for c in produced}
    assert "window_split" in reasons
    for c in produced:
        assert c.token_count <= DEFAULT_CONFIG.max_tokens
