"""Tests for app.parsers.

Two jobs: pin PTO-01's real structure, and pin Markdown/HTML parity so the
downstream chunker cannot behave differently depending on source format.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from app.parsers import Section, parse_corpus, parse_html, parse_markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
PTO_PATH = REPO_ROOT / "data" / "policies" / "pto_and_leave_policy.md"

# The 13 numbered H2 sections of PTO-01, in document order.
PTO_EXPECTED = [
    ("1", "Purpose, Scope, and Definitions"),
    ("2", "PTO Accrual"),
    ("3", "Requesting PTO: Notice Periods and Approvals"),
    ("4", "Carryover and Expiration"),
    ("5", "Payout on Termination"),
    ("6", "Sick Leave"),
    ("7", "Bereavement Leave"),
    ("8", "Parental Leave"),
    ("9", "Unpaid Leave of Absence"),
    ("10", "Interaction with Probation, Remote Work, and Benefits"),
    ("11", "Worked Examples"),
    ("12", "Frequently Asked Questions"),
    ("13", "Roles, Responsibilities, and Document Control"),
]

MAX_SECTION_WORDS = 1200  # CORPUS_SPEC.md §2


@pytest.fixture(scope="module")
def pto_sections() -> list[Section]:
    return parse_markdown(PTO_PATH)


# --------------------------------------------------------------------------
# PTO-01 structure
# --------------------------------------------------------------------------


def test_pto_section_count(pto_sections):
    assert len(pto_sections) == len(PTO_EXPECTED) == 13


def test_pto_numbers_and_headings_in_order(pto_sections):
    assert [(s.number, s.heading) for s in pto_sections] == PTO_EXPECTED


def test_known_heading_has_correct_number(pto_sections):
    """§4 is Carryover and Expiration — number and heading must not drift apart."""
    by_heading = {s.heading: s for s in pto_sections}
    assert "Carryover and Expiration" in by_heading
    assert by_heading["Carryover and Expiration"].number == "4"

    by_number = {s.number: s for s in pto_sections}
    assert by_number["3"].heading == "Requesting PTO: Notice Periods and Approvals"
    assert by_number["9"].heading == "Unpaid Leave of Absence"


def test_pto_document_identity(pto_sections):
    for s in pto_sections:
        assert s.doc_id == "PTO-01"
        assert s.doc_title == "PTO and Leave Policy"
        assert s.source_filename == "pto_and_leave_policy.md"
        assert s.source_format == "markdown"


def test_heading_path_format(pto_sections):
    assert (
        pto_sections[2].heading_path
        == "PTO and Leave Policy (PTO-01) > 3. Requesting PTO: Notice Periods and Approvals"
    )
    for s in pto_sections:
        assert s.heading_path == f"PTO and Leave Policy (PTO-01) > {s.number}. {s.heading}"


def test_h3_nested_not_split(pto_sections):
    """H3s stay inside the parent H2 rather than becoming their own Sections."""
    section_3 = next(s for s in pto_sections if s.number == "3")
    assert "### 3.1 Notice periods" in section_3.text
    assert "### 3.2 Approvers" in section_3.text
    # No Section object was emitted for an H3.
    assert all("." not in s.number for s in pto_sections)


def test_front_matter_and_h1_excluded(pto_sections):
    joined = "\n".join(s.text for s in pto_sections)
    assert "doc_id: PTO-01" not in joined
    assert "# PTO and Leave Policy (PTO-01)" not in joined
    # The heading line itself is not repeated in its own body.
    for s in pto_sections:
        assert not s.text.startswith(f"## {s.number}.")


def test_offsets_are_contiguous_and_slice_correctly(pto_sections):
    raw = PTO_PATH.read_text(encoding="utf-8")
    for i, s in enumerate(pto_sections):
        assert 0 <= s.char_start < s.char_end <= len(raw)
        assert raw[s.char_start : s.char_end].startswith(f"## {s.number}. {s.heading}")
        if i + 1 < len(pto_sections):
            assert s.char_end == pto_sections[i + 1].char_start


def test_known_fact_lands_in_the_right_section(pto_sections):
    """Retrieval depends on facts sitting under the heading that describes them."""
    by_number = {s.number: s for s in pto_sections}
    assert "5 days" in by_number["4"].text and "March 31" in by_number["4"].text
    assert "day 91" in by_number["2"].text
    assert "8 days of sick leave" in by_number["6"].text


def test_no_section_exceeds_chunk_budget(pto_sections):
    oversized = [(s.number, s.word_count) for s in pto_sections if s.word_count > MAX_SECTION_WORDS]
    assert not oversized, f"sections over {MAX_SECTION_WORDS} words: {oversized}"


# --------------------------------------------------------------------------
# Markdown / HTML parity
# --------------------------------------------------------------------------

EQUIVALENT_MD = """---
doc_id: APR-01
title: Approvals and Escalation Matrix
version: v4.2
effective_date: 2026-01-01
owner: People Operations
---

# Approvals and Escalation Matrix (APR-01)

## 1. Purpose

Intro prose for section one.

### 1.1 Detail

Nested detail text.

---

## 2. Thresholds

Second section prose.
"""

EQUIVALENT_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta name="doc_id" content="APR-01">
  <meta name="title" content="Approvals and Escalation Matrix">
  <meta name="version" content="v4.2">
  <meta name="effective_date" content="2026-01-01">
  <meta name="owner" content="People Operations">
  <style>body { color: red; }</style>
</head>
<body>
  <nav><a href="#">Skip me</a></nav>
  <h1>Approvals and Escalation Matrix (APR-01)</h1>
  <h2>1. Purpose</h2>
  <p>Intro prose for section one.</p>
  <h3>1.1 Detail</h3>
  <p>Nested detail text.</p>
  <h2>2. Thresholds</h2>
  <p>Second section prose.</p>
  <script>console.log("skip me");</script>
</body>
</html>
"""


@pytest.fixture()
def pair(tmp_path: Path) -> tuple[list[Section], list[Section]]:
    md = tmp_path / "approvals_and_escalation_matrix.md"
    html = tmp_path / "approvals_and_escalation_matrix.html"
    md.write_text(EQUIVALENT_MD, encoding="utf-8")
    html.write_text(EQUIVALENT_HTML, encoding="utf-8")
    return parse_markdown(md), parse_html(html)


def test_formats_agree_on_structure(pair):
    """The fields a chunker branches on must be identical across formats."""
    md_sections, html_sections = pair
    assert len(md_sections) == len(html_sections) == 2

    structural = ("doc_id", "doc_title", "number", "heading", "heading_path")
    for m, h in zip(md_sections, html_sections):
        assert {f: getattr(m, f) for f in structural} == {f: getattr(h, f) for f in structural}


def test_html_reports_its_own_format_and_filename(pair):
    _, html_sections = pair
    for s in html_sections:
        assert s.source_format == "html"
        assert s.source_filename.endswith(".html")


def test_html_nests_h3_and_strips_chrome(pair):
    _, html_sections = pair
    first = html_sections[0]
    assert "1.1 Detail" in first.text
    assert "Nested detail text." in first.text
    assert "Skip me" not in first.text  # <nav>
    assert "console.log" not in first.text  # <script>
    assert "color: red" not in first.text  # <style>


def test_html_offsets_slice_the_source(pair, tmp_path):
    _, html_sections = pair
    raw = (tmp_path / "approvals_and_escalation_matrix.html").read_text(encoding="utf-8")
    for s in html_sections:
        assert 0 <= s.char_start < s.char_end <= len(raw)
        assert raw[s.char_start : s.char_end].lstrip().lower().startswith("<h2")


def test_both_formats_exclude_h1_and_metadata(pair):
    for sections in pair:
        joined = "\n".join(s.text for s in sections)
        assert "Approvals and Escalation Matrix (APR-01)" not in joined
        assert "doc_id" not in joined


# --------------------------------------------------------------------------
# parse_corpus
# --------------------------------------------------------------------------


def test_parse_corpus_dispatches_by_extension(tmp_path: Path):
    (tmp_path / "a.md").write_text(EQUIVALENT_MD, encoding="utf-8")
    (tmp_path / "b.html").write_text(EQUIVALENT_HTML, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    sections = parse_corpus(tmp_path)
    assert len(sections) == 4
    assert {s.source_format for s in sections} == {"markdown", "html"}
    assert "notes.txt" not in {s.source_filename for s in sections}


def test_parse_corpus_on_real_policies_includes_pto():
    sections = parse_corpus(REPO_ROOT / "data" / "policies")
    pto = [s for s in sections if s.doc_id == "PTO-01"]
    assert len(pto) == 13


def test_parse_corpus_rejects_non_directory():
    with pytest.raises(NotADirectoryError):
        parse_corpus(PTO_PATH)


def test_section_is_a_dataclass_with_the_agreed_fields():
    assert [f.name for f in dataclasses.fields(Section)] == [
        "doc_id",
        "doc_title",
        "number",
        "heading",
        "heading_path",
        "text",
        "source_filename",
        "source_format",
        "char_start",
        "char_end",
    ]
