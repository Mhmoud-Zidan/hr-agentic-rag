"""Heading-aware parsers for the Northwind policy corpus.

Turns a policy document into a flat list of `Section` objects, one per H2.
H3 subsections are *nested*: their text stays inside the parent H2's `text`
rather than becoming separate Sections. That follows CORPUS_SPEC.md §2, which
makes the H2 the retrieval unit ("every H2 section must be independently
meaningful", capped at ~1,200 words so one H2 fits the embedding window).

Markdown and HTML deliberately produce the same Section shape: same field
semantics, same numbering, same heading_path format, same ordering, H1 and
front matter excluded from all section text. Downstream chunking must not
need to know which format a Section came from.

The one unavoidable difference: Markdown `text` retains Markdown syntax
(pipes, asterisks, list markers) while HTML `text` is extracted plain text.
Both are normalized to the same whitespace convention.

No embedding, no chunking, no I/O beyond reading the files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

__all__ = ["Section", "parse_markdown", "parse_html", "parse_corpus"]

MARKDOWN_SUFFIXES = {".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}

# "## 3. Carryover and Expiration" -> number "3", heading "Carryover and Expiration"
_NUMBERED_HEADING = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<heading>.+?)\s*$")
_H1_DOC_ID = re.compile(r"\(([A-Z]{3}-\d{2})\)\s*$")
_MD_H1 = re.compile(r"^#\s+(?P<text>.+?)\s*$")
_MD_H2 = re.compile(r"^##\s+(?P<text>.+?)\s*$")
_FENCE = re.compile(r"^\s*```")
_HR_LINE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_BLANK_RUN = re.compile(r"\n{3,}")
_HTML_H2_TAG = re.compile(r"<h2\b", re.IGNORECASE)

_STRIP_TAGS = ("nav", "script", "style", "header", "footer", "aside", "noscript")


@dataclass
class Section:
    """One H2 section of one policy document, with its H3 subsections nested in `text`."""

    doc_id: str
    doc_title: str
    number: str
    heading: str
    heading_path: str
    text: str
    source_filename: str
    source_format: str
    char_start: int
    char_end: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _split_heading(raw_heading: str) -> tuple[str, str]:
    """'3. Carryover and Expiration' -> ('3', 'Carryover and Expiration')."""
    m = _NUMBERED_HEADING.match(raw_heading)
    if m:
        return m.group("number"), m.group("heading")
    return "", raw_heading.strip()


def _build_path(doc_title: str, doc_id: str, number: str, heading: str) -> str:
    """'PTO and Leave Policy (PTO-01) > 3. Requesting PTO'."""
    label = f"{number}. {heading}" if number else heading
    return f"{doc_title} ({doc_id}) > {label}"


def _normalize(text: str) -> str:
    """Collapse blank runs and drop horizontal rules, so both formats agree."""
    kept = [line.rstrip() for line in text.splitlines() if not _HR_LINE.match(line)]
    return _BLANK_RUN.sub("\n\n", "\n".join(kept)).strip()


def _parse_front_matter(text: str) -> tuple[dict[str, str], int]:
    """Return (front matter mapping, offset where the body starts)."""
    if not text.startswith("---"):
        return {}, 0
    end = text.find("\n---", 3)
    if end == -1:
        return {}, 0
    block = text[3:end]
    body_start = text.find("\n", end + 1)
    body_start = len(text) if body_start == -1 else body_start + 1

    fm: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip().strip("\"'")
    return fm, body_start


def parse_markdown(path: str | Path) -> list[Section]:
    """Parse a Markdown policy document into one Section per H2."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    front_matter, body_start = _parse_front_matter(raw)

    doc_id = front_matter.get("doc_id", "")
    doc_title = front_matter.get("title", "")

    # Locate every H2 by absolute character offset, ignoring fenced code blocks.
    starts: list[tuple[int, str]] = []
    h1_text: str | None = None
    offset = body_start
    in_fence = False
    for line in raw[body_start:].splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if _FENCE.match(stripped):
            in_fence = not in_fence
        elif not in_fence:
            if h1_text is None:
                m1 = _MD_H1.match(stripped)
                if m1:
                    h1_text = m1.group("text")
            m2 = _MD_H2.match(stripped)
            if m2:
                starts.append((offset, m2.group("text")))
        offset += len(line)

    # Fall back to the H1 when front matter is absent or incomplete.
    if h1_text:
        if not doc_id:
            m = _H1_DOC_ID.search(h1_text)
            if m:
                doc_id = m.group(1)
        if not doc_title:
            doc_title = _H1_DOC_ID.sub("", h1_text).strip()

    sections: list[Section] = []
    for i, (start, raw_heading) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(raw)
        number, heading = _split_heading(raw_heading)
        block = raw[start:end]
        body = block.split("\n", 1)[1] if "\n" in block else ""
        sections.append(
            Section(
                doc_id=doc_id,
                doc_title=doc_title,
                number=number,
                heading=heading,
                heading_path=_build_path(doc_title, doc_id, number, heading),
                text=_normalize(body),
                source_filename=path.name,
                source_format="markdown",
                char_start=start,
                char_end=end,
            )
        )
    return sections


def parse_html(path: str | Path) -> list[Section]:
    """Parse an HTML policy document into one Section per <h2>, mirroring parse_markdown."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    meta = {
        m.get("name", "").strip().lower(): (m.get("content") or "").strip()
        for m in soup.find_all("meta")
        if m.get("name")
    }
    doc_id = meta.get("doc_id", "")
    doc_title = meta.get("title", "")

    h1 = soup.find("h1")
    h1_text = " ".join(h1.get_text(" ", strip=True).split()) if h1 else ""
    if h1_text:
        if not doc_id:
            m = _H1_DOC_ID.search(h1_text)
            if m:
                doc_id = m.group(1)
        if not doc_title:
            doc_title = _H1_DOC_ID.sub("", h1_text).strip()

    h2s = soup.find_all("h2")
    # Offsets come from the raw source so they stay comparable with Markdown.
    # bs4 does not expose character positions, and <h2> tags appear in source order.
    tag_starts = [m.start() for m in _HTML_H2_TAG.finditer(raw)]
    if len(tag_starts) != len(h2s):
        tag_starts = [-1] * len(h2s)  # stripped tags shifted the count; degrade, don't lie

    sections: list[Section] = []
    for i, h2 in enumerate(h2s):
        raw_heading = " ".join(h2.get_text(" ", strip=True).split())
        number, heading = _split_heading(raw_heading)

        # Everything after this <h2> up to the next <h2>, H3s included.
        parts: list[str] = []
        for sib in h2.next_siblings:
            if getattr(sib, "name", None) == "h2":
                break
            chunk = (
                sib.get_text(" ", strip=True)
                if getattr(sib, "name", None)
                else str(sib).strip()
            )
            if chunk:
                parts.append(" ".join(chunk.split()))

        start = tag_starts[i] if i < len(tag_starts) else -1
        if start == -1:
            end = -1
        elif i + 1 < len(tag_starts) and tag_starts[i + 1] != -1:
            end = tag_starts[i + 1]
        else:
            end = len(raw)

        sections.append(
            Section(
                doc_id=doc_id,
                doc_title=doc_title,
                number=number,
                heading=heading,
                heading_path=_build_path(doc_title, doc_id, number, heading),
                text=_normalize("\n\n".join(parts)),
                source_filename=path.name,
                source_format="html",
                char_start=start,
                char_end=end,
            )
        )
    return sections


def parse_corpus(directory: str | Path) -> list[Section]:
    """Parse every Markdown and HTML document in `directory`, dispatching by extension."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    sections: list[Section] = []
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix in MARKDOWN_SUFFIXES:
            sections.extend(parse_markdown(path))
        elif suffix in HTML_SUFFIXES:
            sections.extend(parse_html(path))
    return sections
