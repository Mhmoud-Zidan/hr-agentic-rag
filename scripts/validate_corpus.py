#!/usr/bin/env python3
"""Corpus validator for the Northwind Systems policy corpus.

Implements the mechanical checks required by data/CORPUS_SPEC.md §6:

  check 2 - no H2 section exceeds ~1,200 words
  check 3 - front matter present and correct
  check 4 - cross-references point at doc IDs that exist

Plus the structural rules from §2 that are cheap to verify mechanically:
H1 carries the doc ID in parentheses, and H2 headings are numbered.

Checks 1 (every number appears in §3) and 5 (nothing out of scope) are
judgement calls and are deliberately not automated here; §4.6 out-of-scope
topics are flagged as warnings only.

Usage:
    python scripts/validate_corpus.py                  # whole corpus
    python scripts/validate_corpus.py data/policies/pto_and_leave_policy.md

Exit code 0 if every file passes, 1 otherwise. Suitable for CI.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = REPO_ROOT / "data" / "policies"

MAX_SECTION_WORDS = 1200

# §1 of the spec: all documents carry the same version and effective date.
EXPECTED_VERSION = "v4.2"
EXPECTED_EFFECTIVE_DATE = "2026-01-01"

REQUIRED_FRONT_MATTER = ("doc_id", "title", "version", "effective_date", "owner")

# §2 document map. Filename -> doc_id.
DOCUMENT_MAP = {
    "pto_and_leave_policy.md": "PTO-01",
    "remote_and_hybrid_work_policy.md": "REM-01",
    "work_location_and_tax_compliance.md": "TAX-01",
    "expense_and_reimbursement_policy.md": "EXP-01",
    "benefits_and_healthcare_policy.md": "BEN-01",
    "data_security_and_acceptable_use.md": "SEC-01",
    "equipment_and_asset_policy.md": "EQP-01",
    "onboarding_and_probation.md": "ONB-01",
    "workplace_conduct_and_grievance.html": "CON-01",
    "approvals_and_escalation_matrix.html": "APR-01",
}

KNOWN_DOC_IDS = set(DOCUMENT_MAP.values())

# §4.6: topics the corpus must not cover. Warned about, not failed.
OUT_OF_SCOPE_TERMS = (
    "stock option",
    "sabbatical",
    "pet insurance",
    "tuition reimbursement",
    "relocation",
)

DOC_ID_REF = re.compile(r"\b([A-Z]{3}-\d{2})\b")
H1_RE = re.compile(r"^#\s+(?P<text>.+?)\s*$")
H2_RE = re.compile(r"^##\s+(?P<text>.+?)\s*$")
NUMBERED_H2 = re.compile(r"^\d+\.\s+\S")
H1_HAS_DOC_ID = re.compile(r"\(([A-Z]{3}-\d{2})\)\s*$")

HTML_TAG = re.compile(r"<[^>]+>")
HTML_H1 = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
HTML_H2_SPLIT = re.compile(r"<h2\b[^>]*>(.*?)</h2>", re.IGNORECASE | re.DOTALL)
HTML_META = re.compile(
    r"<meta\s+name=[\"'](?P<name>[^\"']+)[\"']\s+content=[\"'](?P<content>[^\"']*)[\"']",
    re.IGNORECASE,
)


@dataclass
class Section:
    heading: str
    body: str

    @property
    def word_count(self) -> int:
        return len(self.body.split())


@dataclass
class Document:
    path: Path
    front_matter: dict[str, str]
    h1: str | None
    sections: list[Section]
    raw: str


@dataclass
class Result:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _strip_tags(fragment: str) -> str:
    return html.unescape(HTML_TAG.sub(" ", fragment))


def parse_markdown(path: Path, text: str) -> Document:
    front_matter: dict[str, str] = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                front_matter[key.strip()] = value.strip().strip("\"'")
            body = parts[2]

    h1: str | None = None
    sections: list[Section] = []
    current: Section | None = None
    in_fence = False

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            m1 = H1_RE.match(line)
            if m1 and h1 is None:
                h1 = m1.group("text")
                continue
            m2 = H2_RE.match(line)
            if m2:
                current = Section(heading=m2.group("text"), body="")
                sections.append(current)
                continue
        if current is not None:
            current.body += line + "\n"

    return Document(path=path, front_matter=front_matter, h1=h1, sections=sections, raw=text)


def parse_html(path: Path, text: str) -> Document:
    front_matter = {
        m.group("name").strip().lower(): html.unescape(m.group("content").strip())
        for m in HTML_META.finditer(text)
    }

    h1_match = HTML_H1.search(text)
    h1 = " ".join(_strip_tags(h1_match.group(1)).split()) if h1_match else None

    sections: list[Section] = []
    pieces = HTML_H2_SPLIT.split(text)
    # pieces = [before_first_h2, heading1, body1, heading2, body2, ...]
    for i in range(1, len(pieces) - 1, 2):
        heading = " ".join(_strip_tags(pieces[i]).split())
        body = _strip_tags(pieces[i + 1])
        sections.append(Section(heading=heading, body=body))

    return Document(path=path, front_matter=front_matter, h1=h1, sections=sections, raw=text)


def load(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".html", ".htm"):
        return parse_html(path, text)
    return parse_markdown(path, text)


def check_front_matter(doc: Document, result: Result) -> None:
    """Spec §6 check 3: front matter present and correct."""
    fm = doc.front_matter
    if not fm:
        result.errors.append(
            "front matter: missing entirely (expected keys: "
            + ", ".join(REQUIRED_FRONT_MATTER)
            + ")"
        )
        return

    for key in REQUIRED_FRONT_MATTER:
        if not fm.get(key):
            result.errors.append(f"front matter: missing or empty key '{key}'")

    expected_id = DOCUMENT_MAP.get(doc.path.name)
    doc_id = fm.get("doc_id")
    if doc_id and doc_id not in KNOWN_DOC_IDS:
        result.errors.append(f"front matter: doc_id '{doc_id}' is not in the §2 document map")
    if expected_id and doc_id and doc_id != expected_id:
        result.errors.append(
            f"front matter: doc_id '{doc_id}' does not match '{expected_id}' for this filename"
        )
    if expected_id is None:
        result.warnings.append("filename is not listed in the §2 document map")

    version = fm.get("version")
    if version and version != EXPECTED_VERSION:
        result.errors.append(
            f"front matter: version '{version}' should be '{EXPECTED_VERSION}' (spec §1)"
        )

    effective = fm.get("effective_date")
    if effective and effective != EXPECTED_EFFECTIVE_DATE:
        result.errors.append(
            f"front matter: effective_date '{effective}' should be "
            f"'{EXPECTED_EFFECTIVE_DATE}' (spec §1)"
        )


def check_structure(doc: Document, result: Result) -> None:
    """Spec §2 structural rules: H1 carries the doc ID, H2 headings are numbered."""
    if not doc.h1:
        result.errors.append("structure: no H1 title found")
    else:
        m = H1_HAS_DOC_ID.search(doc.h1)
        if not m:
            result.errors.append(
                f"structure: H1 '{doc.h1}' does not end with the doc ID in parentheses"
            )
        elif doc.front_matter.get("doc_id") and m.group(1) != doc.front_matter["doc_id"]:
            result.errors.append(
                f"structure: H1 doc ID '{m.group(1)}' does not match front matter "
                f"doc_id '{doc.front_matter['doc_id']}'"
            )

    if not doc.sections:
        result.errors.append("structure: no H2 sections found")

    for section in doc.sections:
        if not NUMBERED_H2.match(section.heading):
            result.errors.append(
                f"structure: H2 '{section.heading}' is not numbered (expected e.g. "
                "'3. Carryover and Expiration')"
            )


def check_section_length(doc: Document, result: Result) -> None:
    """Spec §6 check 2: no H2 section exceeds ~1,200 words."""
    for section in doc.sections:
        words = section.word_count
        if words > MAX_SECTION_WORDS:
            result.errors.append(
                f"length: H2 '{section.heading}' is {words} words, over the "
                f"{MAX_SECTION_WORDS}-word limit"
            )


def check_cross_references(doc: Document, result: Result) -> None:
    """Spec §6 check 4: cross-references point at doc IDs that exist."""
    own_id = doc.front_matter.get("doc_id")
    for ref in sorted(set(DOC_ID_REF.findall(doc.raw))):
        if ref == own_id:
            continue
        if ref not in KNOWN_DOC_IDS:
            result.errors.append(
                f"cross-reference: '{ref}' is not a doc ID in the §2 document map"
            )


def check_out_of_scope(doc: Document, result: Result) -> None:
    """Spec §4.6: these topics are deliberately absent. Warning only."""
    lowered = doc.raw.lower()
    for term in OUT_OF_SCOPE_TERMS:
        if term in lowered:
            result.warnings.append(
                f"out of scope: found '{term}', which §4.6 says the corpus must not cover"
            )


def validate(path: Path) -> Result:
    result = Result(path=path)
    try:
        doc = load(path)
    except (OSError, UnicodeDecodeError) as exc:
        result.errors.append(f"could not read file: {exc}")
        return result

    check_front_matter(doc, result)
    check_structure(doc, result)
    check_section_length(doc, result)
    check_cross_references(doc, result)
    check_out_of_scope(doc, result)
    return result


def collect(targets: list[str]) -> list[Path]:
    if targets:
        return [Path(t).resolve() for t in targets]
    if not POLICY_DIR.is_dir():
        return []
    return sorted(
        p for p in POLICY_DIR.iterdir() if p.suffix.lower() in (".md", ".html", ".htm")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files to validate (default: data/policies/*)")
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as errors"
    )
    args = parser.parse_args(argv)

    paths = collect(args.paths)
    if not paths:
        print("No corpus documents found to validate.")
        return 1

    results = [validate(p) for p in paths]
    failed = 0

    for r in results:
        try:
            label = r.path.relative_to(REPO_ROOT)
        except ValueError:
            label = r.path

        is_fail = bool(r.errors) or (args.strict and r.warnings)
        status = "FAIL" if is_fail else ("WARN" if r.warnings else "PASS")
        print(f"[{status}] {label}")
        for e in r.errors:
            print(f"    ERROR   {e}")
        for w in r.warnings:
            print(f"    WARNING {w}")
        if is_fail:
            failed += 1

    print(f"\n{len(results) - failed}/{len(results)} document(s) passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
