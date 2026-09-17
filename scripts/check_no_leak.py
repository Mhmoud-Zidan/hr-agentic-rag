#!/usr/bin/env python3
"""Fail the build when a mock-data value leaks into policy prose.

Why this exists
---------------
A worked example built on a roster value plants the gold answer inside a
retrievable chunk. The agent then pattern-matches a paragraph instead of
reasoning across balance, notice tier and calendar: workflow-completion numbers
stop measuring anything, and the tools-on/tools-off ablation collapses because
the answer is in the corpus either way. The leak is invisible in the prose --
"an employee with 14.0 days" reads like perfectly good policy writing.

This implements the first binding rule in CORPUS_SPEC.md §6.

What counts as a leak
---------------------
Any PTO balance, sick balance, scheduled-hours figure, carryover amount or
pending-request length from data/mock_data/*.json, appearing as a number in
data/policies/**.

Values that are ALSO canonical policy facts in CORPUS_SPEC.md §3 are allowed:
8 is both Omar's sick balance and the annual sick entitlement, and a policy
document cannot state its own entitlement without using the number. The §3
table is parsed at runtime rather than hardcoded, so the allowlist tracks the
spec instead of drifting from it.

Usage:
    python scripts/check_no_leak.py
    python scripts/check_no_leak.py --verbose    # show the allowlist too

Exit code 0 when clean, 1 on any hit. Suitable for CI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_DATA_DIR = REPO_ROOT / "data" / "mock_data"
POLICY_DIR = REPO_ROOT / "data" / "policies"
SPEC_PATH = REPO_ROOT / "data" / "CORPUS_SPEC.md"

# The fields whose values identify a specific person. A number that only ever
# appears as one of these is, by definition, employee data rather than policy.
LEAKABLE_FIELDS = (
    "pto_balance_days",
    "sick_balance_days",
    "scheduled_hours_per_week",
    "carryover_days",
    "days",  # pending_requests[].days -- a request length is equally identifying
)

POLICY_SUFFIXES = {".md", ".markdown", ".html", ".htm"}

# Spans that contain digits but are never prose figures. Masked before scanning
# so that "§4", "PTO-01", "2026-01-01" and "### 11.2" cannot raise a hit.
#
# ORDER MATTERS. Markup is stripped FIRST, so that the structural patterns below
# see a plain-text line. With the tag strip last, an HTML heading still looked
# like "<h2>4. Expense..." when the ordered-list pattern ran, so the heading
# number survived and was reported as a leak. Same for CSS: digits inside a
# <style> block are not inside a tag and are not prose either.
_MASK_PATTERNS = (
    # --- markup first ---
    re.compile(r"<(style|script)\b.*?</\1>", re.IGNORECASE | re.DOTALL),  # CSS/JS bodies
    re.compile(r"<[^>]+>"),                      # html tags/attributes
    re.compile(r"^---$.*?^---$", re.MULTILINE | re.DOTALL),   # front matter
    # --- then structure, now visible as plain text ---
    re.compile(r"\b[A-Z]{3}-\d{2}\b"),           # doc ids: PTO-01
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),        # ISO dates
    re.compile(r"§\s*\d+(?:\.\d+)*"),            # section refs: §4, §11.2
    re.compile(r"\bv\d+(?:\.\d+)*\b"),           # versions: v4.2
    re.compile(r"^#{1,6}\s+\d+(?:\.\d+)*\.?", re.MULTILINE),  # headings: ## 4.
    re.compile(r"^\s*\d+(?:\.\d+)*[.)]\s", re.MULTILINE),     # ordered/heading numbers
    # A dotted subsection number opening a line, e.g. an HTML "<h3>8.5 Title</h3>"
    # once its tags are stripped. Requires a dot, so a bare figure in prose is
    # still checked -- only "N.N " at line start is treated as structure.
    re.compile(r"^\s*\d+(?:\.\d+)+\s", re.MULTILINE),
)


@dataclass(frozen=True)
class Hit:
    path: Path
    line_number: int
    value: str
    line: str
    fields: tuple[str, ...]


def _walk_numbers(node: Any, field_name: str | None = None) -> Iterable[tuple[str, float]]:
    """Yield (field_name, value) for every number under a leakable key."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_numbers(value, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_numbers(item, field_name)
    elif isinstance(node, bool):
        return  # bool is an int subclass; never a balance
    elif isinstance(node, (int, float)) and field_name in LEAKABLE_FIELDS:
        yield field_name, float(node)


def collect_mock_values(directory: Path) -> dict[float, set[str]]:
    """Map each leakable numeric value to the field names it came from."""
    values: dict[float, set[str]] = {}
    if not directory.is_dir():
        return values

    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        for field_name, value in _walk_numbers(payload):
            values.setdefault(value, set()).add(f"{path.name}:{field_name}")
    return values


def collect_canonical_numbers(spec_path: Path) -> set[str]:
    """Every number in CORPUS_SPEC.md §3, as LITERALLY SPELLED.

    Spellings, not values, because the two carry different information. §3 says
    "14 calendar days"; it never says "14.0". A balance is written "14.0 days",
    and that decimal spelling is a tell that the figure came from a roster rather
    than from policy. Suppressing the whole value 14.0 because the spec contains
    "14" is what let Jane's balance through the first version of this check.
    """
    if not spec_path.is_file():
        print(f"warning: no spec at {spec_path}; allowlist will be empty", file=sys.stderr)
        return set()

    text = spec_path.read_text(encoding="utf-8")
    # §3 runs from its own heading to the next top-level section heading.
    match = re.search(r"^##\s*3\..*?(?=^##\s*\d+\.)", text, re.MULTILINE | re.DOTALL)
    section = match.group(0) if match else ""
    if not section:
        print("warning: could not locate §3 in the spec", file=sys.stderr)
        return set()

    # Drop clock times first: "10:00-15:00" would otherwise contribute "00",
    # allowlisting 0 for no good reason.
    section = re.sub(r"\b\d{1,2}:\d{2}\b", " ", section)
    return set(re.findall(r"\d+(?:\.\d+)?", section.replace(",", "")))


def _blank(match: re.Match[str]) -> str:
    """Replace a span with spaces, PRESERVING newlines.

    Length alone is not enough: a multi-line mask (front matter, an HTML tag
    spanning lines) that swallows its newlines shifts every subsequent line
    number, and the reported location silently points at the wrong line.
    """
    return "".join("\n" if ch == "\n" else " " for ch in match.group(0))


def _mask(text: str) -> str:
    """Blank out digit spans that are structure, not prose figures."""
    for pattern in _MASK_PATTERNS:
        text = pattern.sub(_blank, text)
    return text


def _spellings(value: float) -> set[str]:
    """The written forms of a value a policy author would plausibly use."""
    forms = {f"{value:g}"}
    if value == int(value):
        forms.add(str(int(value)))
        forms.add(f"{int(value)}.0")
    return forms


def _number_pattern(forms: Iterable[str]) -> re.Pattern[str]:
    """Match any of these exact spellings, never a fragment of a larger number."""
    alternatives = "|".join(
        re.escape(f) for f in sorted(forms, key=len, reverse=True)
    )
    return re.compile(rf"(?<![\d.,])(?:{alternatives})(?![\d.,]*\d)")


def scan(
    policy_dir: Path,
    mock_values: dict[float, set[str]],
    allowed: set[str],
) -> tuple[list[Hit], list[tuple[float, set[str]]]]:
    """Return (hits, the (value, spellings) pairs actually checked).

    A spelling is dropped only if §3 uses that exact spelling. A value whose
    every spelling is canonical drops out entirely.
    """
    checked: list[tuple[float, set[str]]] = []
    for value in sorted(mock_values):
        forms = {f for f in _spellings(value) if f not in allowed}
        if forms:
            checked.append((value, forms))

    patterns = [(value, _number_pattern(forms)) for value, forms in checked]

    hits: list[Hit] = []
    if not policy_dir.is_dir():
        return hits, checked

    for path in sorted(policy_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in POLICY_SUFFIXES:
            continue
        raw = path.read_text(encoding="utf-8")
        masked = _mask(raw)

        for line_number, (masked_line, raw_line) in enumerate(
            zip(masked.splitlines(), raw.splitlines()), start=1
        ):
            for value, pattern in patterns:
                match = pattern.search(masked_line)
                if match:
                    hits.append(
                        Hit(
                            path=path,
                            line_number=line_number,
                            value=match.group(0),
                            line=raw_line.strip(),
                            fields=tuple(sorted(mock_values[value])),
                        )
                    )
    return hits, checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--mock-dir", type=Path, default=MOCK_DATA_DIR)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--verbose", action="store_true", help="show the allowlist")
    args = parser.parse_args(argv)

    mock_values = collect_mock_values(args.mock_dir)
    if not mock_values:
        print(f"no mock data found in {args.mock_dir}; nothing to check.")
        return 0

    allowed = collect_canonical_numbers(args.spec)
    hits, checked = scan(args.policy_dir, mock_values, allowed)

    checked_values = {value for value, _ in checked}
    suppressed = sorted(v for v in mock_values if v not in checked_values)
    spellings = sorted(
        {f for _, forms in checked for f in forms}, key=lambda s: (len(s), s)
    )

    print(f"mock values found      : {len(mock_values)}")
    print(f"fully canonical in §3  : {len(suppressed)} suppressed "
          f"({', '.join(f'{v:g}' for v in suppressed) or 'none'})")
    print(f"values checked         : {len(checked)} "
          f"({', '.join(f'{v:g}' for v, _ in checked) or 'none'})")
    print(f"spellings searched     : {', '.join(spellings) or 'none'}")

    if args.verbose:
        print(f"allowlist (§3)         : "
              f"{', '.join(sorted(allowed, key=lambda s: (len(s), s))) or 'empty'}")
    print()

    if not hits:
        print("PASS: no mock-data value appears in policy prose.")
        return 0

    print(f"FAIL: {len(hits)} mock-data value(s) leaked into policy prose.\n")
    for hit in hits:
        try:
            label = hit.path.relative_to(REPO_ROOT)
        except ValueError:
            label = hit.path
        print(f"  {label}:{hit.line_number}  value={hit.value}")
        print(f"      from: {', '.join(hit.fields)}")
        print(f"      {hit.line}")
        print()

    print(
        "Replace each figure with a value that appears nowhere in data/mock_data/.\n"
        "See CORPUS_SPEC.md §6, binding authoring rule 1."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
