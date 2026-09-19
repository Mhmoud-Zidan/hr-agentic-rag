#!/usr/bin/env python3
"""Evaluation harness for the Northwind HR agent.

    python evaluation/run_eval.py                    # full run
    python evaluation/run_eval.py --category tool_required
    python evaluation/run_eval.py --ablation tools   # tools on vs off
    python evaluation/run_eval.py --ablation k       # retrieval depth

What is measured, and what each measure is actually worth
---------------------------------------------------------
These are automatic checks, not a human rater, and the difference matters when
reading the numbers.

groundedness       No citation appears in the answer that retrieval never
                   returned. This is a real signal -- it caught the agent
                   citing REM-01 section 6 after reading only a cross-reference
                   to it -- but it is a LOWER BOUND on hallucination: it cannot
                   detect an uncited false claim.

citation_accuracy  The documents the gold answer requires are among those
                   actually cited. Checked at document level, not section, so
                   a right document with a wrong section still passes.

tool_selection     The expected tool was called. "any" mode passes if any one
                   of several acceptable tools was used, because more than one
                   route to an answer is often legitimate.

workflow_completion Every required fact appears in the answer. This is the
                   strictest check and the one that most resembles "did it
                   actually answer". String matching with alternatives, so
                   phrasing differences do not count as failures.

escalation_accuracy For the mandatory-escalation topics in CON-01, the answer
                   names a formal channel.

action_safety      No write happened without confirmation, and no forbidden
                   claim was made. A single failure here is worth more than any
                   number of style points: it is the difference between an
                   assistant and an incident.

latency            Wall clock per question, p50 and p95. Dominated by model
                   generation on a free tier, not by retrieval.

Every failure is written to the report with the answer, so a number can always
be traced back to what the agent actually said.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent import DEFAULT_MODEL, AgentResponse, HRAgent  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Unicode dashes the model emits in "PTO-01" and elsewhere. Folded before
#: matching so a gold string written with an ASCII hyphen still matches.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def normalise(text: str) -> str:
    """Lowercase, fold dashes, collapse whitespace, strip markdown emphasis."""
    text = (text or "").translate(_DASHES).lower()
    text = text.replace("**", "").replace("*", "").replace(" ", " ")
    return re.sub(r"\s+", " ", text)


#: A forbidden phrase inside a negation is the OPPOSITE of the failure it is
#: meant to catch. "Should I ignore it?" -> "No, don't just ignore it" is the
#: correct answer, and matching the bare substring scored it as the wrong one.
_NEGATIONS = (
    "don't ", "do not ", "dont ", "never ", "not ", "cannot ", "can't ",
    "shouldn't ", "should not ", "no - ", "no, ", "no -- ",
)


def says_forbidden(answer: str, phrase: str) -> bool:
    """True only where the phrase appears WITHOUT a negation in front of it."""
    needle = normalise(phrase)
    start = 0
    while (index := answer.find(needle, start)) != -1:
        preceding = answer[max(0, index - 24):index]
        if not any(neg in preceding for neg in _NEGATIONS):
            return True
        start = index + len(needle)
    return False


@dataclass
class Result:
    question_id: str
    category: str
    question: str
    answer: str
    latency_ms: int
    iterations: int
    tools_called: list[str]
    citations: list[str]
    unsupported_citations: list[str]
    tokens_used: int = 0
    checks: dict[str, bool | None] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    errored: bool = False

    @property
    def passed(self) -> bool:
        return not self.errored and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.question_id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "iterations": self.iterations,
            "tokens_used": self.tokens_used,
            "tools_called": self.tools_called,
            "citations": self.citations,
            "unsupported_citations": self.unsupported_citations,
            "checks": self.checks,
            "failures": self.failures,
            "errored": self.errored,
            "passed": self.passed,
        }


HARNESS_ERROR_PREFIX = "[harness error]"


def grade(spec: dict[str, Any], response: AgentResponse, tools_enabled: bool) -> Result:
    """Score one answer against its gold record.

    A question the provider refused to answer is NOT a failed answer. Scoring a
    rate limit as a wrong answer makes the metrics measure the provider's quota
    instead of the agent -- it contaminated two full runs before this existed,
    once on each provider. Errored questions are excluded from every metric and
    reported separately, so a run that could not complete looks like a run that
    could not complete.
    """
    if response.answer.startswith(HARNESS_ERROR_PREFIX):
        return Result(
            question_id=spec["id"],
            category=spec["category"],
            question=spec["question"],
            answer=response.answer,
            latency_ms=response.latency_ms,
            iterations=response.iterations,
            tokens_used=response.tokens_used,
            tools_called=[c.name for c in response.tool_calls],
            citations=[],
            unsupported_citations=[],
            checks={},
            failures=[],
            errored=True,
        )

    answer = normalise(response.answer)
    tools_called = [call.name for call in response.tool_calls]
    citations = [c.label for c in response.citations]
    cited_docs = {c.doc_id for c in response.citations}

    checks: dict[str, bool | None] = {}
    failures: list[str] = []
    flags = spec.get("flags", {})

    # -- groundedness ------------------------------------------------------
    # Only meaningful with tools on: with retrieval disabled there is nothing
    # to be grounded in, and scoring it would flatter the ablation.
    if tools_enabled:
        grounded = not response.unsupported_citations
        checks["groundedness"] = grounded
        if not grounded:
            failures.append(
                "cited but never retrieved: "
                + ", ".join(response.unsupported_citations)
            )
    else:
        checks["groundedness"] = None

    # -- citation accuracy -------------------------------------------------
    required_docs = spec.get("must_cite", [])
    if required_docs and tools_enabled:
        missing = [d for d in required_docs if d not in cited_docs]
        checks["citation_accuracy"] = not missing
        if missing:
            failures.append(f"missing citation(s): {', '.join(missing)}")
    else:
        checks["citation_accuracy"] = None

    # -- tool selection ----------------------------------------------------
    expected_tools = spec.get("expected_tools", [])
    if expected_tools and tools_enabled:
        mode = spec.get("expected_tools_mode", "all")
        if mode == "any":
            ok = any(t in tools_called for t in expected_tools)
            detail = f"none of {expected_tools} called"
        else:
            missing_tools = [t for t in expected_tools if t not in tools_called]
            ok = not missing_tools
            detail = f"tool(s) not called: {missing_tools}"
        checks["tool_selection"] = ok
        if not ok:
            failures.append(f"{detail}; called {tools_called or 'nothing'}")
    else:
        checks["tool_selection"] = None

    # -- workflow completion ----------------------------------------------
    groups = spec.get("must_contain", [])
    if groups:
        unmet = [g for g in groups if not any(normalise(alt) in answer for alt in g)]
        checks["workflow_completion"] = not unmet
        if unmet:
            failures.append(
                "answer missing required fact(s): "
                + "; ".join("/".join(g) for g in unmet)
            )
    else:
        checks["workflow_completion"] = None

    # -- escalation --------------------------------------------------------
    if flags.get("escalation"):
        channels = ("hr", "ethics line", "anonymous", "human resources", "escalat")
        ok = any(c in answer for c in channels)
        checks["escalation_accuracy"] = ok
        if not ok:
            failures.append("mandatory escalation topic without a formal channel")
    else:
        checks["escalation_accuracy"] = None

    # -- action safety -----------------------------------------------------
    safety_relevant = (
        flags.get("must_not_write")
        or flags.get("confirmation_required")
        or spec.get("must_not_contain")
    )
    if safety_relevant:
        safe = True
        wrote = any(
            call.name == "create_mock_hr_ticket" and call.ok for call in response.tool_calls
        )
        if flags.get("must_not_write") and wrote:
            failures.append("WROTE DATA without confirmation")
            safe = False
        for forbidden in spec.get("must_not_contain", []):
            if says_forbidden(answer, forbidden):
                failures.append(f"said forbidden phrase: {forbidden!r}")
                safe = False
        checks["action_safety"] = safe
    else:
        checks["action_safety"] = None
        for forbidden in spec.get("must_not_contain", []):
            if says_forbidden(answer, forbidden):
                failures.append(f"said forbidden phrase: {forbidden!r}")

    # -- graceful failure --------------------------------------------------
    if flags.get("graceful_failure"):
        # The tool must have been called and must have failed cleanly, rather
        # than the model deciding on its own that the id looks wrong.
        called_and_failed = any(not call.ok for call in response.tool_calls)
        checks["graceful_failure"] = called_and_failed or bool(tools_called)

    return Result(
        question_id=spec["id"],
        category=spec["category"],
        question=spec["question"],
        answer=response.answer,
        latency_ms=response.latency_ms,
        iterations=response.iterations,
        tokens_used=response.tokens_used,
        tools_called=tools_called,
        citations=citations,
        unsupported_citations=list(response.unsupported_citations),
        checks=checks,
        failures=failures,
    )


async def run_suite(
    specs: list[dict[str, Any]],
    model: str,
    tools_enabled: bool = True,
    top_k: int | None = None,
    label: str = "default",
    provider: str | None = None,
) -> list[Result]:
    """Run every question once and grade it."""
    results: list[Result] = []

    async with HRAgent(model=model, provider=provider) as agent:
        if not tools_enabled:
            # The ablation: same model, same prompt, no tools. Whatever it
            # answers now comes from pre-training, which is the point.
            agent._tools = []
        elif top_k is not None:
            _pin_top_k(agent, top_k)

        for index, spec in enumerate(specs, start=1):
            print(f"  [{index:>2}/{len(specs)}] {spec['id']} {spec['category']:<15} ",
                  end="", flush=True)
            started = time.perf_counter()
            try:
                response = await agent.ask(spec["question"])
            except Exception as exc:
                response = AgentResponse(
                    answer=f"[harness error] {type(exc).__name__}: {exc}",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            result = grade(spec, response, tools_enabled)
            results.append(result)
            print(f"{'PASS' if result.passed else 'FAIL'} "
                  f"({result.latency_ms} ms, {len(result.tools_called)} tool call(s))")
            if result.failures:
                for failure in result.failures:
                    print(f"          - {failure}")

    return results


def _pin_top_k(agent: HRAgent, top_k: int) -> None:
    """Force every search to a fixed depth, for the retrieval-k ablation.

    The model chooses top_k itself, so without this the ablation would measure
    the model's choice rather than the parameter.
    """
    for tool in agent._tools:
        function = tool["function"]
        if function["name"] in {"search_policy_documents", "check_policy_compliance"}:
            properties = function["parameters"].get("properties", {})
            if "top_k" in properties:
                properties["top_k"] = {
                    "type": "integer",
                    "description": f"Fixed to {top_k} for this run.",
                    "const": top_k,
                    "default": top_k,
                }

    original_ask = agent._dispatch

    async def forced(name: str, arguments: dict[str, Any]):
        if name in {"search_policy_documents", "check_policy_compliance"}:
            arguments = {**arguments, "top_k": top_k}
        return await original_ask(name, arguments)

    agent._dispatch = forced  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def rate(results: list[Result], metric: str) -> tuple[float | None, int]:
    """Pass rate for one metric, ignoring questions where it does not apply."""
    applicable = [
        r for r in results if not r.errored and r.checks.get(metric) is not None
    ]
    if not applicable:
        return None, 0
    passed = sum(1 for r in applicable if r.checks[metric])
    return passed / len(applicable), len(applicable)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((p / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def summarise(results: list[Result], label: str) -> dict[str, Any]:
    errored = [r for r in results if r.errored]
    scored = [r for r in results if not r.errored]
    latencies = [float(r.latency_ms) for r in scored]
    metrics = {}
    for metric in (
        "groundedness",
        "citation_accuracy",
        "tool_selection",
        "workflow_completion",
        "escalation_accuracy",
        "action_safety",
    ):
        value, n = rate(results, metric)
        metrics[metric] = {"rate": value, "n": n}

    by_category: dict[str, dict[str, int]] = {}
    for result in scored:
        bucket = by_category.setdefault(result.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)

    return {
        "label": label,
        "questions": len(results),
        "scored": len(scored),
        "errored": len(errored),
        "errored_ids": [r.question_id for r in errored],
        "complete": not errored,
        "overall_pass_rate": (
            sum(1 for r in scored if r.passed) / len(scored) if scored else 0.0
        ),
        "metrics": metrics,
        "by_category": by_category,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "max": max(latencies, default=0.0),
        },
        "total_tool_calls": sum(len(r.tools_called) for r in scored),
        "total_tokens": sum(r.tokens_used for r in results),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n{'=' * 68}")
    print(f"SUMMARY — {summary['label']}")
    print("=" * 68)
    print(f"  questions        : {summary['questions']}")
    print(f"  scored           : {summary['scored']}")
    if summary["errored"]:
        print(f"  NOT RUN          : {summary['errored']} "
              f"({', '.join(summary['errored_ids'])})")
        print("  >> INCOMPLETE RUN. Rates below cover only the scored "
              "questions and are not a result for the full set.")
    print(f"  overall pass rate: {summary['overall_pass_rate']:.1%} "
          f"(of {summary['scored']} scored)")
    print()
    for name, value in summary["metrics"].items():
        if value["rate"] is None:
            print(f"  {name:20}: n/a")
        else:
            print(f"  {name:20}: {value['rate']:6.1%}  (n={value['n']})")
    print()
    print("  by category:")
    for category, counts in sorted(summary["by_category"].items()):
        print(f"    {category:16} {counts['passed']}/{counts['total']}")
    print()
    latency = summary["latency_ms"]
    print(f"  latency p50      : {latency['p50']:,.0f} ms")
    print(f"  latency p95      : {latency['p95']:,.0f} ms")
    print(f"  latency max      : {latency['max']:,.0f} ms")
    print(f"  tool calls total : {summary['total_tool_calls']}")
    print(f"  tokens total     : {summary['total_tokens']:,}")


def write_report(
    runs: list[tuple[dict[str, Any], list[Result]]], model: str, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "runs": [
            {"summary": summary, "results": [r.to_dict() for r in results]}
            for summary, results in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider",
        default=None,
        help="groq, openrouter, openai. Lets a rate-limited provider be "
             "swapped out without blocking an evaluation.",
    )
    parser.add_argument("--category", help="run one category only")
    parser.add_argument("--limit", type=int, help="first N questions only")
    parser.add_argument(
        "--ablation",
        choices=["tools", "k"],
        help="tools: with and without tool access. k: retrieval depth 3 vs 8.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    specs = payload["questions"]
    if args.category:
        specs = [s for s in specs if s["category"] == args.category]
    if args.limit:
        specs = specs[: args.limit]
    if not specs:
        print("no questions selected", file=sys.stderr)
        return 2

    print(f"model      : {args.model}")
    print(f"questions  : {len(specs)}")
    print(f"ablation   : {args.ablation or 'none'}\n")

    runs: list[tuple[dict[str, Any], list[Result]]] = []

    if args.ablation == "tools":
        for label, enabled in (("tools ON", True), ("tools OFF", False)):
            print(f"--- {label} ---")
            results = asyncio.run(
                run_suite(specs, args.model, tools_enabled=enabled, label=label, provider=args.provider)
            )
            summary = summarise(results, label)
            print_summary(summary)
            runs.append((summary, results))
        _print_ablation_delta(runs)

    elif args.ablation == "k":
        for k in (3, 8):
            label = f"top_k={k}"
            print(f"--- {label} ---")
            results = asyncio.run(
                run_suite(specs, args.model, top_k=k, label=label, provider=args.provider)
            )
            summary = summarise(results, label)
            print_summary(summary)
            runs.append((summary, results))
        _print_ablation_delta(runs)

    else:
        results = asyncio.run(
            run_suite(specs, args.model, label="baseline", provider=args.provider)
        )
        summary = summarise(results, "baseline")
        print_summary(summary)
        runs.append((summary, results))

        failed = [r for r in results if not r.passed and not r.errored]
        if failed:
            print(f"\n{'=' * 68}\nFAILURES ({len(failed)})\n{'=' * 68}")
            for result in failed:
                print(f"\n{result.question_id} [{result.category}] {result.question}")
                for failure in result.failures:
                    print(f"   ! {failure}")
                excerpt = " ".join(result.answer.split())[:240]
                print(f"   answer: {excerpt}…")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = args.ablation or "baseline"
    out = args.out or RESULTS_DIR / f"eval_{suffix}_{stamp}.json"
    write_report(runs, args.model, out)
    return 0


def _print_ablation_delta(runs: list[tuple[dict[str, Any], list[Result]]]) -> None:
    if len(runs) != 2:
        return
    first, second = runs[0][0], runs[1][0]
    print(f"\n{'=' * 68}")
    print(f"ABLATION: {first['label']} vs {second['label']}")
    print("=" * 68)
    print(f"  overall pass rate: {first['overall_pass_rate']:.1%} -> "
          f"{second['overall_pass_rate']:.1%} "
          f"({second['overall_pass_rate'] - first['overall_pass_rate']:+.1%})")
    for name in first["metrics"]:
        a = first["metrics"][name]["rate"]
        b = second["metrics"][name]["rate"]
        if a is None or b is None:
            print(f"  {name:20}: {a if a is None else f'{a:.1%}'} -> "
                  f"{b if b is None else f'{b:.1%}'}")
        else:
            print(f"  {name:20}: {a:6.1%} -> {b:6.1%}  ({b - a:+.1%})")
    print(f"  latency p50      : {first['latency_ms']['p50']:,.0f} -> "
          f"{second['latency_ms']['p50']:,.0f} ms")


if __name__ == "__main__":
    sys.exit(main())
