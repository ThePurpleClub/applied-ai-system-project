"""Run a fixed set of requests through the main pipeline and print metrics.

Grounding rate is the share of song titles Claude proposed that were in the
retrieved set. When Claude never proposes songs (for example, no API key),
the rate is the share of shown songs that came from retrieval, which is the
list the user actually sees.

Consistency rate is the share of prompts whose repeated runs returned the
same titles in the same order. Fallback rate is the share of runs that showed
the deterministic scorer instead of Claude's list.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from src.pipeline import TextClient, load_local_env, run_pipeline
from src.recommender import max_score

CASES_PATH = Path(__file__).resolve().parent / "cases.json"

TEST_PROMPTS = [
    "upbeat songs for a late-night drive",
    "calm instrumental lofi for studying",
    "sad blues about heartbreak",
    "heavy angry metal",
    "family-friendly happy pop with no explicit lyrics",
    "moody 1980s synthwave with no lyrics",
]


def evaluate(
    prompts: Sequence[str] = TEST_PROMPTS,
    repeats: int = 2,
    client: Optional[TextClient] = None,
    pipeline: Callable = run_pipeline,
) -> Dict:
    """Reuse run_pipeline for every prompt and summarize the runs."""
    details = []
    proposed = 0
    grounded_proposals = 0
    shown = 0
    shown_grounded = 0
    fallback_runs = 0
    total_runs = 0
    consistent_prompts = 0
    comparable_prompts = 0
    self_checks = 0
    self_checks_passed = 0

    for prompt in prompts:
        runs = [pipeline(prompt, client=client) for _ in range(repeats)]
        title_lists = []
        for result in runs:
            total_runs += 1
            if result.mode == "fallback":
                fallback_runs += 1
            if result.mode == "invalid_input":
                continue
            retrieved = {
                (song["title"].strip().lower(), song["artist"].strip().lower())
                for song, _score, _reasons in result.retrieved
            }
            for rec in result.recommendations:
                shown += 1
                if (rec.title.strip().lower(), rec.artist.strip().lower()) in retrieved:
                    shown_grounded += 1
            if result.proposed_count:
                proposed += result.proposed_count
                grounded_proposals += result.grounded_count
            if result.self_check is not None:
                self_checks += 1
                if result.self_check.get("passed"):
                    self_checks_passed += 1
            title_lists.append(tuple(rec.title for rec in result.recommendations))

        if title_lists:
            comparable_prompts += 1
            if len(set(title_lists)) == 1:
                consistent_prompts += 1
        details.append({
            "prompt": prompt,
            "modes": [result.mode for result in runs],
            "titles": title_lists,
            "hallucinations": [list(result.hallucinations) for result in runs],
        })

    if proposed:
        grounding_rate = grounded_proposals / proposed
    elif shown:
        grounding_rate = shown_grounded / shown
    else:
        grounding_rate = 1.0

    return {
        "prompts": len(prompts),
        "runs": total_runs,
        "grounding_rate": grounding_rate,
        "proposal_grounding_rate": (grounded_proposals / proposed) if proposed else None,
        "output_grounding_rate": (shown_grounded / shown) if shown else None,
        "consistency_rate": (consistent_prompts / comparable_prompts) if comparable_prompts else 1.0,
        "fallback_rate": (fallback_runs / total_runs) if total_runs else 0.0,
        "self_check_pass_rate": (self_checks_passed / self_checks) if self_checks else None,
        "details": details,
    }


def format_report(report: Dict) -> str:
    proposal = report["proposal_grounding_rate"]
    proposal_text = "n/a" if proposal is None else f"{proposal:.2f}"
    output = report["output_grounding_rate"]
    output_text = "n/a" if output is None else f"{output:.2f}"
    lines = [
        "Evaluation",
        f"  prompts: {report['prompts']}",
        f"  runs: {report['runs']}",
        f"  grounding rate: {report['grounding_rate']:.2f}",
        f"  Claude proposal grounding: {proposal_text}",
        f"  shown songs found in retrieval: {output_text}",
        f"  consistency rate: {report['consistency_rate']:.2f}",
        f"  fallback rate: {report['fallback_rate']:.2f}",
        "  self-check pass rate: "
        + ("n/a" if report.get("self_check_pass_rate") is None else f"{report['self_check_pass_rate']:.2f}"),
        "",
        "Per prompt",
    ]
    for detail in report["details"]:
        modes = ", ".join(detail["modes"])
        first_titles = detail["titles"][0] if detail["titles"] else ()
        title_text = ", ".join(first_titles) if first_titles else "(none)"
        lines.append(f"  - {detail['prompt']}")
        lines.append(f"    modes: {modes}")
        lines.append(f"    titles: {title_text}")
    return "\n".join(lines)


def load_cases(path: Optional[Path] = None) -> List[Dict]:
    source = path or CASES_PATH
    return json.loads(source.read_text(encoding="utf-8"))


def _add(checks: List[Dict], name: str, ok: bool) -> None:
    checks.append({"name": name, "ok": bool(ok)})


def check_case(case: Dict, result, offline: bool) -> List[Dict]:
    """Return the named checks for one predefined case."""
    expect = case.get("expect") or {}
    checks: List[Dict] = []
    if expect.get("invalid_input"):
        _add(checks, "invalid input rejected", result.mode == "invalid_input")
        return checks
    profile = result.profile or {}
    if offline and expect.get("offline_mode"):
        _add(checks, "fallback mode", result.mode == expect["offline_mode"])
    if "genre_contains" in expect:
        wanted = str(expect["genre_contains"]).lower()
        genres = [str(profile.get("genre") or "").lower()]
        genres.extend(rec.genre.lower() for rec in result.recommendations)
        _add(checks, f"genre contains {wanted}", any(wanted in genre for genre in genres if genre))
    if "mood" in expect:
        _add(checks, "mood", str(profile.get("mood") or "").lower() == str(expect["mood"]).lower())
    if "min_energy" in expect:
        _add(checks, "min energy", float(profile.get("energy") or 0) >= float(expect["min_energy"]))
    if "max_energy" in expect:
        _add(checks, "max energy", float(profile.get("energy") or 1) <= float(expect["max_energy"]))
    if "language" in expect:
        _add(checks, "language", str(profile.get("language") or "").lower() == str(expect["language"]).lower())
    if expect.get("allow_explicit") is False:
        _add(checks, "explicit filter", all(not rec.explicit for rec in result.recommendations))
    if expect.get("no_hallucinations"):
        retrieved = {song["title"].strip().lower() for song, _score, _reasons in result.retrieved}
        _add(
            checks,
            "no hallucinated titles",
            all(rec.title.strip().lower() in retrieved for rec in result.recommendations),
        )
    if "min_results" in expect:
        _add(checks, "enough songs", len(result.recommendations) >= int(expect["min_results"]))
    return checks


def case_confidence(result, checks: Sequence[Dict]) -> float:
    """Blend how many checks passed with how strong the top match is."""
    ratio = sum(1 for item in checks if item["ok"]) / len(checks) if checks else 0.0
    if result.mode == "invalid_input":
        return round(ratio, 2)
    if not result.recommendations or not result.profile or result.recommendations[0].score is None:
        return round(0.6 * ratio, 2)
    ceiling = max_score(result.profile) or 1.0
    strength = max(0.0, min(1.0, float(result.recommendations[0].score) / ceiling))
    return round(0.6 * ratio + 0.4 * strength, 2)


def evaluate_cases(
    cases: Optional[Sequence[Dict]] = None,
    offline: bool = False,
    client: Optional[TextClient] = None,
    pipeline: Callable = run_pipeline,
    songs=None,
) -> Dict:
    """Run the predefined cases and score each one pass or fail."""
    chosen = list(cases) if cases is not None else load_cases()
    rows = []
    for case in chosen:
        kwargs = {"client": None if offline else client}
        if songs is not None:
            kwargs["songs"] = songs
        result = pipeline(case.get("prompt") or "", **kwargs)
        checks = check_case(case, result, offline=offline)
        passed = bool(checks) and all(item["ok"] for item in checks)
        rows.append({
            "id": case.get("id") or case.get("prompt") or "case",
            "prompt": case.get("prompt") or "",
            "passed": passed,
            "confidence": case_confidence(result, checks),
            "checks": checks,
            "mode": result.mode,
            "titles": [rec.title for rec in result.recommendations],
        })
    passed_count = sum(1 for row in rows if row["passed"])
    total = len(rows)
    return {
        "passed": passed_count,
        "total": total,
        "pass_rate": (passed_count / total) if total else 0.0,
        "cases": rows,
    }


def format_case_report(report: Dict) -> str:
    lines = ["case                  result  confidence  notes"]
    for row in report["cases"]:
        failed = ", ".join(item["name"] for item in row["checks"] if not item["ok"]) or "-"
        result = "PASS" if row["passed"] else "FAIL"
        lines.append(f"{row['id']:<22} {result:<7} {row['confidence']:>10.2f}  {failed}")
    lines.append("")
    lines.append(
        f"Overall pass rate: {report['passed']}/{report['total']} ({report['pass_rate']:.0%})"
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score the recommender on predefined cases.")
    parser.add_argument("--offline", action="store_true", help="Do not call Claude, even if a key is set.")
    parser.add_argument("--yes", action="store_true", help="Confirm paid Claude calls.")
    parser.add_argument("--metrics", action="store_true", help="Print the older rate summary instead of the case table.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.metrics:
        print(format_report(evaluate()))
        return 0
    if not args.offline:
        load_local_env()
        if os.environ.get("ANTHROPIC_API_KEY", "").strip() and not args.yes:
            if not sys.stdin.isatty():
                print("Refusing paid Claude calls without --yes. Re-run with --offline.")
                return 2
            answer = input("This makes paid Claude calls. Type yes to continue: ")
            if answer.strip().lower() != "yes":
                print("Cancelled. Re-run with --offline for a free run.")
                return 2
    report = evaluate_cases(offline=args.offline or not os.environ.get("ANTHROPIC_API_KEY", "").strip())
    print(format_case_report(report))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
