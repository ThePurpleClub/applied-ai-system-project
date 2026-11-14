"""Run a fixed set of requests through the main pipeline and print metrics.

Grounding rate is the share of song titles Claude proposed that were in the
retrieved set. When Claude never proposes songs (for example, no API key),
the rate is the share of shown songs that came from retrieval, which is the
list the user actually sees.

Consistency rate is the share of prompts whose repeated runs returned the
same titles in the same order. Fallback rate is the share of runs that showed
the deterministic scorer instead of Claude's list.
"""

import sys
from typing import Callable, Dict, Optional, Sequence

from src.pipeline import TextClient, run_pipeline

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


def main() -> int:
    report = evaluate()
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
