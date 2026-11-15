"""Measure the baseline renderer against the music-curator renderer.

Both renderers are offline templates. They format the same retrieved songs,
so the difference is style, not a paid model call. Pass --live only when you
intend to spend API credit; this script then asks before it calls Claude.
"""

import argparse
import os
import sys

from src.pipeline import load_local_env, run_pipeline
from src.styles import (
    average_reason_words,
    baseline_text,
    curator_text,
    format_compliance,
    grounding_rate,
    style_marker_rate,
)

QUERIES = (
    "upbeat happy pop",
    "songs for a late-night drive",
    "calm instrumental lofi for studying",
)


def _metrics(text: str, titles) -> dict:
    return {
        "format_compliance": format_compliance(text),
        "avg_reason_words": average_reason_words(text),
        "grounding_rate": grounding_rate(text, titles),
        "style_marker_rate": style_marker_rate(text),
    }


def compare(queries=QUERIES, offline: bool = True) -> str:
    """Return a metrics table. offline=True never calls Claude."""
    rows = []
    samples = []
    for query in queries:
        result = run_pipeline(query, offline=offline, save_trace=False, style="baseline")
        titles = [song["title"] for song, _score, _reasons in result.retrieved]
        baseline = baseline_text(query, result.profile or {}, result.recommendations)
        curator = curator_text(query, result.profile or {}, result.recommendations)
        rows.append(("baseline", query, _metrics(baseline, titles)))
        rows.append(("curator", query, _metrics(curator, titles)))
        samples.append((query, baseline, curator))
    lines = [
        "# source: offline deterministic renderers" if offline else "# source: live Claude plus the same metric functions",
        "style      query                                format  words  grounding  markers",
    ]
    for style, query, metrics in rows:
        short = query if len(query) <= 36 else query[:33] + "..."
        lines.append(
            f"{style:<10} {short:<36} {metrics['format_compliance']:>6.2f} "
            f"{metrics['avg_reason_words']:>6.1f} {metrics['grounding_rate']:>10.2f} "
            f"{metrics['style_marker_rate']:>8.2f}"
        )
    base_words = [metrics["avg_reason_words"] for style, _q, metrics in rows if style == "baseline"]
    curator_words = [metrics["avg_reason_words"] for style, _q, metrics in rows if style == "curator"]
    base_format = [metrics["format_compliance"] for style, _q, metrics in rows if style == "baseline"]
    curator_format = [metrics["format_compliance"] for style, _q, metrics in rows if style == "curator"]
    lines.append("")
    lines.append(
        f"Mean reason words: baseline {sum(base_words)/len(base_words):.1f}, "
        f"curator {sum(curator_words)/len(curator_words):.1f}"
    )
    lines.append(
        f"Mean format compliance: baseline {sum(base_format)/len(base_format):.2f}, "
        f"curator {sum(curator_format)/len(curator_format):.2f}"
    )
    lines.append("")
    lines.append("Example (first query)")
    lines.append("--- baseline ---")
    lines.append(samples[0][1])
    lines.append("--- curator ---")
    lines.append(samples[0][2])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and curator playlist text.")
    parser.add_argument("--live", action="store_true", help="Call Claude. Asks for confirmation first.")
    parser.add_argument("--yes", action="store_true", help="Confirm paid Claude calls.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    offline = True
    if args.live:
        load_local_env()
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            print("No API key set. Staying on the offline renderers.")
        elif not args.yes:
            if not sys.stdin.isatty():
                print("Refusing paid Claude calls without --yes.")
                return 2
            answer = input("This makes paid Claude calls. Type yes to continue: ")
            if answer.strip().lower() != "yes":
                print("Cancelled.")
                return 2
            offline = False
        else:
            offline = False
    print(compare(offline=offline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
