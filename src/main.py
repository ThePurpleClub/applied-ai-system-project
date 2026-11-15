"""Command line runner for the music recommender.

Ask in a sentence. The pipeline parses it, retrieves songs, and explains them.
"""

import argparse
import sys

from src.pipeline import format_result, run_pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend songs from a natural-language request."
    )
    parser.add_argument(
        "request",
        nargs="*",
        help='Music request, for example: upbeat songs for a late-night drive',
    )
    args = parser.parse_args(argv)
    text = " ".join(args.request).strip()
    if not text:
        try:
            text = input("What do you want to listen to? ").strip()
        except EOFError:
            text = ""

    result = run_pipeline(text)
    print(format_result(result))
    if result.mode == "invalid_input":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
