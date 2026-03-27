"""Run the full WikiHow kitchen crawl and evaluation pipeline with repo-local paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
WIKI_STUFF_DIR = BASE_DIR / "wiki-stuff"
RAW_DIR = WIKI_STUFF_DIR / "raw"
DEFAULT_RAW = RAW_DIR / "wikihow_articles.jsonl"
DEFAULT_FILTERED = WIKI_STUFF_DIR / "wikihow_articles.json"
DEFAULT_CASES = WIKI_STUFF_DIR / "wikihow_cases.json"
DEFAULT_ONTOLOGY = WIKI_STUFF_DIR / "wikihow_ontology_cases.json"
DEFAULT_RESULTS = WIKI_STUFF_DIR / "wikihow_fit_results.json"
DEFAULT_SUMMARY = WIKI_STUFF_DIR / "wikihow_fit_summary.json"
DEFAULT_REPORT = WIKI_STUFF_DIR / "wikihow_fit_report.json"


def run_step(command: List[str]) -> None:
    """Run one pipeline step and stop on failure."""
    print()
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pages-per-verb", type=int, default=4)
    parser.add_argument("--max-articles-per-verb", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=1.5)
    parser.add_argument("--kitchen-only", action="store_true", default=True)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--filtered-output", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--cases-output", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--ontology-output", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    python = sys.executable or "python3"

    if args.download:
        command = [
            python,
            str(BASE_DIR / "download_wikihow_articles.py"),
            "--kitchen-verbs",
            "--pages-per-verb",
            str(args.pages_per_verb),
            "--max-articles-per-verb",
            str(args.max_articles_per_verb),
            "--sleep-seconds",
            str(args.sleep_seconds),
            "--output",
            str(args.raw_output),
        ]
        if args.kitchen_only:
            command.append("--kitchen-only")
        if args.resume:
            command.append("--resume")
        run_step(command)

    run_step(
        [
            python,
            str(BASE_DIR / "scrape_wikihow_actions.py"),
            "--skip-seed",
            "--input-file",
            str(args.raw_output),
            "--verbs",
            "cut",
            "chop",
            "slice",
            "dice",
            "mince",
            "peel",
            "grate",
            "shred",
            "mix",
            "stir",
            "whisk",
            "knead",
            "pour",
            "drain",
            "strain",
            "rinse",
            "wash",
            "season",
            "spread",
            "scoop",
            "--dedupe-by",
            "url",
            "--output",
            str(args.filtered_output),
        ]
    )
    run_step(
        [
            python,
            str(BASE_DIR / "extract_action_cases.py"),
            "--input",
            str(args.filtered_output),
            "--output",
            str(args.cases_output),
        ]
    )
    run_step(
        [
            python,
            str(BASE_DIR / "map_cases_to_ontology.py"),
            "--input",
            str(args.cases_output),
            "--output",
            str(args.ontology_output),
        ]
    )
    run_step(
        [
            python,
            str(BASE_DIR / "score_template_fit.py"),
            "--input",
            str(args.ontology_output),
            "--output",
            str(args.results_output),
            "--summary-output",
            str(args.summary_output),
            "--report-output",
            str(args.report_output),
        ]
    )

    report = json.loads(args.report_output.read_text(encoding="utf-8"))
    print()
    print("Pipeline complete.")
    print(f"Articles: {report['article_count']}")
    print(f"Evaluations: {report['evaluation_count']}")
    print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()
