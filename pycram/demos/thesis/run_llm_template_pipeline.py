"""Run LLM case generation and the existing ontology/fit pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LLM_STUFF_DIR = BASE_DIR / "llm-stuff"
DEFAULT_CASES = LLM_STUFF_DIR / "llm_action_cases.json"
DEFAULT_ONTOLOGY = LLM_STUFF_DIR / "llm_ontology_cases.json"
DEFAULT_RESULTS = LLM_STUFF_DIR / "llm_fit_results.json"
DEFAULT_SUMMARY = LLM_STUFF_DIR / "llm_fit_summary.json"
DEFAULT_REPORT = LLM_STUFF_DIR / "llm_fit_report.json"
DEFAULT_PROMPT = LLM_STUFF_DIR / "llm_prompt.txt"
DEFAULT_RESPONSE = LLM_STUFF_DIR / "llm_response.json"


def run_step(command: list[str]) -> None:
    print()
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbs", nargs="+", default=["cut", "mix", "pour", "wipe"])
    parser.add_argument(
        "--domains",
        nargs="+",
        default=[
            "food_preparation",
            "grooming",
            "gardening",
            "crafting",
            "construction",
            "cleaning",
        ],
    )
    parser.add_argument("--cases-per-verb", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--model")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cases-output", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--ontology-output", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prompt-output", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--response-output", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    LLM_STUFF_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable or "python3"

    generation_cmd = [
        python,
        str(BASE_DIR / "llm_case_generator.py"),
        "--verbs",
        *args.verbs,
        "--domains",
        *args.domains,
        "--cases-per-verb",
        str(args.cases_per_verb),
        "--batch-size",
        str(args.batch_size),
        "--max-retries",
        str(args.max_retries),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
        "--api-key-env",
        args.api_key_env,
        "--output",
        str(args.cases_output),
        "--prompt-output",
        str(args.prompt_output),
        "--response-output",
        str(args.response_output),
    ]
    if args.model:
        generation_cmd.extend(["--model", args.model])
    if args.api_base:
        generation_cmd.extend(["--api-base", args.api_base])
    if args.dry_run:
        generation_cmd.append("--dry-run")
        run_step(generation_cmd)
        return

    run_step(generation_cmd)
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


if __name__ == "__main__":
    main()
