"""Score ontology-mapped action cases against thesis action-template profiles."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from wikihow_eval.models import OntologyCase
from wikihow_eval.templates import score_case, summarize_results


def load_cases(path: Path) -> List[OntologyCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [OntologyCase(**case) for case in raw_cases]


def build_markdown_table(results: List[dict]) -> str:
    lines = [
        "| WikiHow example | mapped template | fit | score |",
        "| --- | --- | --- | ---: |",
    ]
    for result in results:
        example = f"{result['fit_case_verb']} {result['fit_case_object']}"
        lines.append(
            f"| {example} | {result['template']} | {result['fit']} | {result['score']:.3f} |"
        )
    return "\n".join(lines)


def build_coverage_report(
    cases: List[OntologyCase], results: List[dict]
) -> Dict[str, object]:
    """Aggregate coverage and out-of-scope clusters for larger evaluations."""
    by_template = summarize_results(_dict_to_result_like(results))
    by_verb = Counter(case.verb for case in cases)
    out_of_scope_domains = Counter()
    out_of_scope_objects = Counter()
    out_of_scope_articles = []
    fit_by_template_scores = defaultdict(list)

    case_by_title = {case.title: case for case in cases}
    for result in results:
        fit_by_template_scores[result["template"]].append(result["score"])
        if result["fit"] != "out_of_scope":
            continue
        case = case_by_title[result["article"]]
        out_of_scope_domains[case.domain] += 1
        out_of_scope_objects[case.object_class] += 1
        out_of_scope_articles.append(
            {
                "article": case.title,
                "verb": case.verb,
                "object_text": case.object_text,
                "object_class": case.object_class,
                "domain": case.domain,
                "template": result["template"],
                "reason": result["reason"],
            }
        )

    template_rates = {}
    for template, counts in by_template.items():
        total = sum(counts.values())
        template_rates[template] = {
            **counts,
            "total": total,
            "full_fit_rate": round(counts["full_fit"] / total, 3) if total else 0.0,
            "partial_fit_rate": (
                round(counts["partial_fit"] / total, 3) if total else 0.0
            ),
            "out_of_scope_rate": (
                round(counts["out_of_scope"] / total, 3) if total else 0.0
            ),
            "mean_score": (
                round(
                    sum(fit_by_template_scores[template])
                    / len(fit_by_template_scores[template]),
                    3,
                )
                if fit_by_template_scores[template]
                else 0.0
            ),
        }

    return {
        "article_count": len(cases),
        "evaluation_count": len(results),
        "articles_by_verb": dict(by_verb),
        "template_coverage": template_rates,
        "top_out_of_scope_domains": out_of_scope_domains.most_common(10),
        "top_out_of_scope_object_classes": out_of_scope_objects.most_common(10),
        "out_of_scope_examples": out_of_scope_articles[:25],
    }


def _dict_to_result_like(results: List[dict]):
    for result in results:
        yield _ResultLike(template=result["template"], fit=result["fit"])


class _ResultLike:
    def __init__(self, template: str, fit: str) -> None:
        self.template = template
        self.fit = fit


def build_terminal_report(report: Dict[str, object]) -> str:
    lines = [
        f"Articles: {report['article_count']}",
        f"Evaluations: {report['evaluation_count']}",
        "Coverage by template:",
    ]
    template_coverage = report["template_coverage"]
    assert isinstance(template_coverage, dict)
    for template, stats in sorted(template_coverage.items()):
        lines.append(
            f"- {template}: full={stats['full_fit']} partial={stats['partial_fit']} "
            f"out={stats['out_of_scope']} mean_score={stats['mean_score']:.3f}"
        )
    top_domains = report["top_out_of_scope_domains"]
    if top_domains:
        lines.append("Top out-of-scope domains:")
        for domain, count in top_domains[:5]:
            lines.append(f"- {domain}: {count}")
    top_objects = report["top_out_of_scope_object_classes"]
    if top_objects:
        lines.append("Top out-of-scope object classes:")
        for object_class, count in top_objects[:5]:
            lines.append(f"- {object_class}: {count}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--max-table-rows", type=int, default=40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = load_cases(args.input)
    raw_results = []
    for case in cases:
        for result in score_case(case):
            entry = result.to_dict()
            entry["fit_case_verb"] = case.verb
            entry["fit_case_object"] = case.object_text
            raw_results.append(entry)
    args.output.write_text(json.dumps(raw_results, indent=2), encoding="utf-8")
    if args.summary_output:
        args.summary_output.write_text(
            json.dumps(
                summarize_results(
                    score for case in cases for score in score_case(case)
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    report = build_coverage_report(cases, raw_results)
    if args.report_output:
        args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(build_terminal_report(report))
    print()
    print(build_markdown_table(raw_results[: args.max_table_rows]))


if __name__ == "__main__":
    main()
