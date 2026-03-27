"""Analyze why scraped WikiHow data is a weak fit for template-scope evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import extract_action_cases as extract_mod
import scrape_wikihow_actions as scrape_mod
from wikihow_eval.models import ActionCase, OntologyCase, WikiHowArticle
from wikihow_eval.ontology import map_case_to_ontology
from wikihow_eval.templates import score_case


DEFAULT_RAW_INPUT = (
    Path(__file__).resolve().parent / "wiki-stuff" / "raw" / "wikihow_articles.jsonl"
)
DEFAULT_ERROR_INPUT = (
    Path(__file__).resolve().parent
    / "wiki-stuff"
    / "raw"
    / "wikihow_articles_errors.jsonl"
)
DEFAULT_REPORT_OUTPUT = (
    Path(__file__).resolve().parent / "wiki-stuff" / "wikihow_feasibility_report.json"
)


def load_raw_records(path: Path) -> List[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def load_error_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _safe_extract_case(article: WikiHowArticle) -> ActionCase | None:
    try:
        return extract_mod.extract_case(article)
    except ValueError:
        return None


def _fit_counts(results: Iterable[dict]) -> Dict[str, int]:
    counts = {"full_fit": 0, "partial_fit": 0, "out_of_scope": 0}
    for result in results:
        counts[result["fit"]] += 1
    return counts


def build_report(
    articles: List[WikiHowArticle], raw_records: List[dict], errors: List[dict]
) -> Dict[str, Any]:
    total_articles = len(articles)
    deduped_articles = scrape_mod.deduplicate_articles(articles, dedupe_by="url")
    kitchen_articles = scrape_mod.filter_kitchen_articles(deduped_articles)
    non_kitchen_titles = [
        article.title for article in deduped_articles if article not in kitchen_articles
    ]

    by_query = Counter()
    by_title_pattern = Counter()
    parse_failures: List[dict] = []
    parsed_cases: List[ActionCase] = []
    ontology_cases: List[OntologyCase] = []
    raw_fit_results: List[dict] = []

    source_query_by_url = {
        record.get("url"): record.get("source_query", "unknown") for record in raw_records
    }
    for article in deduped_articles:
        by_query[source_query_by_url.get(article.url, "unknown")] += 1
        title_lower = article.title.lower()
        if title_lower.startswith("how to "):
            by_title_pattern["how_to"] += 1
        elif "how to " in title_lower:
            by_title_pattern["embedded_how_to"] += 1
        else:
            by_title_pattern["non_instructional"] += 1

        extracted = _safe_extract_case(article)
        if extracted is None:
            parse_failures.append({"title": article.title, "url": article.url})
            continue
        parsed_cases.append(extracted)
        ontology_case = map_case_to_ontology(extracted)
        ontology_cases.append(ontology_case)
        for result in score_case(ontology_case):
            entry = result.to_dict()
            entry["fit_case_verb"] = extracted.verb
            entry["fit_case_object"] = extracted.object_text
            raw_fit_results.append(entry)

    object_counter = Counter(case.object_class for case in ontology_cases)
    domain_counter = Counter(case.domain for case in ontology_cases)
    fit_counter_by_template = defaultdict(lambda: {"full_fit": 0, "partial_fit": 0, "out_of_scope": 0})
    for result in raw_fit_results:
        fit_counter_by_template[result["template"]][result["fit"]] += 1

    articles_with_unknown_objects = sum(
        1 for case in ontology_cases if case.object_class == "UnknownObject"
    )
    articles_with_generic_domain = sum(1 for case in ontology_cases if case.domain == "generic")
    non_instructional_articles = by_title_pattern["non_instructional"]

    report = {
        "article_count_raw": total_articles,
        "article_count_deduped": len(deduped_articles),
        "error_count": len(errors),
        "errors_by_query": dict(Counter(error["source_query"] for error in errors)),
        "articles_by_query": dict(by_query),
        "title_pattern_counts": dict(by_title_pattern),
        "kitchen_article_count": len(kitchen_articles),
        "non_kitchen_article_count": len(deduped_articles) - len(kitchen_articles),
        "kitchen_fraction": round(len(kitchen_articles) / len(deduped_articles), 3)
        if deduped_articles
        else 0.0,
        "parse_success_count": len(parsed_cases),
        "parse_failure_count": len(parse_failures),
        "parse_success_fraction": round(len(parsed_cases) / len(deduped_articles), 3)
        if deduped_articles
        else 0.0,
        "unknown_object_count": articles_with_unknown_objects,
        "unknown_object_fraction": round(
            articles_with_unknown_objects / len(ontology_cases), 3
        )
        if ontology_cases
        else 0.0,
        "generic_domain_count": articles_with_generic_domain,
        "generic_domain_fraction": round(
            articles_with_generic_domain / len(ontology_cases), 3
        )
        if ontology_cases
        else 0.0,
        "non_instructional_fraction": round(
            non_instructional_articles / len(deduped_articles), 3
        )
        if deduped_articles
        else 0.0,
        "top_object_classes": object_counter.most_common(10),
        "top_domains": domain_counter.most_common(10),
        "template_fit_counts": dict(fit_counter_by_template),
        "reasons_not_feasible": [
            "Search results are semantically broad and include many non-kitchen or non-manipulation pages.",
            "A large share of articles cannot be mapped to a useful object class or domain.",
            "Many titles are not clean imperative action instances, reducing extraction quality.",
            "Template-fit statistics are dominated by semantic normalization gaps rather than template scope alone.",
        ],
        "example_non_kitchen_titles": non_kitchen_titles[:15],
        "example_parse_failures": parse_failures[:15],
    }
    return report


def build_terminal_summary(report: Dict[str, Any]) -> str:
    lines = [
        f"Raw articles: {report['article_count_raw']}",
        f"Deduped articles: {report['article_count_deduped']}",
        f"Search errors: {report['error_count']}",
        f"Kitchen fraction: {report['kitchen_fraction']:.3f}",
        f"Parse success fraction: {report['parse_success_fraction']:.3f}",
        f"Unknown object fraction: {report['unknown_object_fraction']:.3f}",
        f"Generic domain fraction: {report['generic_domain_fraction']:.3f}",
        f"Non-instructional title fraction: {report['non_instructional_fraction']:.3f}",
        "Template fit counts:",
    ]
    for template, counts in sorted(report["template_fit_counts"].items()):
        lines.append(
            f"- {template}: full={counts['full_fit']} partial={counts['partial_fit']} out={counts['out_of_scope']}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument("--error-input", type=Path, default=DEFAULT_ERROR_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_records = load_raw_records(args.raw_input)
    articles = [scrape_mod._coerce_article_record(record) for record in raw_records]
    errors = load_error_records(args.error_input)
    report = build_report(articles, raw_records, errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(build_terminal_summary(report))
    print()
    print(f"Saved feasibility report to {args.output}")


if __name__ == "__main__":
    main()
