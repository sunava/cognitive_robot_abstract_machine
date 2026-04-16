"""Extract structured action cases from locally collected WikiHow-style articles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from wikihow_eval.models import ActionCase, WikiHowArticle


TITLE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"^How to (?P<verb>\w+)\s+(?:(?:a|an|the)\b\s*)?(?P<object>.+)$", re.IGNORECASE
    ),
    re.compile(r"^How to (?P<verb>\w+)\s+Down\s+(?P<object>.+)$", re.IGNORECASE),
    re.compile(
        r"^.+?:\s*How to (?P<verb>\w+)\s+(?:(?:a|an|the)\b\s*)?(?P<object>.+)$",
        re.IGNORECASE,
    ),
)

TOOL_HINTS = (
    "knife",
    "scissors",
    "whisk",
    "spoon",
    "pitcher",
    "cloth",
    "sponge",
    "rag",
    "mop",
)


def load_articles(path: Path) -> List[WikiHowArticle]:
    raw_articles = json.loads(path.read_text(encoding="utf-8"))
    return [WikiHowArticle(**article) for article in raw_articles]


def infer_tool_hint(steps: Iterable[str]) -> Optional[str]:
    """Find the first tool mention in the article steps."""
    for step in steps:
        step_lower = step.lower()
        for hint in TOOL_HINTS:
            if hint in step_lower:
                return hint
    return None


def infer_domain_hint(categories: Sequence[str]) -> Optional[str]:
    for category in categories:
        lowered = category.lower()
        if "food" in lowered or "baking" in lowered or "vegetable" in lowered:
            return "food_preparation"
        if "clean" in lowered or "housekeeping" in lowered:
            return "cleaning"
        if "hair" in lowered or "personal care" in lowered:
            return "grooming"
        if "construction" in lowered or "home improvement" in lowered:
            return "construction"
    return None


def _normalize_object_text(text: str) -> str:
    """Normalize extracted object strings into a compact object phrase."""
    object_text = text.strip()
    object_text = re.sub(r"\s*\([^)]*\)", "", object_text)
    object_text = re.sub(r"\s*[-:]\s*.*$", "", object_text)
    object_text = re.sub(r"^(and|or)\s+use\s+it$", "", object_text, flags=re.IGNORECASE)
    object_text = re.sub(
        r"\s+(and|or)\s+use\s+it$", "", object_text, flags=re.IGNORECASE
    )
    object_text = re.sub(
        r"\s+(properly|easily|safely|quickly)$", "", object_text, flags=re.IGNORECASE
    )
    return object_text.strip().lower()


def _extract_from_title(title: str) -> tuple[str, str]:
    """Extract verb and object from common WikiHow title variants."""
    for pattern in TITLE_PATTERNS:
        match = pattern.match(title.strip())
        if match:
            verb = match.group("verb").lower()
            object_text = _normalize_object_text(match.group("object"))
            if not object_text and ":" in title:
                prefix = title.split(":", 1)[0]
                prefix = re.sub(
                    r"\bexplained\b$", "", prefix, flags=re.IGNORECASE
                ).strip()
                object_text = _normalize_object_text(prefix)
            if object_text:
                return verb, object_text

    marker = re.search(r"How to (?P<verb>\w+)\s+", title, re.IGNORECASE)
    if marker:
        verb = marker.group("verb").lower()
        object_text = _normalize_object_text(title[marker.end() :])
        if object_text:
            return verb, object_text

    raise ValueError(f"Could not parse action title: {title}")


def extract_case(article: WikiHowArticle) -> ActionCase:
    """Extract verb, object, tool hint, and domain hint from one article."""
    verb, object_text = _extract_from_title(article.title)
    return ActionCase(
        title=article.title,
        verb=verb,
        action_word=verb,
        object_text=object_text,
        tool_hint=infer_tool_hint(article.steps),
        domain_hint=infer_domain_hint(article.categories),
        categories=list(article.categories),
        steps=list(article.steps),
        url=article.url,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-output", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    error_output = args.error_output or args.output.with_name(
        args.output.stem + "_errors.jsonl"
    )
    cases = []
    errors = []
    for article in load_articles(args.input):
        try:
            cases.append(extract_case(article))
        except ValueError as exc:
            errors.append(
                {
                    "title": article.title,
                    "url": article.url,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    args.output.write_text(
        json.dumps([case.to_dict() for case in cases], indent=2), encoding="utf-8"
    )
    if errors:
        with error_output.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, ensure_ascii=True))
                handle.write("\n")
    print(
        f"Wrote {len(cases)} extracted action cases to {args.output}; "
        f"errors={len(errors)}"
    )


if __name__ == "__main__":
    main()
