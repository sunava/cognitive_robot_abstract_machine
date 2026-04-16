"""Collect WikiHow-style action articles for offline evaluation.

The script can merge many local JSON/JSONL article dumps, filter them by verb,
deduplicate entries, and emit one normalized article list for the downstream
extraction step. It stays offline on purpose, so large WikiHow batches can be
prepared externally and then processed reproducibly here.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from wikihow_eval.models import WikiHowArticle


DEFAULT_VERBS = ("cut", "mix", "pour", "wipe")
DEFAULT_SEED_FILE = (
    Path(__file__).resolve().parent
    / "wikihow_eval"
    / "data"
    / "seed_wikihow_articles.json"
)
KITCHEN_CATEGORY_KEYWORDS = (
    "food and entertaining",
    "food",
    "cooking",
    "baking",
    "recipes",
    "kitchen",
    "vegetables",
    "fruit",
    "drinks",
)
KITCHEN_TITLE_KEYWORDS = (
    "banana",
    "bread",
    "carrot",
    "cucumber",
    "onion",
    "potato",
    "apple",
    "mango",
    "avocado",
    "cake",
    "cookie",
    "dough",
    "batter",
    "salad",
    "soup",
    "rice",
    "pasta",
    "egg",
    "cheese",
    "chicken",
    "meat",
    "vegetable",
    "fruit",
    "herb",
    "sauce",
)
KITCHEN_STEP_KEYWORDS = (
    "knife",
    "cutting board",
    "bowl",
    "pan",
    "pot",
    "whisk",
    "spoon",
    "spatula",
    "oven",
    "stove",
    "cook",
    "bake",
    "ingredients",
    "recipe",
    "mixing bowl",
)
NON_KITCHEN_CATEGORY_KEYWORDS = (
    "health",
    "personal care and style",
    "hair care",
    "beauty",
    "makeup",
    "first aid and emergency health care",
    "injuries and medical emergencies",
    "grooming",
)
NON_KITCHEN_TITLE_KEYWORDS = (
    "hair",
    "cuts quickly",
    "infected cut",
    "deep cuts",
    "nose",
    "tongue",
    "mouth",
    "wolf cut",
    "man's hair",
    "fake cuts",
    "makeup",
    "heal",
    "treat",
)


WIKIHOW_ARTICLE_FIELDS = {"title", "categories", "steps", "url"}


def _coerce_article_record(record: dict) -> WikiHowArticle:
    """Drop downloader-only metadata and build a normalized article object."""
    normalized = {
        key: value for key, value in record.items() if key in WIKIHOW_ARTICLE_FIELDS
    }
    return WikiHowArticle(**normalized)


def _load_json_records(path: Path) -> List[dict]:
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_payload, list):
        return raw_payload
    if isinstance(raw_payload, dict):
        if "articles" in raw_payload and isinstance(raw_payload["articles"], list):
            return raw_payload["articles"]
        return [raw_payload]
    raise ValueError(f"Unsupported JSON payload in {path}")


def _load_jsonl_records(path: Path) -> List[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def load_articles(path: Path) -> List[WikiHowArticle]:
    """Load one local article dump."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        raw_articles = _load_jsonl_records(path)
    else:
        raw_articles = _load_json_records(path)
    return [_coerce_article_record(article) for article in raw_articles]


def iter_input_files(
    seed_file: Path,
    input_files: Sequence[Path],
    input_glob: str | None,
    skip_seed: bool,
) -> Iterator[Path]:
    """Yield all configured source files in a stable order."""
    yielded = OrderedDict()
    if not skip_seed:
        yielded[seed_file.resolve()] = seed_file
    for path in input_files:
        yielded[path.resolve()] = path
    if input_glob:
        for path in sorted(Path().glob(input_glob)):
            yielded[path.resolve()] = path
    yield from yielded.values()


def deduplicate_articles(
    articles: Sequence[WikiHowArticle], dedupe_by: str
) -> List[WikiHowArticle]:
    """Drop duplicate articles by title or URL."""
    deduped = OrderedDict()
    for article in articles:
        if dedupe_by == "url" and article.url:
            key = article.url.strip().lower()
        else:
            key = article.title.strip().lower()
        deduped[key] = article
    return list(deduped.values())


def filter_articles(
    articles: Sequence[WikiHowArticle], verbs: Iterable[str]
) -> List[WikiHowArticle]:
    """Keep only articles whose titles contain a ``How to <verb>`` phrase."""
    patterns = [
        re.compile(rf"\bhow to {re.escape(verb.lower())}\b", re.IGNORECASE)
        for verb in verbs
    ]
    selected: List[WikiHowArticle] = []
    for article in articles:
        if any(pattern.search(article.title) for pattern in patterns):
            selected.append(article)
    return selected


def filter_kitchen_articles(articles: Sequence[WikiHowArticle]) -> List[WikiHowArticle]:
    """Keep only articles that look like kitchen-domain instructions."""
    selected: List[WikiHowArticle] = []
    for article in articles:
        category_text = " ".join(article.categories).lower()
        title_text = article.title.lower()
        step_text = " ".join(article.steps).lower()
        if any(keyword in category_text for keyword in NON_KITCHEN_CATEGORY_KEYWORDS):
            continue
        if any(keyword in title_text for keyword in NON_KITCHEN_TITLE_KEYWORDS):
            continue

        kitchen_signals = 0
        if any(keyword in category_text for keyword in KITCHEN_CATEGORY_KEYWORDS):
            kitchen_signals += 2
        if any(keyword in title_text for keyword in KITCHEN_TITLE_KEYWORDS):
            kitchen_signals += 1
        if any(keyword in step_text for keyword in KITCHEN_STEP_KEYWORDS):
            kitchen_signals += 1
        if kitchen_signals >= 2:
            selected.append(article)
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--input-file", type=Path, action="append", default=[])
    parser.add_argument("--input-glob")
    parser.add_argument("--verbs", nargs="+", default=list(DEFAULT_VERBS))
    parser.add_argument("--dedupe-by", choices=("title", "url"), default="url")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kitchen-only", action="store_true")
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    all_articles: List[WikiHowArticle] = []
    input_paths = list(
        iter_input_files(
            args.seed_file, args.input_file, args.input_glob, args.skip_seed
        )
    )
    for path in input_paths:
        all_articles.extend(load_articles(path))
    articles = filter_articles(
        deduplicate_articles(all_articles, args.dedupe_by), args.verbs
    )
    if args.kitchen_only:
        articles = filter_kitchen_articles(articles)
    if args.limit is not None:
        articles = articles[: args.limit]
    args.output.write_text(
        json.dumps([article.to_dict() for article in articles], indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(articles)} filtered articles to {args.output} "
        f"from {len(input_paths)} source file(s)"
    )


if __name__ == "__main__":
    main()
