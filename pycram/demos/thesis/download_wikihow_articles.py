"""Download WikiHow search results and article metadata into local JSONL dumps.

This script is intended for batch collection before the offline extraction and
template-fit evaluation pipeline. It uses a conservative request rate, writes one
JSON object per line, and stores raw article metadata under ``wiki-stuff/raw`` by
default.

Use it responsibly and check the target site's robots and terms before large runs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://www.wikihow.com"
DEFAULT_SEARCH_PATH = "/wikiHowTo?search={query}"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "wiki-stuff" / "raw" / "wikihow_articles.jsonl"
)
DEFAULT_SEARCH_PAGE_SIZE = 15
DEFAULT_VERBS = ("cut", "mix", "pour", "wipe", "peel", "scrub", "slice", "chop", "stir")
KITCHEN_VERBS = (
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
NON_ARTICLE_SLUGS = {
    "Main-Page",
    "Quizzes",
    "Pro",
    "About-wikiHow",
    "Special:CategoryListing",
    "Special:Randomizer",
    "Community",
    "About-This-Article",
}
NON_ARTICLE_PREFIXES = (
    "Special:",
    "Category:",
    "User:",
    "Topic:",
    "Video:",
)


@dataclass(frozen=True)
class DownloadedArticle:
    """Raw downloaded article entry persisted as JSONL."""

    title: str
    categories: List[str]
    steps: List[str]
    url: str
    source_query: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


@dataclass(frozen=True)
class DownloadError:
    """Structured error record for failed search or article fetches."""

    stage: str
    url: str
    source_query: str
    error_type: str
    error_message: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


class SearchResultsParser(HTMLParser):
    """Extract candidate WikiHow article links from a search page."""

    def __init__(self, base_url: str, query_term: Optional[str] = None) -> None:
        super().__init__()
        self.base_url = base_url
        self.query_term = (query_term or "").strip().lower()
        self.article_urls: List[str] = []
        self._search_container_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = dict(attrs)
        if tag == "div" and attr_map.get("id") == "searchresults_list":
            self._search_container_depth = 1
            return
        if self._search_container_depth and tag == "div":
            self._search_container_depth += 1
        if self._search_container_depth == 0:
            return
        if tag != "a":
            return
        href = attr_map.get("href")
        if not href:
            return
        class_attr = attr_map.get("class", "")
        if "result_link" not in class_attr.split():
            return
        absolute = urljoin(self.base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc and "wikihow.com" not in parsed.netloc:
            return
        if "/wikiHowTo" in parsed.path or "/Special:" in parsed.path:
            return
        slug = parsed.path.strip("/")
        if not slug or slug.startswith("images") or slug.startswith("video"):
            return
        if slug in NON_ARTICLE_SLUGS:
            return
        if any(slug.startswith(prefix) for prefix in NON_ARTICLE_PREFIXES):
            return
        if self.query_term and self.query_term not in slug.lower():
            return
        if absolute not in self.article_urls:
            self.article_urls.append(absolute)

    def handle_endtag(self, tag: str) -> None:
        if self._search_container_depth and tag == "div":
            self._search_container_depth -= 1


class ArticlePageParser(HTMLParser):
    """Extract title, categories, and visible step text from an article page."""

    def __init__(self) -> None:
        super().__init__()
        self._collect_title = False
        self._collect_step = False
        self._collect_category = False
        self._title_chunks: List[str] = []
        self._step_chunks: List[str] = []
        self._category_chunks: List[str] = []
        self.steps: List[str] = []
        self.categories: List[str] = []
        self.title: Optional[str] = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = dict(attrs)
        class_attr = attr_map.get("class", "")
        itemprop = attr_map.get("itemprop", "")
        data_test = attr_map.get("data-testid", "")
        href = attr_map.get("href", "")

        if tag in {"title", "h1"}:
            self._collect_title = True
        if (
            itemprop in {"name", "headline"}
            or "mf-section-0" in class_attr
            or data_test == "headline"
        ):
            self._collect_title = True
        if (
            itemprop == "text"
            or "step" in class_attr.lower()
            or data_test in {"step", "steps"}
        ):
            self._collect_step = True
        if (
            "/Category:" in href
            or "cat_" in class_attr.lower()
            or "breadcrumb" in class_attr.lower()
        ):
            self._collect_category = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"title", "h1"}:
            self._collect_title = False
        if self._collect_step and tag in {"div", "p", "li", "span", "b"}:
            text = _normalize_whitespace(" ".join(self._step_chunks))
            if text and len(text) > 15 and text not in self.steps:
                self.steps.append(text)
            self._step_chunks.clear()
            self._collect_step = False
        if self._collect_category and tag in {"a", "span", "li"}:
            text = _normalize_whitespace(" ".join(self._category_chunks))
            if text and text not in self.categories:
                self.categories.append(text)
            self._category_chunks.clear()
            self._collect_category = False

    def handle_data(self, data: str) -> None:
        cleaned = _normalize_whitespace(data)
        if not cleaned:
            return
        if self._collect_title:
            self._title_chunks.append(cleaned)
            if cleaned.lower().startswith("how to "):
                self.title = cleaned
        if self._collect_step:
            self._step_chunks.append(cleaned)
        if self._collect_category:
            self._category_chunks.append(cleaned)

    def close(self) -> None:
        super().close()
        if not self.title:
            combined = _normalize_whitespace(" ".join(self._title_chunks))
            if combined:
                self.title = combined.split(" - wikiHow", 1)[0]


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def fetch_html(url: str, timeout: float, user_agent: str) -> str:
    """Fetch one HTML page."""
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_article_urls(
    search_html: str,
    base_url: str = DEFAULT_BASE_URL,
    query_term: Optional[str] = None,
) -> List[str]:
    """Extract candidate article URLs from one search-result page."""
    parser = SearchResultsParser(base_url, query_term=query_term)
    parser.feed(search_html)
    parser.close()
    return parser.article_urls


def _extract_ld_json_objects(html: str) -> List[dict]:
    marker = '<script type="application/ld+json"'
    objects: List[dict] = []
    start = 0
    while True:
        idx = html.find(marker, start)
        if idx == -1:
            break
        start_tag_end = html.find(">", idx)
        end_idx = html.find("</script>", start_tag_end)
        if start_tag_end == -1 or end_idx == -1:
            break
        payload = html[start_tag_end + 1 : end_idx].strip()
        start = end_idx + len("</script>")
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            objects.extend(item for item in obj if isinstance(item, dict))
        elif isinstance(obj, dict):
            objects.append(obj)
    return objects


def _flatten_howto_steps(step_value) -> List[str]:
    if isinstance(step_value, str):
        return [_normalize_whitespace(step_value)]
    if isinstance(step_value, list):
        steps: List[str] = []
        for item in step_value:
            if isinstance(item, str):
                steps.append(_normalize_whitespace(item))
            elif isinstance(item, dict):
                text = item.get("text") or item.get("name")
                if text:
                    steps.append(_normalize_whitespace(text))
        return [step for step in steps if step]
    if isinstance(step_value, dict):
        text = step_value.get("text") or step_value.get("name")
        return [_normalize_whitespace(text)] if text else []
    return []


def parse_article_html(html: str, url: str, source_query: str) -> DownloadedArticle:
    """Parse one article page, preferring structured HowTo metadata when present."""
    for obj in _extract_ld_json_objects(html):
        graph = obj.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict) and item.get("@type") in {"HowTo", ["HowTo"]}:
                    obj = item
                    break
        obj_type = obj.get("@type")
        if obj_type not in {"HowTo", "Article"} and obj_type != ["HowTo"]:
            continue
        title = obj.get("name") or obj.get("headline")
        steps = _flatten_howto_steps(obj.get("step"))
        categories = []
        breadcrumbs = obj.get("breadcrumb")
        if isinstance(breadcrumbs, list):
            for item in breadcrumbs:
                if isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        categories.append(_normalize_whitespace(name))
        if title and steps:
            return DownloadedArticle(
                title=_normalize_whitespace(title),
                categories=categories,
                steps=steps,
                url=url,
                source_query=source_query,
            )

    parser = ArticlePageParser()
    parser.feed(html)
    parser.close()
    if not parser.title:
        raise ValueError(f"Could not parse article title for {url}")
    return DownloadedArticle(
        title=parser.title,
        categories=parser.categories,
        steps=parser.steps,
        url=url,
        source_query=source_query,
    )


def build_search_url(base_url: str, query: str, page: int) -> str:
    """Construct one search URL for the given query and page."""
    if page <= 1:
        path = DEFAULT_SEARCH_PATH.format(query=quote_plus(query))
    else:
        start_offset = (page - 1) * DEFAULT_SEARCH_PAGE_SIZE
        path = (
            DEFAULT_SEARCH_PATH.format(query=quote_plus(query))
            + f"&start={start_offset}"
        )
    return urljoin(base_url, path)


def load_seen_urls(path: Path) -> Set[str]:
    """Read already-written JSONL entries to support resume mode."""
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            seen.add(json.loads(line)["url"])
        except Exception:
            continue
    return seen


def append_articles(path: Path, articles: Iterable[DownloadedArticle]) -> int:
    """Append articles to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for article in articles:
            handle.write(article.to_json())
            handle.write("\n")
            count += 1
    return count


def append_errors(path: Path, errors: Iterable[DownloadError]) -> int:
    """Append download errors to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for error in errors:
            handle.write(error.to_json())
            handle.write("\n")
            count += 1
    return count


def is_kitchen_article(article: DownloadedArticle) -> bool:
    """Heuristic filter for kitchen-domain articles."""
    category_text = " ".join(article.categories).lower()
    title_text = article.title.lower()
    step_text = " ".join(article.steps).lower()
    if any(keyword in category_text for keyword in NON_KITCHEN_CATEGORY_KEYWORDS):
        return False
    if any(keyword in title_text for keyword in NON_KITCHEN_TITLE_KEYWORDS):
        return False

    kitchen_signals = 0
    if any(keyword in category_text for keyword in KITCHEN_CATEGORY_KEYWORDS):
        kitchen_signals += 2
    if any(keyword in title_text for keyword in KITCHEN_TITLE_KEYWORDS):
        kitchen_signals += 1
    if any(keyword in step_text for keyword in KITCHEN_STEP_KEYWORDS):
        kitchen_signals += 1
    return kitchen_signals >= 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbs", nargs="+", default=list(DEFAULT_VERBS))
    parser.add_argument("--kitchen-verbs", action="store_true")
    parser.add_argument("--pages-per-verb", type=int, default=1)
    parser.add_argument("--max-articles-per-verb", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--user-agent", default="Mozilla/5.0 (compatible; thesis-bot/0.1)"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--kitchen-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.kitchen_verbs:
        args.verbs = list(KITCHEN_VERBS)
    seen_urls = load_seen_urls(args.output) if args.resume else set()
    written = 0
    discovered = 0
    error_count = 0
    error_output = args.error_output or args.output.with_name(
        args.output.stem + "_errors.jsonl"
    )

    print(
        f"Starting WikiHow download: verbs={list(args.verbs)} "
        f"pages_per_verb={args.pages_per_verb} "
        f"max_articles_per_verb={args.max_articles_per_verb} "
        f"output={args.output}"
    )
    if args.resume:
        print(
            f"Resume enabled: loaded {len(seen_urls)} existing URL(s) from {args.output}"
        )

    for verb in args.verbs:
        per_verb_count = 0
        print(f"[verb={verb}] starting")
        for page in range(1, args.pages_per_verb + 1):
            search_url = build_search_url(args.base_url, verb, page)
            print(
                f"[verb={verb}] fetching search page {page}/{args.pages_per_verb}: {search_url}"
            )
            try:
                search_html = fetch_html(
                    search_url, timeout=args.timeout, user_agent=args.user_agent
                )
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                append_errors(
                    error_output,
                    [
                        DownloadError(
                            stage="search",
                            url=search_url,
                            source_query=verb,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    ],
                )
                error_count += 1
                print(
                    f"[verb={verb}] search page failed and will be skipped: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            article_urls = extract_article_urls(
                search_html, base_url=args.base_url, query_term=verb
            )
            discovered += len(article_urls)
            print(
                f"[verb={verb}] page {page}: found {len(article_urls)} candidate URL(s), "
                f"total discovered={discovered}"
            )
            for article_url in article_urls:
                if article_url in seen_urls:
                    print(f"[verb={verb}] skip already seen: {article_url}")
                    continue
                print(
                    f"[verb={verb}] downloading article {per_verb_count + 1}/"
                    f"{args.max_articles_per_verb}: {article_url}"
                )
                try:
                    article_html = fetch_html(
                        article_url, timeout=args.timeout, user_agent=args.user_agent
                    )
                    article = parse_article_html(
                        article_html, article_url, source_query=verb
                    )
                except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                    append_errors(
                        error_output,
                        [
                            DownloadError(
                                stage="article",
                                url=article_url,
                                source_query=verb,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )
                        ],
                    )
                    error_count += 1
                    print(
                        f"[verb={verb}] article failed and will be skipped: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    time.sleep(args.sleep_seconds)
                    continue
                if args.kitchen_only and not is_kitchen_article(article):
                    print(
                        f"[verb={verb}] skip non-kitchen article: "
                        f"title={article.title!r} categories={article.categories[:3]!r}"
                    )
                    time.sleep(args.sleep_seconds)
                    continue
                append_articles(args.output, [article])
                seen_urls.add(article_url)
                written += 1
                per_verb_count += 1
                print(
                    f"[verb={verb}] wrote article {per_verb_count}: "
                    f"title={article.title!r} total_written={written}"
                )
                time.sleep(args.sleep_seconds)
                if per_verb_count >= args.max_articles_per_verb:
                    print(
                        f"[verb={verb}] reached max_articles_per_verb="
                        f"{args.max_articles_per_verb}"
                    )
                    break
            if per_verb_count >= args.max_articles_per_verb:
                break
            print(
                f"[verb={verb}] sleeping {args.sleep_seconds:.1f}s before next search page"
            )
            time.sleep(args.sleep_seconds)
        print(f"[verb={verb}] done, wrote {per_verb_count} article(s)")

    print(
        f"Wrote {written} article(s) to {args.output} "
        f"from {len(args.verbs)} query term(s), discovered {discovered} candidate URL(s), "
        f"errors={error_count}"
    )


if __name__ == "__main__":
    main()
