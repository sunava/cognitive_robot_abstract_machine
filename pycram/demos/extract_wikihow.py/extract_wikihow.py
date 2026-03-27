import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup

try:
    import spacy
except ImportError:
    spacy = None


TOOL_WORDS = {
    "knife",
    "chef's knife",
    "paring knife",
    "spoon",
    "fork",
    "whisk",
    "bowl",
    "pan",
    "pot",
    "plate",
    "sponge",
    "cloth",
    "rag",
    "brush",
    "peeler",
    "scissors",
    "grater",
    "cutting board",
    "board",
    "cup",
    "measuring cup",
    "ladle",
    "spatula",
    "tongs",
    "container",
    "bottle",
    "glass",
    "mug",
    "jar",
    "blender",
    "mixer",
    "colander",
    "strainer",
    "rolling pin",
    "tray",
    "skillet",
    "saucepan",
    "napkin",
    "paper towel",
}

VERB_MAP = {
    "slice": "cut",
    "chop": "cut",
    "dice": "cut",
    "mince": "cut",
    "carve": "cut",
    "saw": "cut",
    "stir": "mix",
    "whisk": "mix",
    "blend": "mix",
    "combine": "mix",
    "fold": "mix",
    "drizzle": "pour",
    "tilt": "pour",
    "empty": "pour",
    "scrub": "wipe",
    "clean": "wipe",
    "rub": "wipe",
    "dry": "wipe",
}

PRECONDITION_PATTERNS = [
    r"\bbefore\b",
    r"\bfirst\b",
    r"\bmake sure\b",
    r"\bensure\b",
    r"\bprepare\b",
    r"\bwash\b",
    r"\bpeel\b",
    r"\bset up\b",
    r"\bgather\b",
]

CONSTRAINT_PATTERNS = [
    r"\bavoid\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bbe careful\b",
    r"\bcareful not to\b",
    r"\bmake sure not to\b",
    r"\btry not to\b",
    r"\bprevent\b",
]

ORDERING_PATTERNS = [
    r"\bthen\b",
    r"\bnext\b",
    r"\bafter\b",
    r"\bfinally\b",
    r"\blower\b",
    r"\bstart by\b",
]

ACTION_HINTS = {
    "cut",
    "mix",
    "pour",
    "wipe",
    "slice",
    "chop",
    "dice",
    "mince",
    "stir",
    "whisk",
    "blend",
    "combine",
    "pour",
    "tilt",
    "drizzle",
    "wipe",
    "scrub",
    "clean",
    "rub",
}

FALLBACK_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "with",
    "for",
    "in",
    "on",
    "at",
    "by",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "up",
    "down",
    "your",
    "their",
    "its",
    "it",
    "them",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "be",
    "was",
    "were",
    "as",
    "if",
    "when",
    "while",
    "until",
    "than",
    "too",
    "very",
    "just",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
}


@dataclass
class StepExtraction:
    step_id: int
    text: str
    verbs: List[str] = field(default_factory=list)
    normalized_verbs: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    ordering_cues: List[str] = field(default_factory=list)


@dataclass
class ArticleExtraction:
    source: str
    title: Optional[str]
    raw_text: str
    steps: List[StepExtraction] = field(default_factory=list)
    verbs: List[str] = field(default_factory=list)
    normalized_verbs: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    likely_action_families: List[str] = field(default_factory=list)


class WikiHowExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception:
                self.nlp = None

    def extract_from_file(self, path: Path) -> ArticleExtraction:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if self._looks_like_html(text):
            title, cleaned_text, steps = self._parse_wikihow_html(text)
        else:
            title = path.stem
            cleaned_text, steps = self._parse_plain_text(text)
        return self._extract_article(
            source=str(path),
            title=title,
            raw_text=cleaned_text,
            step_texts=steps,
        )

    def extract_from_text(
        self, text: str, source: str = "<memory>"
    ) -> ArticleExtraction:
        if self._looks_like_html(text):
            title, cleaned_text, steps = self._parse_wikihow_html(text)
        else:
            title = None
            cleaned_text, steps = self._parse_plain_text(text)
        return self._extract_article(
            source=source,
            title=title,
            raw_text=cleaned_text,
            step_texts=steps,
        )

    def _extract_article(
        self, source: str, title: Optional[str], raw_text: str, step_texts: List[str]
    ) -> ArticleExtraction:
        step_results = []
        all_verbs: List[str] = []
        all_normalized_verbs: List[str] = []
        all_objects: List[str] = []
        all_tools: List[str] = []
        all_modifiers: List[str] = []
        all_preconditions: List[str] = []
        all_constraints: List[str] = []

        for i, step_text in enumerate(step_texts, start=1):
            step_result = self._extract_step(i, step_text)
            step_results.append(step_result)
            all_verbs.extend(step_result.verbs)
            all_normalized_verbs.extend(step_result.normalized_verbs)
            all_objects.extend(step_result.objects)
            all_tools.extend(step_result.tools)
            all_modifiers.extend(step_result.modifiers)
            all_preconditions.extend(step_result.preconditions)
            all_constraints.extend(step_result.constraints)

        likely_action_families = self._infer_action_families(
            all_normalized_verbs, all_tools, raw_text
        )

        return ArticleExtraction(
            source=source,
            title=title,
            raw_text=raw_text,
            steps=step_results,
            verbs=self._sorted_unique(all_verbs),
            normalized_verbs=self._sorted_unique(all_normalized_verbs),
            objects=self._sorted_unique(all_objects),
            tools=self._sorted_unique(all_tools),
            modifiers=self._sorted_unique(all_modifiers),
            preconditions=self._sorted_unique(all_preconditions),
            constraints=self._sorted_unique(all_constraints),
            likely_action_families=likely_action_families,
        )

    def _extract_step(self, step_id: int, text: str) -> StepExtraction:
        text = self._normalize_whitespace(text)
        verbs: List[str] = []
        normalized_verbs: List[str] = []
        objects: List[str] = []
        tools: List[str] = []
        modifiers: List[str] = []
        preconditions: List[str] = []
        constraints: List[str] = []
        ordering_cues: List[str] = []

        if self.nlp is not None:
            doc = self.nlp(text)
            verbs = self._extract_verbs_spacy(doc)
            normalized_verbs = [self._normalize_verb(v) for v in verbs]
            objects = self._extract_objects_spacy(doc)
            modifiers = self._extract_modifiers_spacy(doc)
            tools = self._extract_tools_spacy(doc, text)
        else:
            verbs = self._extract_verbs_fallback(text)
            normalized_verbs = [self._normalize_verb(v) for v in verbs]
            objects = self._extract_objects_fallback(text)
            modifiers = self._extract_modifiers_fallback(text)
            tools = self._extract_tools_fallback(text)

        preconditions = self._extract_pattern_hits(text, PRECONDITION_PATTERNS)
        constraints = self._extract_pattern_hits(text, CONSTRAINT_PATTERNS)
        ordering_cues = self._extract_pattern_hits(text, ORDERING_PATTERNS)

        return StepExtraction(
            step_id=step_id,
            text=text,
            verbs=self._sorted_unique(verbs),
            normalized_verbs=self._sorted_unique(normalized_verbs),
            objects=self._sorted_unique(objects),
            tools=self._sorted_unique(tools),
            modifiers=self._sorted_unique(modifiers),
            preconditions=self._sorted_unique(preconditions),
            constraints=self._sorted_unique(constraints),
            ordering_cues=self._sorted_unique(ordering_cues),
        )

    def _parse_wikihow_html(self, html: str) -> Tuple[Optional[str], str, List[str]]:
        soup = BeautifulSoup(html, "html.parser")

        title = None
        if soup.title and soup.title.text:
            title = self._normalize_whitespace(soup.title.text)

        for bad in soup(["script", "style", "noscript", "svg", "img", "footer", "nav"]):
            bad.decompose()

        step_texts = []

        step_candidates = soup.select(
            '[data-testid*="step"], .step, .steps li, ol li, ul li'
        )
        for node in step_candidates:
            txt = self._normalize_whitespace(node.get_text(" ", strip=True))
            if self._looks_like_real_step(txt):
                step_texts.append(txt)

        if not step_texts:
            text = self._normalize_whitespace(soup.get_text(" ", strip=True))
            _, step_texts = self._parse_plain_text(text)
            cleaned_text = text
        else:
            article_text = self._normalize_whitespace(soup.get_text(" ", strip=True))
            cleaned_text = article_text

        step_texts = self._deduplicate_preserve_order(step_texts)
        return title, cleaned_text, step_texts

    def _parse_plain_text(self, text: str) -> Tuple[str, List[str]]:
        text = self._normalize_whitespace(text)

        numbered_steps = re.split(
            r"(?:^|\s)(?:Step\s+\d+[:.]?|\d+[.)])\s+", text, flags=re.IGNORECASE
        )
        numbered_steps = [s.strip() for s in numbered_steps if s and s.strip()]

        if len(numbered_steps) >= 2:
            return text, numbered_steps

        sentence_candidates = re.split(r"(?<=[.!?])\s+", text)
        sentence_candidates = [
            self._normalize_whitespace(s) for s in sentence_candidates if s.strip()
        ]

        grouped_steps = []
        current = []

        for sent in sentence_candidates:
            current.append(sent)
            if len(current) >= 2 or any(
                re.search(p, sent, flags=re.IGNORECASE) for p in ORDERING_PATTERNS
            ):
                grouped_steps.append(" ".join(current))
                current = []

        if current:
            grouped_steps.append(" ".join(current))

        grouped_steps = [s for s in grouped_steps if self._looks_like_real_step(s)]
        if not grouped_steps:
            grouped_steps = sentence_candidates

        return text, grouped_steps

    def _extract_verbs_spacy(self, doc) -> List[str]:
        verbs = []
        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_.lower().strip()
                if lemma.isalpha():
                    verbs.append(lemma)
        return verbs

    def _extract_objects_spacy(self, doc) -> List[str]:
        objects = []
        for token in doc:
            if token.dep_ in {"dobj", "pobj", "obj", "attr"}:
                phrase = self._expand_np(token)
                if phrase:
                    objects.append(phrase.lower())
        return objects

    def _extract_modifiers_spacy(self, doc) -> List[str]:
        modifiers = []
        for token in doc:
            if token.pos_ == "ADV":
                word = token.text.lower().strip()
                if word.isalpha():
                    modifiers.append(word)
        return modifiers

    def _extract_tools_spacy(self, doc, text: str) -> List[str]:
        found = set()
        lowered_text = text.lower()
        for tool in TOOL_WORDS:
            if tool in lowered_text:
                found.add(tool)
        for token in doc:
            t = token.text.lower().strip()
            if t in TOOL_WORDS:
                found.add(t)
        return list(found)

    def _extract_verbs_fallback(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        return [t for t in tokens if t in ACTION_HINTS or t in VERB_MAP]

    def _extract_objects_fallback(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        objects = []
        for t in tokens:
            if t in FALLBACK_STOPWORDS:
                continue
            if t in TOOL_WORDS:
                continue
            if t in ACTION_HINTS:
                continue
            if len(t) <= 2:
                continue
            objects.append(t)
        return objects[:12]

    def _extract_modifiers_fallback(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        return [t for t in tokens if t.endswith("ly")]

    def _extract_tools_fallback(self, text: str) -> List[str]:
        lowered = text.lower()
        return [tool for tool in TOOL_WORDS if tool in lowered]

    def _extract_pattern_hits(self, text: str, patterns: List[str]) -> List[str]:
        hits = []
        lowered = text.lower()
        for pattern in patterns:
            m = re.search(pattern, lowered, flags=re.IGNORECASE)
            if m:
                hits.append(m.group(0).lower())
        return hits

    def _infer_action_families(
        self, normalized_verbs: List[str], tools: List[str], raw_text: str
    ) -> List[str]:
        score: Dict[str, int] = {
            "cutting": 0,
            "mixing": 0,
            "pouring": 0,
            "wiping": 0,
        }

        for v in normalized_verbs:
            if v == "cut":
                score["cutting"] += 2
            elif v == "mix":
                score["mixing"] += 2
            elif v == "pour":
                score["pouring"] += 2
            elif v == "wipe":
                score["wiping"] += 2

        tool_set = set(t.lower() for t in tools)
        if (
            "knife" in tool_set
            or "chef's knife" in tool_set
            or "paring knife" in tool_set
        ):
            score["cutting"] += 1
        if "whisk" in tool_set or "blender" in tool_set or "mixer" in tool_set:
            score["mixing"] += 1
        if (
            "cup" in tool_set
            or "measuring cup" in tool_set
            or "bottle" in tool_set
            or "ladle" in tool_set
        ):
            score["pouring"] += 1
        if (
            "sponge" in tool_set
            or "cloth" in tool_set
            or "rag" in tool_set
            or "brush" in tool_set
        ):
            score["wiping"] += 1

        lowered = raw_text.lower()
        if "pit" in lowered or "slice" in lowered or "chop" in lowered:
            score["cutting"] += 1
        if "stir" in lowered or "combine" in lowered:
            score["mixing"] += 1
        if "tilt" in lowered or "liquid" in lowered or "fill" in lowered:
            score["pouring"] += 1
        if "surface" in lowered or "clean" in lowered or "scrub" in lowered:
            score["wiping"] += 1

        ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
        return [name for name, s in ranked if s > 0]

    def _normalize_verb(self, verb: str) -> str:
        verb = verb.lower().strip()
        return VERB_MAP.get(verb, verb)

    def _expand_np(self, token) -> Optional[str]:
        if token is None:
            return None
        left = list(token.lefts)
        right = list(token.rights)
        span_tokens = []
        for t in left:
            if t.dep_ in {"det", "amod", "compound", "nummod"}:
                span_tokens.append(t)
        span_tokens.append(token)
        for t in right:
            if t.dep_ in {"compound", "amod"}:
                span_tokens.append(t)
        span_tokens = sorted(span_tokens, key=lambda x: x.i)
        text = " ".join(t.text for t in span_tokens).strip()
        text = self._normalize_whitespace(text.lower())
        if not text or text in FALLBACK_STOPWORDS:
            return None
        return text

    def _looks_like_html(self, text: str) -> bool:
        return (
            "<html" in text.lower() or "<body" in text.lower() or "<div" in text.lower()
        )

    def _looks_like_real_step(self, text: str) -> bool:
        if not text:
            return False
        if len(text.split()) < 3:
            return False
        if len(text) < 15:
            return False
        lowered = text.lower()
        if any(
            x in lowered
            for x in [
                "advertisement",
                "references",
                "community q&a",
                "things you'll need",
            ]
        ):
            return False
        return True

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _sorted_unique(self, items: List[str]) -> List[str]:
        return sorted(set(i.strip() for i in items if i and i.strip()))

    def _deduplicate_preserve_order(self, items: List[str]) -> List[str]:
        seen: Set[str] = set()
        out = []
        for item in items:
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out


def collect_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = []
    for ext in ("*.html", "*.htm", "*.txt"):
        files.extend(sorted(input_path.rglob(ext)))
    return files


def write_json(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to a .html/.txt file or a directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for JSON files"
    )
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name")
    parser.add_argument(
        "--merge", action="store_true", help="Also write one merged JSON file"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    extractor = WikiHowExtractor(spacy_model=args.model)
    input_files = collect_input_files(input_path)

    merged = []

    for file_path in input_files:
        article = extractor.extract_from_file(file_path)
        article_dict = asdict(article)
        merged.append(article_dict)

        out_name = file_path.stem + ".json"
        out_path = output_dir / out_name
        write_json(article_dict, out_path)
        print(f"Wrote {out_path}")

    if args.merge:
        merged_path = output_dir / "merged_extractions.json"
        write_json({"articles": merged}, merged_path)
        print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()
