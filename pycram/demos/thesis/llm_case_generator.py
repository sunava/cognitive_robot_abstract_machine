"""Generate structured action cases with an OpenAI-compatible LLM API.

The model is used as a candidate generator only. The downstream ontology mapping
and template-fit scorer remain the actual evaluation mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wikihow_eval.models import ActionCase


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "llm-stuff" / "llm_action_cases.json"
DEFAULT_VERBS = ("cut", "mix", "pour", "wipe")
DEFAULT_DOMAINS = (
    "food_preparation",
    "grooming",
    "gardening",
    "crafting",
    "construction",
    "cleaning",
)


CASE_SCHEMA: Dict[str, Any] = {
    "name": "action_case_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "verb": {"type": "string"},
                        "object_text": {"type": "string"},
                        "tool_hint": {"type": ["string", "null"]},
                        "domain_hint": {"type": ["string", "null"]},
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "expected_scope": {
                            "type": "string",
                            "enum": ["in_scope", "borderline", "out_of_scope"],
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "title",
                        "verb",
                        "object_text",
                        "tool_hint",
                        "domain_hint",
                        "categories",
                        "steps",
                        "expected_scope",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["cases"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    api_base: str = DEFAULT_API_BASE


def chunked(values: Sequence[str], chunk_size: int) -> List[List[str]]:
    """Split a sequence into fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [
        list(values[idx : idx + chunk_size])
        for idx in range(0, len(values), chunk_size)
    ]


def build_prompt(
    verbs: Sequence[str],
    domains: Sequence[str],
    cases_per_verb: int,
    template_names: Sequence[str],
) -> str:
    """Build a prompt for structured case generation."""
    return (
        "Generate structured manipulation action cases for robotic template-scope analysis. "
        "For each requested verb, produce diverse examples across the provided domains, "
        "including clear in-scope, borderline, and out-of-scope cases. "
        "Prefer concrete objects and tools. "
        f"Requested verbs: {', '.join(verbs)}. "
        f"Requested domains: {', '.join(domains)}. "
        f"Target templates under study: {', '.join(template_names)}. "
        f"Generate approximately {cases_per_verb} cases per verb. "
        "Return only cases that are action instructions and use concise titles like "
        "'How to Cut a Banana' or 'How to Cut Hair'. "
        "Use domain_hint values such as food_preparation, grooming, gardening, crafting, construction, cleaning. "
        "Steps should be short and realistic. "
        "expected_scope is the model's prior guess only; the downstream scorer will evaluate fit formally."
    )


def build_request_payload(prompt: str, model: str) -> Dict[str, Any]:
    """Construct an OpenAI-compatible chat completions request with JSON schema output."""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate structured robotic action cases. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": CASE_SCHEMA,
        },
        "temperature": 0.8,
    }


def _extract_message_text(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError("LLM response did not contain choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        if text_parts:
            return "".join(text_parts)
    raise ValueError("LLM response did not contain textual content")


def parse_cases_from_response(response_payload: Dict[str, Any]) -> List[ActionCase]:
    """Parse the structured JSON output into ActionCase objects."""
    content = _extract_message_text(response_payload)
    parsed = json.loads(content)
    raw_cases = parsed.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Structured output missing 'cases' array")
    cases: List[ActionCase] = []
    for raw_case in raw_cases:
        metadata = {
            "expected_scope": raw_case["expected_scope"],
            "rationale": raw_case["rationale"],
            "generator": "llm",
        }
        cases.append(
            ActionCase(
                title=raw_case["title"],
                verb=raw_case["verb"],
                action_word=raw_case["verb"],
                object_text=raw_case["object_text"],
                tool_hint=raw_case["tool_hint"],
                domain_hint=raw_case["domain_hint"],
                categories=list(raw_case["categories"]),
                steps=list(raw_case["steps"]),
                source="llm",
                metadata=metadata,
            )
        )
    return cases


def _http_error_message(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return body or str(exc)


def call_openai_compatible_api(
    config: LLMConfig,
    payload: Dict[str, Any],
    max_retries: int,
    retry_backoff_seconds: float,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible chat completions endpoint with retry on transient failures."""
    request_body = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        request = Request(
            url=f"{config.api_base.rstrip('/')}/chat/completions",
            data=request_body,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = _http_error_message(exc)
            if exc.code != 429 or attempt >= max_retries:
                raise RuntimeError(
                    f"LLM API request failed: HTTPError {exc.code}: {detail}"
                ) from exc
            sleep_seconds = retry_backoff_seconds * (2**attempt)
            print(
                f"LLM API rate-limited (attempt {attempt + 1}/{max_retries + 1}). "
                f"Retrying in {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)
        except URLError as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LLM API request failed: {type(exc).__name__}: {exc}"
                ) from exc
            sleep_seconds = retry_backoff_seconds * (2**attempt)
            print(
                f"LLM API network error (attempt {attempt + 1}/{max_retries + 1}). "
                f"Retrying in {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)
    raise RuntimeError("LLM API request failed after retries")


def save_cases(path: Path, cases: Iterable[ActionCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([case.to_dict() for case in cases], indent=2), encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbs", nargs="+", default=list(DEFAULT_VERBS))
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--cases-per-verb", type=int, default=12)
    parser.add_argument(
        "--templates", nargs="+", default=["cutting", "mixing", "pouring", "wiping"]
    )
    parser.add_argument(
        "--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-output", type=Path)
    parser.add_argument("--response-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dry_run:
        prompt = build_prompt(
            verbs=args.verbs,
            domains=args.domains,
            cases_per_verb=args.cases_per_verb,
            template_names=args.templates,
        )
        print(prompt)
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in environment variable {args.api_key_env}"
        )

    config = LLMConfig(api_key=api_key, model=args.model, api_base=args.api_base)
    prompts: List[str] = []
    responses: List[Dict[str, Any]] = []
    cases: List[ActionCase] = []
    verb_batches = chunked(args.verbs, args.batch_size)
    for batch_index, verb_batch in enumerate(verb_batches, start=1):
        prompt = build_prompt(
            verbs=verb_batch,
            domains=args.domains,
            cases_per_verb=args.cases_per_verb,
            template_names=args.templates,
        )
        prompts.append(prompt)
        print(
            f"Generating batch {batch_index}/{len(verb_batches)} for verbs={verb_batch}"
        )
        payload = build_request_payload(prompt, model=args.model)
        response_payload = call_openai_compatible_api(
            config,
            payload,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        responses.append(response_payload)
        cases.extend(parse_cases_from_response(response_payload))

    if args.prompt_output:
        args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_output.write_text("\n\n---\n\n".join(prompts), encoding="utf-8")
    if args.response_output:
        args.response_output.parent.mkdir(parents=True, exist_ok=True)
        args.response_output.write_text(
            json.dumps(responses, indent=2), encoding="utf-8"
        )

    save_cases(args.output, cases)
    print(f"Wrote {len(cases)} LLM-generated action cases to {args.output}")


if __name__ == "__main__":
    main()
