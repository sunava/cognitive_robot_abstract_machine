"""Minimal OpenAI-compatible API connectivity test.

This is intentionally separate from the thesis pipeline so you can verify
account, quota, model access, and base URL behavior without any schema or
post-processing around it.
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly the word: ok",
            }
        ],
        "max_tokens": 8,
        "temperature": 0,
    }

    request = Request(
        url=f"{args.api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            print("HTTP 200")
            print(body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}")
        print(body)
        raise SystemExit(1)
    except URLError as exc:
        print(f"URL error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
