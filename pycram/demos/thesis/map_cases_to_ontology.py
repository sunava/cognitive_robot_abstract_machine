"""Map structured action cases to lightweight ontology labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from wikihow_eval.models import ActionCase
from wikihow_eval.ontology import map_case_to_ontology


def load_cases(path: Path) -> List[ActionCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [ActionCase(**case) for case in raw_cases]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ontology_cases = [map_case_to_ontology(case) for case in load_cases(args.input)]
    args.output.write_text(
        json.dumps([case.to_dict() for case in ontology_cases], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(ontology_cases)} ontology-mapped cases to {args.output}")


if __name__ == "__main__":
    main()
