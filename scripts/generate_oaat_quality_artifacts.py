#!/usr/bin/env python3
"""Generate thesis artifacts for OAAT implementation quality.

The script extracts structural metrics from the current implementation and
writes LaTeX/Mermaid/DOT artifacts into doc/generated.
"""

from __future__ import annotations

import ast
import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_BASED = (
    REPO_ROOT / "coraplex/src/coraplex/robot_plans/actions/composite/tool_based.py"
)
ENUMS = REPO_ROOT / "coraplex/src/coraplex/datastructures/enums.py"
SEMANTIC_ANNOTATIONS = (
    REPO_ROOT
    / "semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/semantic_annotations.py"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "doc/generated"

ACTION_CLASSES = [
    "MixingAction",
    "WipingAction",
    "CuttingAction",
    "SimplePouringAction",
]
BASE_CLASS = "GeneralizedActionPlan"
POLICY_ENUMS = [
    "CuttingPartitionPolicy",
    "MixingDurationPolicy",
    "WipingCoveragePolicy",
    "PouringSidePolicy",
]
SEMANTIC_ACTION_ANNOTATIONS = [
    "MixingContainer",
    "MixingTool",
    "WipableSurface",
    "WipingTool",
    "PourableContainer",
    "ReceivingContainer",
    "Bread",
    "Knife",
    "CuttingBoard",
    "ToolAttachment",
]


@dataclass(frozen=True)
class CodeRange:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class ClassInfo:
    name: str
    start: int
    end: int
    bases: tuple[str, ...]
    methods: tuple[CodeRange, ...]

    def method(self, name: str) -> CodeRange | None:
        return next((method for method in self.methods if method.name == name), None)

    def methods_with_prefix(self, *prefixes: str) -> list[CodeRange]:
        return [
            method
            for method in self.methods
            if any(method.name.startswith(prefix) for prefix in prefixes)
        ]


def parse_classes(path: Path) -> dict[str, ClassInfo]:
    tree = ast.parse(path.read_text())
    classes: dict[str, ClassInfo] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(
                    CodeRange(
                        name=child.name,
                        start=child.lineno,
                        end=child.end_lineno or child.lineno,
                    )
                )
        classes[node.name] = ClassInfo(
            name=node.name,
            start=node.lineno,
            end=node.end_lineno or node.lineno,
            bases=tuple(base_name(base) for base in node.bases),
            methods=tuple(methods),
        )
    return classes


def base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return base_name(node.value)
    return ast.unparse(node)


def count_code_lines(path: Path, code_range: CodeRange | ClassInfo) -> int:
    lines = path.read_text().splitlines()
    count = 0
    for line in lines[code_range.start - 1 : code_range.end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def total_code_lines(path: Path, ranges: list[CodeRange | ClassInfo]) -> int:
    return sum(count_code_lines(path, code_range) for code_range in ranges)


def class_range(classes: dict[str, ClassInfo], name: str) -> ClassInfo:
    try:
        return classes[name]
    except KeyError as exc:
        raise RuntimeError(f"Could not find class {name}") from exc


def method_range(
    classes: dict[str, ClassInfo], class_name: str, method_name: str
) -> CodeRange:
    method = class_range(classes, class_name).method(method_name)
    if method is None:
        raise RuntimeError(f"Could not find {class_name}.{method_name}()")
    return method


def dataclass_ranges(classes: dict[str, ClassInfo]) -> list[ClassInfo]:
    suffixes = ("Scene", "Constraints", "Candidate")
    names = ["ArmToolCandidate"]
    return [
        info
        for info in classes.values()
        if info.name in names or info.name.endswith(suffixes)
    ]


def policy_methods(info: ClassInfo) -> list[CodeRange]:
    policy_names = (
        "resolve_",
        "select_",
        "_resolved_",
        "_requested_",
    )
    explicit = {
        "_cutting_partition_candidates",
        "_cutting_partition_constraints",
        "_cutting_partition_score",
        "_mixing_constraints",
        "_mixing_parameter_candidates",
        "_mixing_candidate_score",
        "_wiping_constraints",
        "_wiping_parameter_candidates",
        "_wiping_candidate_score",
        "_pouring_constraints",
        "_pouring_parameter_candidates",
        "_pouring_candidate_score",
        "pouring_arm_tool_candidates",
        "arm_tool_candidates",
    }
    return [
        method
        for method in info.methods
        if method.name in explicit
        or any(method.name.startswith(prefix) for prefix in policy_names)
    ]


def write_latex_table(output_dir: Path) -> Path:
    classes = parse_classes(TOOL_BASED)
    enums = parse_classes(ENUMS)
    semantic = parse_classes(SEMANTIC_ANNOTATIONS)

    base = class_range(classes, BASE_CLASS)
    action_ranges = [class_range(classes, name) for name in ACTION_CLASSES]
    sample_ranges = [
        method
        for name in ACTION_CLASSES
        if (method := class_range(classes, name).method("_sample_points")) is not None
    ]
    policy_ranges_by_action = {
        action.name: policy_methods(action) for action in action_ranges
    }
    enum_ranges = [class_range(enums, name) for name in POLICY_ENUMS if name in enums]
    semantic_ranges = [
        class_range(semantic, name)
        for name in SEMANTIC_ACTION_ANNOTATIONS
        if name in semantic
    ]

    execute_lines = count_code_lines(
        TOOL_BASED, method_range(classes, BASE_CLASS, "execute")
    )
    motion_lines = count_code_lines(
        TOOL_BASED, method_range(classes, BASE_CLASS, "_build_motion")
    )
    logging_lines = total_code_lines(
        TOOL_BASED,
        [
            method_range(classes, BASE_CLASS, "_update_waypoint_progress_for_logging"),
            method_range(classes, BASE_CLASS, "_current_tool_xyz"),
            method_range(classes, BASE_CLASS, "_waypoint_xyz"),
        ],
    )
    arm_tool_lines = total_code_lines(
        TOOL_BASED,
        [
            method_range(classes, BASE_CLASS, "arm_tool_candidates"),
            method_range(classes, BASE_CLASS, "select_arm_tool_candidate"),
            method_range(classes, BASE_CLASS, "resolve_arm_tool"),
        ],
    )
    sample_counts = [
        count_code_lines(TOOL_BASED, code_range) for code_range in sample_ranges
    ]
    policy_counts = [
        total_code_lines(TOOL_BASED, ranges)
        for ranges in policy_ranges_by_action.values()
        if ranges
    ]
    dataclass_lines = total_code_lines(TOOL_BASED, dataclass_ranges(classes))
    enum_lines = total_code_lines(ENUMS, enum_ranges)
    semantic_lines = total_code_lines(SEMANTIC_ANNOTATIONS, semantic_ranges)

    content = f"""% Generated by scripts/generate_oaat_quality_artifacts.py.
% Re-run the script after changing the OAAT implementation.
\\begin{{table}}[h]
\\centering
\\caption{{Implementation effort required to add or extend an \\ac{{OAAT}}
         action class, compared with infrastructure reused from the framework.
         Line counts are non-empty, non-comment lines measured automatically.}}
\\label{{tab:oaat-extension-cost-generated}}
\\begin{{tabular}}{{lcc}}
\\toprule
Component & New code & Reused / inherited \\\\
\\midrule
Execution lifecycle & 0 lines & {execute_lines} lines \\\\
Motion building & 0 lines & {motion_lines} lines \\\\
Waypoint-progress logging & 0 lines & {logging_lines} lines \\\\
Generic arm--tool resolution & 0 lines & {arm_tool_lines} lines \\\\
\\texttt{{\\_sample\\_points()}} implementation & {min(sample_counts)}--{max(sample_counts)} lines & --- \\\\
Scene / constraints / candidate dataclasses & {dataclass_lines} lines total & shared pattern \\\\
Policy enum blocks & {enum_lines} lines total & enum pattern \\\\
Policy / query / resolver block & {min(policy_counts)}--{max(policy_counts)} lines & shared structure \\\\
Semantic action annotations & {semantic_lines} lines total & \\ac{{SDT}} base classes \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    path = output_dir / "oaat_implementation_quality_table.tex"
    path.write_text(content)
    return path


def write_summary(output_dir: Path) -> Path:
    classes = parse_classes(TOOL_BASED)
    lines = [
        "% Generated by scripts/generate_oaat_quality_artifacts.py.",
        "% Use this as source material for the implementation quality section.",
        "\\begin{itemize}",
    ]
    base = class_range(classes, BASE_CLASS)
    lines.append(
        f"  \\item \\texttt{{{BASE_CLASS}}}: {count_code_lines(TOOL_BASED, base)} measured code lines."
    )
    for action_name in ACTION_CLASSES:
        info = class_range(classes, action_name)
        sample = info.method("_sample_points")
        policy_count = total_code_lines(TOOL_BASED, policy_methods(info))
        sample_text = (
            f"{count_code_lines(TOOL_BASED, sample)} lines"
            if sample is not None
            else "not applicable"
        )
        lines.append(
            f"  \\item \\texttt{{{action_name}}}: {count_code_lines(TOOL_BASED, info)} "
            f"measured code lines; sampling core: {sample_text}; "
            f"query/policy/resolver methods: {policy_count} lines."
        )
    lines.append("\\end{itemize}")
    path = output_dir / "oaat_implementation_quality_summary.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_mermaid(output_dir: Path) -> Path:
    classes = parse_classes(TOOL_BASED)
    lines = [
        "flowchart TB",
        '    base["GeneralizedActionPlan<br/>execute lifecycle<br/>motion building<br/>logging<br/>generic arm/tool query"]',
    ]
    for action_name in ACTION_CLASSES:
        info = class_range(classes, action_name)
        methods = {method.name for method in info.methods}
        parts = [action_name]
        if any(name.startswith("resolve_") for name in methods):
            parts.append("resolver boundary")
        if any(name.startswith("select_") for name in methods):
            parts.append("policy selector")
        if any("candidate" in name or "score" in name for name in methods):
            parts.append("candidate scoring")
        if "_sample_points" in methods:
            parts.append("object-local sampling")
        label = "<br/>".join(parts)
        node = action_name.replace("Action", "").lower()
        lines.append(f'    {node}["{label}"]')
        if "GeneralizedActionPlan" in info.bases:
            lines.append(f"    base --> {node}")
        else:
            lines.append(f"    {node} -. custom execute .-> base")
    lines.extend(
        [
            "    semantic[(SDT semantic annotations)]",
            "    policies[(typed policy enums)]",
            "    candidates[(frozen candidate/constraint dataclasses)]",
        ]
    )
    for action_name in ACTION_CLASSES:
        node = action_name.replace("Action", "").lower()
        lines.append(f"    semantic --> {node}")
        lines.append(f"    policies --> {node}")
        lines.append(f"    candidates --> {node}")
    path = output_dir / "oaat_query_policy_architecture.mmd"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_dot(output_dir: Path) -> Path:
    classes = parse_classes(TOOL_BASED)
    lines = [
        "digraph OAAT {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica", fontsize=16, labelloc="t", label="OAAT query and policy extension structure", bgcolor="white"];',
        '  node [shape=record, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#2f3a45", penwidth=1.6];',
        '  edge [fontname="Helvetica", fontsize=9, color="#5b6773", arrowsize=0.8, penwidth=1.4];',
        "",
        "  subgraph cluster_infrastructure {",
        '    label="shared execution infrastructure";',
        '    color="#8aa4c8";',
        "    penwidth=1.8;",
        '    style="rounded,filled";',
        '    fillcolor="#eef4fb";',
        '    base [fillcolor="#d8e8f8", label="{GeneralizedActionPlan|execute lifecycle|motion building|waypoint logging|generic arm/tool query}"];',
        "  }",
        "",
        "  subgraph cluster_knowledge {",
        '    label="query inputs and policy types";',
        '    color="#b9a26b";',
        "    penwidth=1.8;",
        '    style="rounded,filled";',
        '    fillcolor="#fff8e6";',
        '    semantic [shape=box, fillcolor="#f8e3a3", label="SDT semantic annotations"];',
        '    policies [shape=box, fillcolor="#f4d184", label="typed policy enums"];',
        '    candidates [shape=box, fillcolor="#fdebc3", label="frozen scene/constraint/candidate dataclasses"];',
        "  }",
        "",
        "  subgraph cluster_actions {",
        '    label="action-specific extension points";',
        '    color="#77a892";',
        "    penwidth=1.8;",
        '    style="rounded,filled";',
        '    fillcolor="#edf8f3";',
    ]
    for action_name in ACTION_CLASSES:
        info = class_range(classes, action_name)
        fields = [action_name]
        if any(method.name.startswith("resolve_") for method in info.methods):
            fields.append("resolver boundary")
        if any(method.name.startswith("select_") for method in info.methods):
            fields.append("policy selector")
        if any(
            "candidate" in method.name or "score" in method.name
            for method in info.methods
        ):
            fields.append("candidate scoring")
        if info.method("_sample_points") is not None:
            fields.append("object-local sampling")
        node = action_name.replace("Action", "").lower()
        fill = "#dff1e8" if "GeneralizedActionPlan" in info.bases else "#f2e2f7"
        label = "|".join(fields)
        lines.append(f'    {node} [fillcolor="{fill}", label="{{{label}}}"];')
    lines.append("  }")
    lines.append("")
    for action_name in ACTION_CLASSES:
        info = class_range(classes, action_name)
        node = action_name.replace("Action", "").lower()
        if "GeneralizedActionPlan" in info.bases:
            lines.append(f'  base -> {node} [label="inherits", color="#3778bf"];')
        else:
            lines.append(
                f'  {node} -> base [style=dashed, label="reuses concepts", color="#8a55a2"];'
            )
        lines.append(f'  semantic -> {node} [label="query", color="#a07822"];')
        lines.append(f'  policies -> {node} [label="policy", color="#b05f2c"];')
        lines.append(f'  candidates -> {node} [label="rank", color="#5b8f76"];')
    lines.append("}")
    path = output_dir / "oaat_query_policy_architecture.dot"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated thesis artifacts.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written = [
        write_latex_table(output_dir),
        write_summary(output_dir),
        write_mermaid(output_dir),
        write_dot(output_dir),
    ]
    for path in written:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
