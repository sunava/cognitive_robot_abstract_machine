"""Template-fit scoring logic for WikiHow-derived action cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from .models import OntologyCase, TemplateFitResult


FULL_FIT_THRESHOLD = 0.8
PARTIAL_FIT_THRESHOLD = 0.4

WEIGHTS: Dict[str, float] = {
    "object": 0.35,
    "tool": 0.25,
    "domain": 0.20,
    "function": 0.20,
}


@dataclass(frozen=True)
class TemplateProfile:
    """Semantic scope description for one reusable action template."""

    name: str
    implementation_name: str
    object_classes: Set[str]
    partial_object_classes: Set[str]
    forbidden_object_classes: Set[str]
    tool_classes: Set[str]
    domains: Set[str]
    partial_domains: Set[str]
    required_tags: Set[str]


TEMPLATE_PROFILES: Dict[str, TemplateProfile] = {
    "cutting": TemplateProfile(
        name="cutting",
        implementation_name="cut",
        object_classes={"FoodItem"},
        partial_object_classes=set(),
        forbidden_object_classes={"BodyPart", "PlantPart"},
        tool_classes={"CuttingTool"},
        domains={"food_preparation"},
        partial_domains=set(),
        required_tags={"cuttable"},
    ),
    "mixing": TemplateProfile(
        name="mixing",
        implementation_name="volume_agitation",
        object_classes={"FoodMixture", "PourableLiquid"},
        partial_object_classes={"ConstructionMaterial"},
        forbidden_object_classes=set(),
        tool_classes={"MixingTool"},
        domains={"food_preparation"},
        partial_domains={"construction"},
        required_tags={"mixable"},
    ),
    "pouring": TemplateProfile(
        name="pouring",
        implementation_name="pour",
        object_classes={"PourableLiquid"},
        partial_object_classes={"ConstructionMaterial"},
        forbidden_object_classes=set(),
        tool_classes={"PouringTool"},
        domains={"food_preparation"},
        partial_domains={"construction"},
        required_tags={"pourable"},
    ),
    "wiping": TemplateProfile(
        name="wiping",
        implementation_name="surface_wipe_raster_scrub",
        object_classes={"Surface"},
        partial_object_classes=set(),
        forbidden_object_classes=set(),
        tool_classes={"CleaningTool"},
        domains={"cleaning"},
        partial_domains=set(),
        required_tags={"wipe_target"},
    ),
}


def _score_membership(
    value: str, allowed: Set[str], partial: Set[str], forbidden: Set[str]
) -> float:
    if value in forbidden:
        return 0.0
    if value in allowed:
        return 1.0
    if value in partial:
        return 0.5
    if value == "UnknownObject" or value == "UnknownTool":
        return 0.25
    return 0.0


def _score_domain(domain: str, allowed: Set[str], partial: Set[str]) -> float:
    if domain in allowed:
        return 1.0
    if domain in partial:
        return 0.5
    if domain == "generic":
        return 0.25
    return 0.0


def _score_required_tags(case_tags: Sequence[str], required_tags: Set[str]) -> float:
    if not required_tags:
        return 1.0
    case_set = set(case_tags)
    overlap = len(case_set & required_tags)
    return overlap / len(required_tags)


def _fit_label(score: float) -> str:
    if score >= FULL_FIT_THRESHOLD:
        return "full_fit"
    if score >= PARTIAL_FIT_THRESHOLD:
        return "partial_fit"
    return "out_of_scope"


def score_case_against_template(
    case: OntologyCase, template: TemplateProfile
) -> TemplateFitResult:
    """Score one ontology case against one template profile."""
    object_score = _score_membership(
        case.object_class,
        template.object_classes,
        template.partial_object_classes,
        template.forbidden_object_classes,
    )
    tool_score = _score_membership(case.tool_class, template.tool_classes, set(), set())
    domain_score = _score_domain(
        case.domain, template.domains, template.partial_domains
    )
    function_score = _score_required_tags(case.functional_tags, template.required_tags)
    score = round(
        WEIGHTS["object"] * object_score
        + WEIGHTS["tool"] * tool_score
        + WEIGHTS["domain"] * domain_score
        + WEIGHTS["function"] * function_score,
        3,
    )
    forced_out_of_scope = case.object_class in template.forbidden_object_classes
    if forced_out_of_scope:
        score = min(score, PARTIAL_FIT_THRESHOLD - 0.001)
    reasons: List[str] = [
        f"object_class={case.object_class}",
        f"tool_class={case.tool_class}",
        f"domain={case.domain}",
        f"required_tags={sorted(template.required_tags)}",
    ]
    if case.object_class in template.forbidden_object_classes:
        reasons.append(f"forbidden_object_class={case.object_class}")
    if object_score < 1.0:
        reasons.append(f"object_score={object_score:.2f}")
    if tool_score < 1.0:
        reasons.append(f"tool_score={tool_score:.2f}")
    if domain_score < 1.0:
        reasons.append(f"domain_score={domain_score:.2f}")
    if function_score < 1.0:
        reasons.append(f"function_score={function_score:.2f}")
    return TemplateFitResult(
        article=case.title,
        template=template.name,
        fit="out_of_scope" if forced_out_of_scope else _fit_label(score),
        score=score,
        object_score=object_score,
        tool_score=tool_score,
        domain_score=domain_score,
        function_score=function_score,
        reason=reasons,
    )


def score_case(case: OntologyCase) -> List[TemplateFitResult]:
    """Score a case against the template candidates implied by its verb."""
    template_names = case.template_candidates or sorted(TEMPLATE_PROFILES)
    return [
        score_case_against_template(case, TEMPLATE_PROFILES[name])
        for name in template_names
        if name in TEMPLATE_PROFILES
    ]


def summarize_results(
    results: Iterable[TemplateFitResult],
) -> Dict[str, Dict[str, int]]:
    """Aggregate fit labels per template for quick coverage inspection."""
    summary: Dict[str, Dict[str, int]] = {}
    for result in results:
        template_counts = summary.setdefault(
            result.template, {"full_fit": 0, "partial_fit": 0, "out_of_scope": 0}
        )
        template_counts[result.fit] += 1
    return summary
