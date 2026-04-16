"""Lightweight ontology mapping for offline template-scope evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import ActionCase, OntologyCase


OBJECT_CLASS_RULES: Sequence[Tuple[Set[str], str, str, Set[str]]] = (
    (
        {"carrot", "mango", "bread", "cucumber", "potato", "apple"},
        "FoodItem",
        "FoodMaterial",
        {"separable", "cuttable", "mixable_if_processed"},
    ),
    (
        {"batter", "salad", "dough"},
        "FoodMixture",
        "FoodMaterial",
        {"mixable", "container_compatible"},
    ),
    (
        {"cement", "concrete", "paint"},
        "ConstructionMaterial",
        "ConstructionMaterial",
        {"mixable"},
    ),
    (
        {"water", "juice", "milk", "soup"},
        "PourableLiquid",
        "Liquid",
        {"pourable", "mixable"},
    ),
    (
        {"counter", "counters", "countertop", "window", "floor", "table"},
        "Surface",
        "RigidSurface",
        {"wipe_target", "surface_contact"},
    ),
    (
        {"hair", "beard", "nails"},
        "BodyPart",
        "LivingTissue",
        {"animate_part", "cuttable"},
    ),
    (
        {"hedge", "grass", "bush"},
        "PlantPart",
        "PlantMaterial",
        {"animate_part", "cuttable"},
    ),
)

TOOL_CLASS_RULES: Sequence[Tuple[Set[str], str, Set[str]]] = (
    (
        {"knife", "chef's knife", "paring knife", "bread knife", "scissors"},
        "CuttingTool",
        {"sharp_edge"},
    ),
    ({"spoon", "whisk", "spatula"}, "MixingTool", {"agitates"}),
    (
        {"cup", "jug", "pitcher", "bottle", "measuring cup"},
        "PouringTool",
        {"dispenses"},
    ),
    (
        {"cloth", "sponge", "rag", "paper towel", "mop"},
        "CleaningTool",
        {"absorbs", "surface_contact"},
    ),
)

DOMAIN_RULES: Sequence[Tuple[Set[str], str]] = (
    (
        {"food and entertaining", "food", "cooking", "baking", "vegetables"},
        "food_preparation",
    ),
    ({"cleaning", "housekeeping", "home and garden"}, "cleaning"),
    ({"personal care and style", "hair care", "beauty"}, "grooming"),
    ({"crafts", "office"}, "crafting"),
    ({"home improvement", "construction"}, "construction"),
)

VERB_TO_TEMPLATE = {
    "cut": ["cutting"],
    "slice": ["cutting"],
    "chop": ["cutting"],
    "mix": ["mixing"],
    "stir": ["mixing"],
    "whisk": ["mixing"],
    "pour": ["pouring"],
    "drain": ["pouring"],
    "wipe": ["wiping"],
    "scrub": ["wiping"],
}


def _normalize(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _first_match_label(
    tokens: Iterable[str], rules: Sequence[Tuple[Set[str], str]]
) -> Optional[str]:
    token_set = {_normalize(token) for token in tokens if token}
    for candidates, label in rules:
        if token_set & candidates:
            return label
    return None


def infer_domain(categories: Sequence[str], title: str) -> Tuple[str, List[str]]:
    """Infer a coarse domain from categories, with a title-based fallback."""
    notes: List[str] = []
    category_tokens = {_normalize(category) for category in categories}
    for candidates, label in DOMAIN_RULES:
        if category_tokens & candidates:
            notes.append(f"domain={label}:category_match")
            return label, notes
    title_lower = _normalize(title)
    if "hair" in title_lower:
        notes.append("domain=grooming:title_fallback")
        return "grooming", notes
    if "counter" in title_lower or "window" in title_lower:
        notes.append("domain=cleaning:title_fallback")
        return "cleaning", notes
    notes.append("domain=generic:default")
    return "generic", notes


def classify_object(object_text: str) -> Tuple[str, str, Set[str], List[str]]:
    """Map an extracted object phrase to a coarse ontology class."""
    normalized = _normalize(object_text)
    notes: List[str] = []
    for aliases, object_class, material_class, tags in OBJECT_CLASS_RULES:
        if normalized in aliases:
            notes.append(f"object_class={object_class}:lexicon_match")
            return object_class, material_class, set(tags), notes
    if normalized.endswith("s"):
        singular = normalized[:-1]
        for aliases, object_class, material_class, tags in OBJECT_CLASS_RULES:
            if singular in aliases:
                notes.append(f"object_class={object_class}:singularized")
                return object_class, material_class, set(tags), notes
    notes.append("object_class=UnknownObject:default")
    return "UnknownObject", "UnknownMaterial", set(), notes


def classify_tool(
    tool_text: Optional[str], steps: Sequence[str]
) -> Tuple[str, List[str]]:
    """Infer tool class from explicit tool hints or step text."""
    notes: List[str] = []
    candidates: List[str] = []
    if tool_text:
        candidates.append(tool_text)
    candidates.extend(steps)
    normalized_candidates = [_normalize(candidate) for candidate in candidates]
    for aliases, tool_class, _tags in TOOL_CLASS_RULES:
        for candidate in normalized_candidates:
            if any(alias in candidate for alias in aliases):
                notes.append(f"tool_class={tool_class}:lexicon_match")
                return tool_class, notes
    notes.append("tool_class=UnknownTool:default")
    return "UnknownTool", notes


def map_case_to_ontology(action_case: ActionCase) -> OntologyCase:
    """Enrich a structured action case with coarse ontology labels."""
    object_class, material_class, functional_tags, object_notes = classify_object(
        action_case.object_text
    )
    tool_class, tool_notes = classify_tool(action_case.tool_hint, action_case.steps)
    domain, domain_notes = infer_domain(action_case.categories, action_case.title)
    template_candidates = VERB_TO_TEMPLATE.get(action_case.verb, [])
    functional_tags = set(functional_tags)
    if tool_class == "CuttingTool":
        functional_tags.add("requires_sharp_tool")
    if tool_class == "MixingTool":
        functional_tags.add("requires_agitation_tool")
    if tool_class == "CleaningTool":
        functional_tags.add("requires_wipe_tool")
    if tool_class == "PouringTool":
        functional_tags.add("supports_flow")
    return OntologyCase(
        title=action_case.title,
        verb=action_case.verb,
        template_candidates=template_candidates,
        object_text=action_case.object_text,
        object_class=object_class,
        tool_text=action_case.tool_hint,
        tool_class=tool_class,
        domain=domain,
        material_class=material_class,
        functional_tags=sorted(functional_tags),
        categories=list(action_case.categories),
        steps=list(action_case.steps),
        url=action_case.url,
        mapping_notes=object_notes + tool_notes + domain_notes,
    )
