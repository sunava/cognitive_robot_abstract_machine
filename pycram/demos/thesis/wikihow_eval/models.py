"""Datamodels shared by the WikiHow template-fit evaluation scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WikiHowArticle:
    """Raw article metadata used as pipeline input."""

    title: str
    categories: List[str]
    steps: List[str]
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionCase:
    """Structured action case extracted from a WikiHow-style article."""

    title: str
    verb: str
    action_word: str
    object_text: str
    tool_hint: Optional[str]
    domain_hint: Optional[str]
    categories: List[str]
    steps: List[str]
    url: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OntologyCase:
    """Action case enriched with lightweight ontology labels and affordances."""

    title: str
    verb: str
    template_candidates: List[str]
    object_text: str
    object_class: str
    tool_text: Optional[str]
    tool_class: str
    domain: str
    material_class: str
    functional_tags: List[str]
    categories: List[str]
    steps: List[str]
    url: Optional[str] = None
    mapping_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemplateFitResult:
    """Fit outcome for a single template evaluated against one ontology case."""

    article: str
    template: str
    fit: str
    score: float
    object_score: float
    tool_score: float
    domain_score: float
    function_score: float
    reason: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
