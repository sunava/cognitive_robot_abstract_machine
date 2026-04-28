from dataclasses import dataclass

from krrood.entity_query_language.predicate import Predicate, symbolic_function
from typing_extensions import Optional, Tuple, Type, Union

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Colander,
    Core,
    Food,
    Knife,
    Peel,
    Peeler,
    Potato,
    Produce,
    Shell,
    Stem,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


FoodOrType = Union[Food, Type[Food]]
AnnotationType = Type[SemanticAnnotation]


@dataclass(frozen=True)
class PouringKnowledge:
    technique: str
    food_type: Type[Food]
    tool_type: AnnotationType
    min_angle: float = 0.0
    max_angle: float = 90.0
    min_duration: float = 0.0
    max_duration: float = 10.0


@dataclass(frozen=True)
class CuttingKnowledge:
    technique: str
    food_type: Type[Food]
    tool_type: AnnotationType = Knife
    position: str = "middle"
    repetition: str = "1"
    prior_task: Optional[str] = None
    removable_parts: Tuple[AnnotationType, ...] = ()
    peeling_tool_type: Optional[AnnotationType] = None


POURING_KNOWLEDGE = (
    PouringKnowledge(technique="Draining", food_type=Food, tool_type=Colander),
)


CUTTING_KNOWLEDGE = (
    CuttingKnowledge(technique="Cutting Action", food_type=Food),
    CuttingKnowledge(technique="Slicing", food_type=Food),
    CuttingKnowledge(technique="Dicing", food_type=Food),
    CuttingKnowledge(technique="Halving", food_type=Food, repetition="exactly 1"),
    CuttingKnowledge(technique="Quartering", food_type=Produce, repetition="exactly 1"),
    CuttingKnowledge(
        technique="Peeling",
        food_type=Potato,
        removable_parts=(Peel,),
        peeling_tool_type=Peeler,
    ),
    CuttingKnowledge(
        technique="Cutting Action",
        food_type=Potato,
        removable_parts=(Peel,),
        peeling_tool_type=Peeler,
    ),
    CuttingKnowledge(
        technique="Cutting Action",
        food_type=Produce,
        removable_parts=(Stem, Core),
    ),
    CuttingKnowledge(
        technique="Shelling",
        food_type=Food,
        removable_parts=(Shell,),
    ),
)


def _normalize_technique(technique: str) -> str:
    return technique.strip().casefold()


def _matches_food(food: FoodOrType, food_type: Type[Food]) -> bool:
    if isinstance(food, type):
        return issubclass(food, food_type)
    return isinstance(food, food_type)


def _best_match(food: FoodOrType, technique: str, candidates):
    normalized_technique = _normalize_technique(technique)
    matches = [
        candidate
        for candidate in candidates
        if normalized_technique == _normalize_technique(candidate.technique)
        and _matches_food(food, candidate.food_type)
    ]
    if not matches:
        return None
    return max(matches, key=lambda candidate: len(candidate.food_type.mro()))


def _cutting_knowledge(food: FoodOrType, technique: str) -> Optional[CuttingKnowledge]:
    return _best_match(food, technique, CUTTING_KNOWLEDGE)


def _pouring_knowledge(food: FoodOrType, technique: str) -> Optional[PouringKnowledge]:
    return _best_match(food, technique, POURING_KNOWLEDGE)


def _best_cutting_match_with_condition(
    food: FoodOrType, condition
) -> Optional[CuttingKnowledge]:
    matches = [
        candidate
        for candidate in CUTTING_KNOWLEDGE
        if _matches_food(food, candidate.food_type) and condition(candidate)
    ]
    if not matches:
        return None
    return max(matches, key=lambda candidate: len(candidate.food_type.mro()))


@symbolic_function
def get_cutting_tool(
    food: FoodOrType, technique: str = "Cutting Action"
) -> Optional[AnnotationType]:
    knowledge = _cutting_knowledge(food, technique)
    return knowledge.tool_type if knowledge else None


@symbolic_function
def get_prior_cutting_task(food: FoodOrType, technique: str) -> Optional[str]:
    knowledge = _cutting_knowledge(food, technique)
    return knowledge.prior_task if knowledge else None


@symbolic_function
def get_cutting_position(food: FoodOrType, technique: str) -> Optional[str]:
    knowledge = _cutting_knowledge(food, technique)
    return knowledge.position if knowledge else None


@symbolic_function
def get_cutting_repetition(food: FoodOrType, technique: str) -> Optional[str]:
    knowledge = _cutting_knowledge(food, technique)
    return knowledge.repetition if knowledge else None


@symbolic_function
def get_peeling_tool(
    food: FoodOrType, technique: Optional[str] = None
) -> Optional[AnnotationType]:
    knowledge = (
        _cutting_knowledge(food, technique)
        if technique is not None
        else _best_cutting_match_with_condition(
            food, lambda candidate: candidate.peeling_tool_type is not None
        )
    )
    return knowledge.peeling_tool_type if knowledge else None


@symbolic_function
def get_pouring_tool(food: FoodOrType, technique: str) -> Optional[AnnotationType]:
    knowledge = _pouring_knowledge(food, technique)
    return knowledge.tool_type if knowledge else None


@symbolic_function
def get_min_pouring_angle(food: FoodOrType, technique: str) -> Optional[float]:
    knowledge = _pouring_knowledge(food, technique)
    return knowledge.min_angle if knowledge else None


@symbolic_function
def get_max_pouring_angle(food: FoodOrType, technique: str) -> Optional[float]:
    knowledge = _pouring_knowledge(food, technique)
    return knowledge.max_angle if knowledge else None


@symbolic_function
def get_min_pouring_duration(food: FoodOrType, technique: str) -> Optional[float]:
    knowledge = _pouring_knowledge(food, technique)
    return knowledge.min_duration if knowledge else None


@symbolic_function
def get_max_pouring_duration(food: FoodOrType, technique: str) -> Optional[float]:
    knowledge = _pouring_knowledge(food, technique)
    return knowledge.max_duration if knowledge else None


@dataclass
class UsesCuttingTool(Predicate):
    food: FoodOrType
    tool_type: AnnotationType
    technique: str = "Cutting Action"

    def __call__(self) -> bool:
        return get_cutting_tool(self.food, self.technique) == self.tool_type


@dataclass
class RequiresPriorCuttingTask(Predicate):
    food: FoodOrType
    technique: str
    prior_task: str

    def __call__(self) -> bool:
        return get_prior_cutting_task(self.food, self.technique) == self.prior_task


@dataclass
class HasCuttingPosition(Predicate):
    food: FoodOrType
    technique: str
    position: str

    def __call__(self) -> bool:
        return get_cutting_position(self.food, self.technique) == self.position


@dataclass
class HasCuttingRepetition(Predicate):
    food: FoodOrType
    technique: str
    repetition: str

    def __call__(self) -> bool:
        return get_cutting_repetition(self.food, self.technique) == self.repetition


@dataclass
class RequiresRemovingPart(Predicate):
    food: FoodOrType
    part_type: AnnotationType
    technique: Optional[str] = None

    def __call__(self) -> bool:
        knowledge = (
            _cutting_knowledge(self.food, self.technique)
            if self.technique is not None
            else _best_cutting_match_with_condition(
                self.food,
                lambda candidate: self.part_type in candidate.removable_parts,
            )
        )
        return knowledge is not None and self.part_type in knowledge.removable_parts


@dataclass
class UsesPeelingTool(Predicate):
    food: FoodOrType
    tool_type: AnnotationType
    technique: Optional[str] = None

    def __call__(self) -> bool:
        return get_peeling_tool(self.food, self.technique) == self.tool_type


@dataclass
class UsesPouringTool(Predicate):
    food: FoodOrType
    technique: str
    tool_type: AnnotationType

    def __call__(self) -> bool:
        return get_pouring_tool(self.food, self.technique) == self.tool_type


@dataclass
class PouringAngleInRange(Predicate):
    food: FoodOrType
    technique: str
    angle: float

    def __call__(self) -> bool:
        minimum = get_min_pouring_angle(self.food, self.technique)
        maximum = get_max_pouring_angle(self.food, self.technique)
        return minimum is not None and maximum is not None and minimum <= self.angle <= maximum


@dataclass
class PouringDurationInRange(Predicate):
    food: FoodOrType
    technique: str
    duration: float

    def __call__(self) -> bool:
        minimum = get_min_pouring_duration(self.food, self.technique)
        maximum = get_max_pouring_duration(self.food, self.technique)
        return minimum is not None and maximum is not None and minimum <= self.duration <= maximum
