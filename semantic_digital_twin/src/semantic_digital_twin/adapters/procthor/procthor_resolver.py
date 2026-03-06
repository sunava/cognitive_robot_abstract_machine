from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Type

from typing_extensions import List

from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
)
from ...semantic_annotations.semantic_annotations import (
    Bottle,
    Apple,
    Plate,
    Bowl,
    Fork,
    Knife,
    Mug,
    Cup,
    Pan,
    PanLid,
    Pencil,
    Ball,
    Baseball,
    SprayBottle,
)

semantic_annotation_to_ycb_mesh_mapping = {
    Bottle: ["mustard_bottle.stl"],
    Apple: ["apple.stl"],
    Plate: ["plate.stl"],
    Bowl: ["bowl.stl"],
    Fork: ["fork.stl"],
    Knife: ["knife.stl"],
    Mug: ["mug.stl"],
    Cup: ["cup_a.stl"],
    Pan: ["skillet.stl"],
    PanLid: ["skillet_lid.stl"],
    Pencil: ["small_marker.stl", "large_marker.stl"],
    Ball: [
        "softball.stl",
        "baseball.stl",
        "tennisball.stl",
        "racequetball.stl",
        "golfball.stl",
        "mini_soccerball.stl",
    ],
    Baseball: ["baseball.stl"],
    SprayBottle: ["spraybottle.stl"],
}
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@dataclass
class ProcthorResolver:
    """Central resolver that deterministically maps a ProcTHOR name to exactly one class."""

    classes: List[Type[HasRootBody]] = field(default_factory=list)

    def resolve(self, name: str) -> Optional[Type[HasRootBody]]:
        """
        Resolve a given name to a class based on the number of matching tokens
        with the class name tokens or synonyms. The method preprocesses the
        name by removing numbers and splitting it into tokens, and then compares
        these tokens with the corresponding data in the available classes to
        find the best match.

        :param name: The name to resolve, represented as a string.
        :return: The class with the best match to the given name, or None if no matches are found.
        """
        # remove all numbers from the name
        name_tokens = set(n.lower() for n in re.sub(r"\d+", "", name).split("_"))
        possible_results = []
        for cls in self.classes:
            matches = cls.class_name_tokens().intersection(
                name_tokens
            ) or cls._synonyms.intersection(name_tokens)
            possible_results.append((cls, matches))

        if len(possible_results) == 0:
            return None
        # sort by max number of matches
        possible_results = sorted(
            possible_results, key=lambda x: len(x[1]), reverse=True
        )

        # if there are no matches, don't choose a class
        if len(possible_results[0][1]) == 0:
            return None
        best_cls, best_matches = possible_results[0]
        return best_cls
