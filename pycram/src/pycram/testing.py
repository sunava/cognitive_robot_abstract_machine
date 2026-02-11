import logging
import os
import time
import unittest
from copy import deepcopy
from typing import Optional

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
from .datastructures.dataclasses import Context
from .plan import Plan

logger = logging.getLogger(__name__)


def _build_package_resolver():
    resolver = {}
    env_paths = os.environ.get("PYCRAM_PACKAGE_PATH", "")
    for path in env_paths.split(":"):
        if path:
            resolver[os.path.basename(path)] = path
    kitchen_path = os.environ.get("IAI_KITCHEN_PATH")
    if kitchen_path:
        resolver["iai_kitchen"] = kitchen_path
    default_kitchen_path = os.path.expanduser("~/workspace/ros/src/iai_maps/iai_kitchen")
    if os.path.exists(default_kitchen_path):
        resolver.setdefault("iai_kitchen", default_kitchen_path)
    apartment_path = os.environ.get("IAI_APARTMENT_PATH")
    if apartment_path:
        resolver["iai_apartment"] = apartment_path
    default_apartment_path = os.path.expanduser("~/workspace/ros/src/iai_maps/iai_apartment")
    if os.path.exists(default_apartment_path):
        resolver.setdefault("iai_apartment", default_apartment_path)
    default_pr2_path = os.path.expanduser("~/workspace/ros/src/iai_pr2/iai_pr2_description")
    if os.path.exists(default_pr2_path):
        resolver.setdefault("iai_pr2_description", default_pr2_path)
    return resolver or None


try:
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    logger.info(
        "Could not import VizMarkerPublisher. This is probably because you are not running ROS."
    )


def setup_world(urdf_path: Optional[str] = None) -> World:
    logger.setLevel(logging.DEBUG)

    resolver = _build_package_resolver()
    if urdf_path:
        URDFParser.from_file(file_path=urdf_path, package_resolver=resolver).parse()
    pr2_sem_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "robots",
            "pr2_calibrated_with_ft.urdf",
        ),
        package_resolver=resolver,
    ).parse()
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "worlds",
            "apartment.urdf",
        ),
        package_resolver=resolver,
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "milk.stl"
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    # apartment_world.merge_world(pr2_sem_world)
    apartment_world.merge_world(milk_world)
    apartment_world.merge_world(cereal_world)

    with apartment_world.modify_world():
        pr2_root = pr2_sem_world.get_body_by_name("base_footprint")
        apartment_root = apartment_world.root
        c_root_bf = OmniDrive.create_with_dofs(
            parent=apartment_root, child=pr2_root, world=apartment_world
        )
        apartment_world.merge_world(pr2_sem_world, c_root_bf)
        c_root_bf.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0)

    apartment_world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        )
    )
    apartment_world.get_body_by_name(
        "breakfast_cereal.stl"
    ).parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.37, 1.8, 1.05, reference_frame=apartment_world.root
    )
    milk_view = Milk(root=apartment_world.get_body_by_name("milk.stl"))
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    return apartment_world


class SemanticWorldTestCase(unittest.TestCase):
    world: World

    @classmethod
    def setUpClass(cls):
        resolver = _build_package_resolver()
        cls.pr2_sem_world = URDFParser(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "resources",
                "robots",
                "pr2_calibrated_with_ft.urdf",
            ),
            package_resolver=resolver,
        ).parse()
        cls.apartment_world = URDFParser(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "resources",
                "worlds",
                "apartment.urdf",
            ),
            package_resolver=resolver,
        ).parse()
        cls.apartment_world.merge_world(cls.pr2_sem_world)
