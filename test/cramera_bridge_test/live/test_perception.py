"""
Perception steps in the published and recorded plan: what a step was asked, which
objects it reported once it ended, and why it failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from typing_extensions import List, Optional

from coraplex.datastructures.enums import DetectionTechnique
from coraplex.exceptions import NothingDetected
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, MotionNode, PlanNode
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.motions.misc import DetectingMotion
from cramera_bridge.live.bundles import PlanTrees
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SeenQuery:
    """
    Stands in for a perception query: it answers the bodies the robot's camera sees.
    """

    seen: List[Body]
    """
    The bodies seen.
    """

    world: Optional[World] = None
    """
    The world asked about.
    """

    robot: Optional[object] = None
    """
    The robot that perceives, None for a step whose images are not taken.
    """

    def from_world(self) -> List[Body]:
        """
        :return: The bodies seen.
        """
        return self.seen


@dataclass
class DetectionPlan:
    """
    A plan of a root, a detect action and the motion answering its query.
    """

    plan: Plan
    """
    The plan.
    """

    action: ActionNode
    """
    The detect action.
    """

    motion: MotionNode
    """
    The motion answering the action's query.
    """


@pytest.fixture
def detection(kitchen) -> DetectionPlan:
    """
    :return: A plan detecting handles, whose query sees the milk.
    """
    plan = Plan(None)
    root = PlanNode()
    plan.add_node(root)
    action = ActionNode(
        designator=DetectAction(
            technique=DetectionTechnique.TYPES, object_sem_annotation=Handle
        )
    )
    root.add_child(action)
    motion = MotionNode(designator=DetectingMotion(query=SeenQuery([kitchen.milk])))
    action.add_child(motion)
    return DetectionPlan(plan=plan, action=action, motion=motion)


def nodes(publisher):
    """
    :return: The published plan's root, action and motion.
    """
    return publisher.current_plan().nodes


def test_a_perception_step_is_marked_and_names_the_class_it_was_asked_for(
    publisher, detection
):
    publisher.begin_plan(detection.plan)
    root, action, motion = nodes(publisher)
    assert action.parameters["object_sem_annotation"] == Handle.__name__
    assert (root.perceived, action.perceived, motion.perceived) == (None, None, [])


def test_a_perception_step_that_ended_reports_what_it_saw(
    publisher, detection, kitchen
):
    publisher.begin_plan(detection.plan)
    detection.motion.status = LifeCycleValues.SUCCEEDED
    publisher.observe_node_ended(detection.motion)
    *_, motion = nodes(publisher)
    assert motion.perceived == [str(kitchen.milk.id)]
    assert motion.reason is None


def test_a_failed_perception_step_reports_nothing_and_says_why(publisher, detection):
    publisher.begin_plan(detection.plan)
    detection.motion.status = LifeCycleValues.FAILED
    detection.motion.reason = NothingDetected(Handle)
    publisher.observe_node_ended(detection.motion)
    *_, motion = nodes(publisher)
    assert motion.perceived == []
    assert motion.reason == str(NothingDetected(Handle))


def test_the_recording_keeps_what_a_perception_step_saw(publisher, detection, kitchen):
    publisher.begin_plan(detection.plan)
    detection.motion.status = LifeCycleValues.SUCCEEDED
    publisher.observe_node_ended(detection.motion)
    (root,) = PlanTrees(plan=publisher.current_plan(), robot=None).roots()
    (action,) = root.children
    (motion,) = action.children
    assert motion.perceived == [str(kitchen.milk.id)]
    assert action.perceived is None
