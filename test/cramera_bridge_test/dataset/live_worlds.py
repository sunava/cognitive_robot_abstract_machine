"""
Plans, statecharts and marker messages for the live publisher tests, built from native
CRAM classes and plain message stand-ins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, MotionNode, PlanNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from coraplex.robot_plans.motions.container import OpeningMotion
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.monitors.feature_monitors import HeightMonitor
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% plans


@dataclass
class SmallPlan:
    """
    A plan of a root, an action moving the torso and a motion opening a body with the
    left arm.
    """

    plan: Plan
    """
    The plan.
    """

    root: PlanNode
    """
    Its root.
    """

    action: ActionNode
    """
    Its action.
    """

    motion: MotionNode
    """
    The action's motion.
    """

    @classmethod
    def build(
        cls, opened: Body, arm: Arm, context: Optional[Context] = None
    ) -> SmallPlan:
        """
        :param opened: The body the motion opens.
        :param arm: The arm the motion opens it with.
        :param context: The context naming the robot that performs the plan, or None
            for a plan without one.
        :return: A new plan; no node has run.
        """
        plan = Plan(context)
        root = PlanNode()
        plan.add_node(root)
        action = ActionNode(designator=MoveTorsoAction(TorsoState.HIGH))
        root.add_child(action)
        motion = MotionNode(designator=OpeningMotion(object_part=opened, arm=arm))
        action.add_child(motion)
        return cls(plan=plan, root=root, action=action, motion=motion)


# %% statecharts


@dataclass
class HeightChart:
    """
    A monitor-only statechart observing a body's height, ticked by giskardpy.
    """

    chart: MotionStatechart
    """
    The chart.
    """

    executor: Executor
    """
    Its executor.
    """

    @classmethod
    def build(cls, world: World, body: Body) -> HeightChart:
        """
        :param world: The world.
        :param body: The observed body.
        :return: The chart, compiled.
        """
        chart = MotionStatechart()
        chart.add_node(
            HeightMonitor(
                root_link=world.root,
                tip_link=body,
                reference_point=Point3(0, 0, 0, reference_frame=world.root),
                tip_point=Point3(0, 0, 0, reference_frame=body),
                lower_limit=0.5,
                upper_limit=2.0,
            )
        )
        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=chart)
        return cls(chart=chart, executor=executor)


# %% marker messages


@dataclass
class Vector:
    """
    Stands in for a message with ``x``, ``y`` and ``z``.
    """

    x: float = 0.0
    """
    The x component.
    """

    y: float = 0.0
    """
    The y component.
    """

    z: float = 0.0
    """
    The z component.
    """


@dataclass
class Orientation:
    """
    Stands in for a quaternion message.
    """

    x: float = 0.0
    """
    The x component.
    """

    y: float = 0.0
    """
    The y component.
    """

    z: float = 0.0
    """
    The z component.
    """

    w: float = 1.0
    """
    The scalar component.
    """


@dataclass
class MarkerPose:
    """
    Stands in for a pose message.
    """

    position: Vector = field(default_factory=Vector)
    """
    The position.
    """

    orientation: Orientation = field(default_factory=Orientation)
    """
    The orientation.
    """


@dataclass
class MarkerColor:
    """
    Stands in for a colour message.
    """

    r: float = 1.0
    """
    The red channel.
    """

    g: float = 0.0
    """
    The green channel.
    """

    b: float = 0.0
    """
    The blue channel.
    """

    a: float = 1.0
    """
    The opacity.
    """


@dataclass
class Header:
    """
    Stands in for a message header.
    """

    frame_id: str = "map"
    """
    The frame the message is given in.
    """


@dataclass
class MarkerMessage:
    """
    Stands in for a ``visualization_msgs/Marker``.
    """

    ns: str = "points"
    """
    The namespace.
    """

    id: int = 0
    """
    The identity within the namespace.
    """

    type: int = 1
    """
    The marker type number.
    """

    action: int = 0
    """
    The marker action number.
    """

    header: Header = field(default_factory=Header)
    """
    The header.
    """

    pose: MarkerPose = field(default_factory=MarkerPose)
    """
    The pose.
    """

    scale: Vector = field(default_factory=lambda: Vector(0.1, 0.1, 0.1))
    """
    The scale.
    """

    color: MarkerColor = field(default_factory=MarkerColor)
    """
    The colour.
    """

    points: List[Vector] = field(default_factory=list)
    """
    The points.
    """

    text: str = ""
    """
    The text.
    """
