from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.ros_context import RosContextExtension
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import ExecutionType
from coraplex.perception import PerceptionInterface, PerceptionQuery
from coraplex.plans.executables import GiskardExecutable, HasResult
from coraplex.robot_plans.motions.base import BaseMotion

# %% perceiving inside the motion chart


@dataclass(eq=False, repr=False)
class PerceptionTask(Task, HasResult):
    """
    Motion statechart node that answers a perception query and writes what it saw into
    the world.

    The node adds no motion constraints. The query is answered on the node's first tick,
    so the whole detection takes one tick however long the source needs to reply, and the
    node then observes
    :attr:`~giskardpy.motion_statechart.data_types.ObservationStateValues.TRUE` so the
    surrounding sequence continues. Its :attr:`result` is the :attr:`detected_body`.

    .. warning:: That tick blocks until the source replies, which on the real robot holds
        up the control loop for as long as the pipeline takes to answer.
    """

    query: PerceptionQuery = field(kw_only=True)
    """
    What to look for and where.
    """

    execution_type: ExecutionType | None = field(kw_only=True)
    """
    Which source answers the query.

    Carried by the node rather than read from the execution environment, because on the
    real robot the chart is answered in the controller's process, where that environment
    does not exist. None when the chart was built without one, which :meth:`build`
    rejects.
    """

    perception_source: PerceptionInterface | None = field(init=False, default=None)
    """
    The source answering the query, resolved during :meth:`build`.
    """

    detected_body: Body | None = field(init=False, default=None, repr=False)
    """
    The body the query was answered with and moved to where it was seen, None while the
    query has not been answered.
    """

    accept_first_if_multiple: bool = False
    """
    Whether several candidates may be resolved by taking the first one.

    When False, several candidates raise
    :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen between.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.perception_source = PerceptionInterface.for_execution_type(
            self.execution_type,
            context.require_extension(RosContextExtension).ros_node,
        )
        return NodeArtifacts()

    @property
    def result(self) -> Body | None:
        return self.detected_body

    def on_start(self, context: MotionStatechartContext) -> None:
        self.detected_body = None

    def on_tick(
        self, context: MotionStatechartContext
    ) -> ObservationStateValues | None:
        if self.detected_body is not None:
            return ObservationStateValues.TRUE
        detection = self.perception_source.detect(
            self.query, self.accept_first_if_multiple
        )
        detected_annotations = detection.apply_to(
            self.query.world, trust_orientation=self.query.trust_detected_orientation
        )
        self.detected_body = detected_annotations[0].root
        return ObservationStateValues.TRUE


@dataclass
class DetectingMotion(BaseMotion):
    """
    Detects the object a perception query asks for and moves it to where it was seen.

    The detection runs inside the motion statechart, so it merges with the surrounding
    motions and a motion planned after it binds the corrected pose when it starts.
    """

    query: PerceptionQuery
    """
    Query for the perception system that should be answered.
    """

    accept_first_if_multiple: bool = False
    """
    Whether several candidates may be resolved by taking the first one.

    When False, several candidates raise
    :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen between.
    """

    @property
    def _motion_chart(self) -> Task:
        return PerceptionTask(
            query=self.query,
            execution_type=GiskardExecutable.execution_type,
            accept_first_if_multiple=self.accept_first_if_multiple,
        )
