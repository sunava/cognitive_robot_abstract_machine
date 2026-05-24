from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Any, ClassVar

from typing_extensions import TYPE_CHECKING

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    ExternalCollisionDistanceMonitor,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion, CancelMotion
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from pycram.datastructures.enums import ExecutionType
from semantic_digital_twin.robots.abstract_robot import AbstractRobot

from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from pycram.plans.plan_node import PlanNode

logger = logging.getLogger(__name__)
DEBUG_PROFILE_MOTION_EXECUTOR = True


@dataclass
class MotionExecutor:
    DEFAULT_SIMULATION_TIMEOUT_TICKS: ClassVar[int] = 2000
    """
    Default number of simulation ticks before stagnation checks may abort a motion.
    Individual actions may override this via ActionDescription.motion_timeout_ticks.
    """

    STAGNATION_CHECK_INTERVAL_TICKS: ClassVar[int] = 25
    STAGNATION_DISTANCE_EPSILON_M: ClassVar[float] = 0.002
    STAGNATION_CHECKS_BEFORE_ABORT: ClassVar[int] = 4
    WIGGLE_DISTANCE_EPSILON_M: ClassVar[float] = 0.025
    ROBOT_WIGGLE_DISTANCE_EPSILON_M: ClassVar[dict[str, float]] = {
        "Justin": 0.07,
    }
    ACTION_WIGGLE_DISTANCE_EPSILON_M: ClassVar[dict[str, float]] = {
        "MixingAction": 0.06,
        "WipingAction": 0.05,
    }
    WIGGLE_CHECKS_BEFORE_ABORT: ClassVar[int] = 8
    HARD_TIMEOUT_FACTOR: ClassVar[int] = 10

    motions: List[MotionStatechartNode]
    """
    The motions to execute
    """

    world: World
    """
    The world in which the motions should be executed.
    """

    motion_state_chart: MotionStatechart = field(init=False)
    """
    Giskard's motion state chart that is created from the motions.
    """

    ros_node: Any = field(kw_only=True, default=None)
    """
    ROS node that should be used for communication. Only relevant for real execution.
    """

    with_collision_avoidance: ClassVar[bool] = True
    plan_node: PlanNode = field(kw_only=True)
    """
    The plan node that created this executor.
    """

    execution_type: ClassVar[ExecutionType] = None

    @property
    def simulation_timeout_ticks(self) -> int:
        timeout_ticks = getattr(self.plan_node.action, "motion_timeout_ticks", None)
        if timeout_ticks is None:
            return self.DEFAULT_SIMULATION_TIMEOUT_TICKS
        return int(timeout_ticks)

    @property
    def simulation_hard_timeout_ticks(self) -> int:
        hard_timeout_ticks = getattr(
            self.plan_node.action, "motion_hard_timeout_ticks", None
        )
        if hard_timeout_ticks is not None:
            return int(hard_timeout_ticks)
        return self.simulation_timeout_ticks * self.HARD_TIMEOUT_FACTOR

    def _progress_robot(self):
        try:
            return self.world.get_semantic_annotations_by_type(AbstractRobot)[0]
        except Exception:
            return None

    def _progress_bodies(self, robot):
        if robot is None:
            return []

        bodies = []
        for chain in getattr(robot, "manipulator_chains", []) or []:
            bodies.extend(getattr(chain, "bodies", []) or [])

        unique_bodies = []
        seen = set()
        for body in bodies:
            body_id = id(body)
            if body_id in seen:
                continue
            seen.add(body_id)
            unique_bodies.append(body)
        return unique_bodies

    def _wiggle_distance_epsilon(self, robot) -> float:
        if robot is None:
            return self.WIGGLE_DISTANCE_EPSILON_M
        return self.ROBOT_WIGGLE_DISTANCE_EPSILON_M.get(
            robot.__class__.__name__,
            self.WIGGLE_DISTANCE_EPSILON_M,
        )

    def _body_position_snapshot(self, bodies):
        snapshot = {}
        for body in bodies:
            try:
                position = body.global_pose.to_position().to_np()
                snapshot[id(body)] = tuple(float(v) for v in position[:3])
            except Exception:
                continue
        return snapshot

    @staticmethod
    def _max_snapshot_displacement(previous_snapshot, current_snapshot) -> float:
        max_displacement = 0.0
        for body_id, current_position in current_snapshot.items():
            previous_position = previous_snapshot.get(body_id)
            if previous_position is None:
                continue
            displacement = (
                sum(
                    (current_position[idx] - previous_position[idx]) ** 2
                    for idx in range(3)
                )
                ** 0.5
            )
            max_displacement = max(max_displacement, displacement)
        return max_displacement

    def construct_msc(self):
        self.motion_state_chart = MotionStatechart()
        sequence_node = Sequence(nodes=self.motions)
        if self.with_collision_avoidance:
            self.motion_state_chart.add_node(ExternalCollisionAvoidance())
            robot = self.world.get_semantic_annotations_by_type(AbstractRobot)[0]
            self.motion_state_chart.add_node(
                monitor1 := ExternalCollisionDistanceMonitor(
                    body=robot.root,
                    threshold=0,
                )
            )
            #
            # self.motion_state_chart.add_node(
            #     monitor2 := ExternalCollisionDistanceMonitor(
            #         body=self.world.get_body_by_name("r_shoulder_pan_link"),
            #         threshold=0,
            #     )
            # )
            # self.motion_state_chart.add_node(
            #     monitor3 := ExternalCollisionDistanceMonitor(
            #         body=self.world.get_body_by_name("l_shoulder_pan_link"),
            #         threshold=0,
            #     )
            # )
            # self.motion_state_chart.add_node(
            #     CancelMotion.when_any_true([monitor1], exception=Exception(":("))
            # )
            self.motion_state_chart.add_node(
                CancelMotion.when_true(monitor1, exception=Exception(":("))
            )
        self.motion_state_chart.add_node(sequence_node)

        self.motion_state_chart.add_node(EndMotion.when_true(sequence_node))

    def execute(self):
        """
        Executes the constructed motion state chart in the given world.
        """
        # If there are no motions to construct an msc, return
        if len(self.motions) == 0:
            return
        match MotionExecutor.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_for_simulation()
            case ExecutionType.REAL:
                self._execute_for_real()
            case ExecutionType.NO_EXECUTION:
                return
            case _:
                logger.error(f"Unknown execution type: {MotionExecutor.execution_type}")

    def _execute_for_simulation(self):
        """
        Creates an executor and executes the motion state chart until it is done.
        """
        logger.debug(f"Executing {self.motions} motions in simulation")
        setup_start = time.time()
        executor = Ros2Executor(
            context=MotionStatechartContext(
                world=self.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
            ros_node=self.ros_node,
        )
        compile_start = time.time()
        executor.compile(self.motion_state_chart)
        if DEBUG_PROFILE_MOTION_EXECUTOR:
            logger.warning(
                "MotionExecutor profile: motions=%s compile=%.3fs setup=%.3fs world_bodies=%s collision_avoidance=%s",
                len(self.motions),
                time.time() - compile_start,
                time.time() - setup_start,
                len(getattr(self.world, "bodies", [])),
                self.with_collision_avoidance,
            )
        try:
            # execute the motion state chart until it is done
            counter = 0
            timeout_ticks = self.simulation_timeout_ticks
            hard_timeout_ticks = self.simulation_hard_timeout_ticks
            progress_robot = self._progress_robot()
            progress_bodies = self._progress_bodies(progress_robot)
            wiggle_distance_epsilon = self._wiggle_distance_epsilon(progress_robot)
            wiggle_distance_epsilon = max(
                wiggle_distance_epsilon,
                self.ACTION_WIGGLE_DISTANCE_EPSILON_M.get(
                    self.plan_node.action.__class__.__name__,
                    wiggle_distance_epsilon,
                ),
            )
            previous_progress_snapshot = self._body_position_snapshot(progress_bodies)
            stagnant_checks = 0
            wiggle_checks = 0
            tick_start = time.time()
            while counter < hard_timeout_ticks:
                if self.plan_node.is_interrupted:
                    return
                elif self.plan_node.is_paused:
                    time.sleep(0.01)
                    continue

                executor.tick()
                counter += 1
                if executor.motion_statechart.is_end_motion():
                    break

                if (
                    counter >= timeout_ticks
                    and counter % self.STAGNATION_CHECK_INTERVAL_TICKS == 0
                ):
                    current_progress_snapshot = self._body_position_snapshot(
                        progress_bodies
                    )
                    max_displacement = self._max_snapshot_displacement(
                        previous_progress_snapshot,
                        current_progress_snapshot,
                    )
                    previous_progress_snapshot = current_progress_snapshot

                    if max_displacement < self.STAGNATION_DISTANCE_EPSILON_M:
                        stagnant_checks += 1
                        wiggle_checks += 1
                        progress_status = "stagnant"
                    elif max_displacement < wiggle_distance_epsilon:
                        stagnant_checks = 0
                        wiggle_checks += 1
                        progress_status = "wiggle"
                    else:
                        stagnant_checks = 0
                        wiggle_checks = 0
                        progress_status = "moving"

                    logger.warning(
                        "MotionExecutor progress check: action=%s ticks=%s manipulator_links=%s status=%s max_link_displacement=%.5fm wiggle_epsilon=%.5fm stagnant_checks=%s/%s wiggle_checks=%s/%s",
                        self.plan_node.action.__class__.__name__,
                        counter,
                        len(progress_bodies),
                        progress_status,
                        max_displacement,
                        wiggle_distance_epsilon,
                        stagnant_checks,
                        self.STAGNATION_CHECKS_BEFORE_ABORT,
                        wiggle_checks,
                        self.WIGGLE_CHECKS_BEFORE_ABORT,
                    )

                    if (
                        stagnant_checks >= self.STAGNATION_CHECKS_BEFORE_ABORT
                        or wiggle_checks >= self.WIGGLE_CHECKS_BEFORE_ABORT
                    ):
                        wiggle_detected = progress_status == "wiggle"
                        raise TimeoutError(
                            "Motion stalled while waiting for end of motion "
                            f"after {counter} ticks in {self.plan_node.action.__class__.__name__} "
                            f"(status={progress_status}, max link displacement "
                            f"{max_displacement:.5f}m over {self.STAGNATION_CHECK_INTERVAL_TICKS} ticks, "
                            f"wiggle={wiggle_detected})."
                        )
            else:
                raise TimeoutError(
                    "Hard timeout reached while waiting for end of motion "
                    f"after {hard_timeout_ticks} ticks in {self.plan_node.action.__class__.__name__}."
                )
            if DEBUG_PROFILE_MOTION_EXECUTOR:
                logger.warning(
                    "MotionExecutor profile: motions=%s ticks=%s tick_total=%.3fs world_bodies=%s",
                    len(self.motions),
                    counter,
                    time.time() - tick_start,
                    len(getattr(self.world, "bodies", [])),
                )

        except TimeoutError as e:
            accept_timeout = getattr(
                self.plan_node.action, "_accept_motion_timeout_as_success", None
            )
            if accept_timeout is not None and accept_timeout(e):
                return

            failed_nodes = [
                (
                    node
                    if node.life_cycle_state
                    not in [LifeCycleValues.DONE, LifeCycleValues.NOT_STARTED]
                    else None
                )
                for node in self.motion_state_chart.nodes
            ]
            failed_nodes = list(filter(None, failed_nodes))
            logger.error(f"Failed Nodes: {failed_nodes}")
            raise e
        finally:
            executor._set_velocity_acceleration_jerk_to_zero()
            executor.motion_statechart.cleanup_nodes(context=executor.context)
            executor.context.cleanup()

    def _monitor_interrupt(self, giskard_wrapper, kill_event: threading.Event):
        while True:
            if self.plan_node.is_paused:
                raise NotImplementedError("Pause not implemented for real execution")
            elif self.plan_node.is_interrupted or kill_event.is_set():
                giskard_wrapper.cancel_goal_async()
            time.sleep(0.01)

    def _execute_for_real(self):
        from giskardpy_ros.python_interface.python_interface import GiskardWrapper

        giskard = GiskardWrapper(self.ros_node)

        kill_event = threading.Event()
        interrupt_thread = threading.Thread(
            target=self._monitor_interrupt, args=(giskard, kill_event)
        )
        interrupt_thread.start()

        giskard.execute(self.motion_state_chart)

        kill_event.set()
        interrupt_thread.join()


@dataclass
class ExecutionEnvironment:
    """
    Base class for managing execution context of all actions within. Instances of this class is to be used with a
    "with" context block

    Example:

        >>> with ExecutionEnvironment(ExecutionType.SIMULATED):
        >>>     SequentialPlan(context, NavigateActionDescription, ...)

    """

    execution_type: ExecutionType
    """
    The type of the execution environment 
    """

    previous_type: ExecutionType = field(init=False, default=None)
    """
    Type of the execution environment before setting it, used for nested environments
    """

    with_collision_avoidance: bool = field(kw_only=True, default=True)
    """
    Whether to use collision avoidance in the execution environment
    """

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set :py:attr:`~MotionExecutor.execution_type` and
        sets it to 'real'
        """
        self.pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = self.execution_type
        MotionExecutor.with_collision_avoidance = self.with_collision_avoidance

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, sets the :py:attr:`~MotionExecutor.execution_type` to the previously
        used one.
        """
        MotionExecutor.execution_type = self.pre

    def __call__(self):
        return self


# These are imported, so they don't have to be initialized when executing with
simulated_robot = ExecutionEnvironment(ExecutionType.SIMULATED)
simulated_robot_without_collision = ExecutionEnvironment(
    ExecutionType.SIMULATED, with_collision_avoidance=False
)
simulated_robot_with_collision = ExecutionEnvironment(
    ExecutionType.SIMULATED, with_collision_avoidance=True
)
real_robot = ExecutionEnvironment(ExecutionType.REAL)
semi_real_robot = ExecutionEnvironment(ExecutionType.SEMI_REAL)
no_execution = ExecutionEnvironment(ExecutionType.NO_EXECUTION)
