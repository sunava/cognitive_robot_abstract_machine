"""
Unit tests for the live bridge: state, plan/statechart serializers, hooks and HTTP.

The plan/statechart tests run against stub plan nodes and stub statecharts — no coraplex
or giskardpy needed — because the serializers only touch duck-typed attributes
(children, status.name, designator, life_cycle_state, rx_graph, …). The hook tests
monkeypatch the module-level references to the classes each hook wraps, so no real
coraplex/giskardpy objects are needed either. What IS covered: bottom-up status
aggregation in the plan tree, freeze semantics when a motion group finishes, structure
signatures that let the frontend distinguish "re-colour only" from "rebuild", world
discovery against a real world, and every hook/HTTP route.
"""

import json as json_module
import sys
import types
import urllib.error
import urllib.request
from dataclasses import dataclass

import pytest

import cram_viz.live.hooks as hooks_module
import cram_viz.live.runner as runner_module
from cram_viz.live.bridge import Bridge, MoveRequest, TaskStatus
from cram_viz.live.http import serve
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture()
def fixed_object_world():
    """
    A world with a spoon rigidly fixed to a drawer — not draggable.
    """
    world = World()
    drawer = Body(name=PrefixedName("drawer"))
    spoon = Body(name=PrefixedName("spoon.stl"))
    with world.modify_world():
        world.add_body(drawer)
        connection = FixedConnection.create_with_dofs(world, parent=drawer, child=spoon)
        world.add_connection(connection)
    return world, spoon


class _Status:
    def __init__(self, name):
        self.name = name


def make_node(kind, status="CREATED", designator=None, children=()):
    """
    A stub mimicking the plan-node interface the bridge serializes.
    """
    cls = type(kind, (object,), {})
    node = cls()
    node.status = _Status(status)
    node.designator = designator
    node.children = list(children)
    node.parent_action_node = None
    return node


@dataclass
class _Designator:
    """
    Mimics coraplex's heterogeneous designator schema: some designator subclasses
    declare an object-reference field, others an arm field, or both.
    """

    obj: object = None
    arm: object = None


class _LifeCycleTask:
    """
    Mimics a giskardpy task's life-cycle exposure (``motion_mappings`` values).
    """

    def __init__(self, life_cycle):
        self.life_cycle_state = life_cycle


@pytest.fixture()
def plan_bridge():
    """
    A bridge bound to a small plan: root -> action -> [condition, motion].
    """
    bridge = Bridge()
    motion = make_node(
        "MotionNode",
        designator=_Designator(obj=types.SimpleNamespace(name="world/milk.stl")),
    )
    condition = make_node("ConditionNode")
    action = make_node(
        "ActionNode", designator=_Designator(arm="LEFT"), children=[condition, motion]
    )
    root = make_node("SequentialNode", status="SUCCEEDED", children=[action])
    bridge._bodies = {"milk.stl": object(), "__base__": object()}
    bridge._plan = types.SimpleNamespace(root=root)
    return bridge, root, action, condition, motion


def by_kind(bridge):
    return {n["kind"]: n for n in bridge.get_plan()["nodes"]}


class TestPlanSnapshot:
    def test_running_task_bubbles_up(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        bridge._motion_tasks[id(motion)] = _LifeCycleTask(1)  # RUNNING
        bridge.snapshot_plan()
        nodes = by_kind(bridge)
        assert nodes["MotionNode"]["status"] == "RUNNING"
        assert nodes["MotionNode"]["derived"] is True
        assert nodes["ActionNode"]["status"] == "RUNNING"

    def test_designator_metadata_is_serialized(self, plan_bridge):
        bridge, *_ = plan_bridge
        bridge.snapshot_plan()
        nodes = by_kind(bridge)
        assert nodes["MotionNode"]["target"] == "milk.stl"
        assert nodes["ActionNode"]["arm"] == "LEFT"

    def test_real_status_wins_over_derivation(self, plan_bridge):
        bridge, *_ = plan_bridge
        bridge.snapshot_plan()
        assert by_kind(bridge)["SequentialNode"]["status"] == "SUCCEEDED"
        assert by_kind(bridge)["SequentialNode"]["derived"] is False

    def test_partially_done_parent_is_running_not_succeeded(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        bridge._frozen[id(motion)] = TaskStatus.SUCCEEDED  # condition still CREATED
        bridge.snapshot_plan()
        assert by_kind(bridge)["ActionNode"]["status"] == "RUNNING"

    def test_fully_done_parent_is_succeeded(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        bridge._frozen[id(motion)] = TaskStatus.SUCCEEDED
        bridge._frozen[id(condition)] = TaskStatus.SUCCEEDED
        bridge.snapshot_plan()
        assert by_kind(bridge)["ActionNode"]["status"] == "SUCCEEDED"

    def test_failure_outranks_done_sibling(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        bridge._frozen[id(motion)] = TaskStatus.SUCCEEDED
        bridge._frozen[id(condition)] = TaskStatus.FAILED
        bridge.snapshot_plan()
        assert by_kind(bridge)["ActionNode"]["status"] == "FAILED"

    def test_signature_is_stable_across_status_changes(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        bridge._motion_tasks[id(motion)] = _LifeCycleTask(1)
        bridge.snapshot_plan()
        sig1 = bridge.get_plan()["signature"]
        bridge._motion_tasks.clear()
        bridge._frozen[id(motion)] = TaskStatus.SUCCEEDED
        bridge.snapshot_plan()
        assert bridge.get_plan()["signature"] == sig1


class TestFreezeSemantics:
    def test_freeze_pins_status_and_releases_tasks(self, plan_bridge):
        bridge, root, action, condition, motion = plan_bridge
        executable = types.SimpleNamespace(
            motion_mappings={motion: _LifeCycleTask(1)},
            pre_condition_node=condition,
            post_condition_node=None,
        )
        bridge.bind_motion_group(executable)
        assert id(motion) in bridge._motion_tasks
        bridge.freeze_motion_group(executable, TaskStatus.SUCCEEDED)
        assert id(motion) not in bridge._motion_tasks
        bridge.snapshot_plan()
        nodes = by_kind(bridge)
        assert nodes["MotionNode"]["status"] == "SUCCEEDED"
        assert nodes["ConditionNode"]["status"] == "SUCCEEDED"

    def test_motion_group_title_walks_up_to_the_action_designator(self):
        action_node = make_node("ActionNode", designator=_Designator(arm="LEFT"))
        motion = make_node("MotionNode")
        motion.parent_action_node = action_node
        executable = types.SimpleNamespace(motion_mappings={motion: _LifeCycleTask(1)})
        bridge = Bridge()
        bridge.bind_motion_group(executable)
        assert bridge._chart_title == "_Designator"


# %% apply move ---------------------------------------------------------------
class TestApplyMove:
    def test_dragged_object_lands_at_the_dropped_world_position(
        self, shelved_object_world
    ):
        world, milk = shelved_object_world
        bridge = Bridge()
        bridge.world = world
        bridge._bodies = {"milk.stl": milk}
        bridge.queue_move(
            MoveRequest(
                object_key="milk.stl",
                position=[2.0, 1.0, 0.5],
                quaternion=[0, 0, 0, 1],
                final=True,
            )
        )
        bridge.apply_moves()
        position = milk.global_pose.to_position().to_np().flatten()
        assert position[:3] == pytest.approx([2.0, 1.0, 0.5])

    def test_dragged_object_orientation_is_applied(self, shelved_object_world):
        world, milk = shelved_object_world
        bridge = Bridge()
        bridge.world = world
        bridge._bodies = {"milk.stl": milk}
        quaternion = [0.0, 0.0, 0.70710678, 0.70710678]  # 90 degrees about Z
        bridge.queue_move(
            MoveRequest(
                object_key="milk.stl",
                position=[2.0, 1.0, 0.5],
                quaternion=quaternion,
                final=True,
            )
        )
        bridge.apply_moves()
        result = milk.global_pose.to_quaternion().to_np().flatten()
        assert result[:4] == pytest.approx(quaternion, abs=1e-4)

    def test_quaternion_omitted_keeps_existing_orientation(self, shelved_object_world):
        world, milk = shelved_object_world
        bridge = Bridge()
        bridge.world = world
        bridge._bodies = {"milk.stl": milk}
        quaternion = [0.0, 0.0, 0.70710678, 0.70710678]
        bridge.queue_move(
            MoveRequest(
                object_key="milk.stl",
                position=[2.0, 1.0, 0.5],
                quaternion=quaternion,
                final=True,
            )
        )
        bridge.apply_moves()
        bridge.queue_move(
            MoveRequest(
                object_key="milk.stl", position=[3.0, 1.0, 0.5], quaternion=None, final=False
            )
        )
        bridge.apply_moves()
        result = milk.global_pose.to_quaternion().to_np().flatten()
        assert result[:4] == pytest.approx(quaternion, abs=1e-4)

    def test_unknown_object_key_is_skipped(self, shelved_object_world):
        world, milk = shelved_object_world
        bridge = Bridge()
        bridge.world = world
        bridge._bodies = {"milk.stl": milk}
        original = milk.global_pose.to_position().to_np().flatten()[:3]
        bridge.queue_move(
            MoveRequest(object_key="unknown.stl", position=[9.0, 9.0, 9.0], quaternion=None, final=True)
        )
        bridge.apply_moves()
        position = milk.global_pose.to_position().to_np().flatten()
        assert position[:3] == pytest.approx(original)

    def test_fixed_connection_is_not_draggable(self, fixed_object_world):
        world, spoon = fixed_object_world
        bridge = Bridge()
        bridge.world = world
        bridge._bodies = {"spoon.stl": spoon}
        original = spoon.global_pose.to_position().to_np().flatten()[:3]
        bridge.queue_move(
            MoveRequest(object_key="spoon.stl", position=[9.0, 9.0, 9.0], quaternion=None, final=True)
        )
        bridge.apply_moves()
        position = spoon.global_pose.to_position().to_np().flatten()
        assert position[:3] == pytest.approx(original)


# %% world binding -------------------------------------------------------------
class TestWorldBinding:
    def test_bind_discovers_mesh_named_bodies_and_builds_catalog(
        self, shelved_object_world
    ):
        world, milk = shelved_object_world
        bridge = Bridge()
        bridge.world = world
        bridge._bind()
        assert "milk.stl" in bridge._bodies
        assert {entry["key"] for entry in bridge.object_meta} == {"milk.stl"}

    def test_snapshot_publishes_object_pose(self, shelved_object_world):
        world, milk = shelved_object_world
        bridge = Bridge()
        bridge.world = world
        bridge._bind()
        bridge.snapshot()
        state = bridge.get_state()
        assert state["objects"]["milk.stl"][:3] == pytest.approx(
            [2.37, 2.0, 1.05], abs=1e-3
        )


# %% motion statechart ---------------------------------------------------------
class _StatechartNode:
    def __init__(self, index, name, parent=None):
        self.index = index
        self.name = name
        self.parent_node_index = parent


class _Transition:
    def __init__(self, kind):
        self.kind = types.SimpleNamespace(name=kind)


class _TransitionGraph:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    def edge_index_map(self):
        return {i: e for i, e in enumerate(self._edges)}

    def get_node_data(self, i):
        return self._nodes[i]


def make_chart(life=(1, 1, 0), obs=(0.5, 0.5, 0.0)):
    chart = types.SimpleNamespace()
    chart.nodes = [
        _StatechartNode(0, "Goal"),
        _StatechartNode(1, "MoveJoints", 0),
        _StatechartNode(2, "JointGoalReached"),
    ]
    chart.rx_graph = _TransitionGraph(
        chart.nodes, [(0, 1, _Transition("START")), (1, 2, _Transition("END"))]
    )
    chart.life_cycle_state = types.SimpleNamespace(data=list(life))
    chart.observation_state = types.SimpleNamespace(data=list(obs))
    return chart


class TestChartSnapshot:
    def test_structure_and_states(self):
        bridge = Bridge()
        bridge._chart_title = "TransportAction"
        bridge.observe_chart(make_chart())
        chart = bridge.get_chart()
        assert chart["title"] == "TransportAction"
        assert [n["life"] for n in chart["nodes"]] == [
            "RUNNING",
            "RUNNING",
            "NOT_STARTED",
        ]
        assert [n["observation"] for n in chart["nodes"]] == ["UNKNOWN", "UNKNOWN", "FALSE"]
        assert chart["nodes"][1]["parent"] == "s0"
        assert chart["edges"] == [
            {"from": "s0", "to": "s1", "kind": "START"},
            {"from": "s1", "to": "s2", "kind": "END"},
        ]

    def test_lifecycle_update_keeps_signature(self):
        bridge = Bridge()
        chart = make_chart()
        bridge.observe_chart(chart)
        sig = bridge.get_chart()["signature"]
        chart.life_cycle_state.data = [3, 3, 3]
        bridge.observe_chart(chart)
        assert bridge.get_chart()["signature"] == sig
        assert [n["life"] for n in bridge.get_chart()["nodes"]] == ["DONE"] * 3

    def test_new_chart_replaces_structure(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart())
        sig = bridge.get_chart()["signature"]
        other = types.SimpleNamespace(
            nodes=[_StatechartNode(0, "OtherGoal")],
            rx_graph=_TransitionGraph([_StatechartNode(0, "OtherGoal")], []),
            life_cycle_state=types.SimpleNamespace(data=[1]),
            observation_state=types.SimpleNamespace(data=[1.0]),
        )
        bridge.observe_chart(other)
        chart = bridge.get_chart()
        assert chart["signature"] != sig
        assert len(chart["nodes"]) == 1
        assert chart["nodes"][0]["observation"] == "TRUE"

    def test_trinary_observation_thresholds(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart(obs=(0.0, 0.5, 1.0)))
        assert [n["observation"] for n in bridge.get_chart()["nodes"]] == [
            "FALSE",
            "UNKNOWN",
            "TRUE",
        ]

    def test_none_chart_is_a_noop(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart())
        before = bridge.get_chart()
        bridge.observe_chart(None)
        assert bridge.get_chart() == before


# %% tick hook ------------------------------------------------------------------
class _StubExecutorContext:
    def __init__(self):
        self.world = None


def _make_stub_executor():
    """
    A fresh ``Executor``-shaped class, so each test's monkeypatch of ``tick`` starts
    unpatched (installing a hook mutates the class in place).
    """

    class _StubExecutor:
        def __init__(self):
            self.context = _StubExecutorContext()
            self.motion_statechart = None

        def tick(self, *args, **kwargs):
            return "ticked"

    return _StubExecutor


class TestTickHook:
    def test_tick_hook_binds_world_and_snapshots(self, monkeypatch, shelved_object_world):
        world, milk = shelved_object_world
        stub_executor = _make_stub_executor()
        monkeypatch.setattr(hooks_module, "Executor", stub_executor)
        bridge = Bridge()
        hooks_module.install_tick_hook(bridge)
        executor = stub_executor()
        executor.context.world = world
        result = executor.tick()
        assert result == "ticked"
        assert bridge.world is world
        assert bridge.seq >= 1

    def test_tick_hook_survives_bridge_errors(self, monkeypatch):
        stub_executor = _make_stub_executor()
        monkeypatch.setattr(hooks_module, "Executor", stub_executor)
        bridge = Bridge()
        bridge.world = object()  # truthy, so the tick hook tries to snapshot
        bridge.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        hooks_module.install_tick_hook(bridge)
        executor = stub_executor()
        result = executor.tick()
        assert result == "ticked"

    def test_tick_hook_rejects_double_install(self, monkeypatch):
        stub_executor = _make_stub_executor()
        monkeypatch.setattr(hooks_module, "Executor", stub_executor)
        hooks_module.install_tick_hook(Bridge())
        with pytest.raises(hooks_module.HookAlreadyInstalledError):
            hooks_module.install_tick_hook(Bridge())


# %% plan hooks -----------------------------------------------------------------
def _make_stub_plan():
    """
    A fresh ``Plan``-shaped class (see :func:`_make_stub_executor` for why).
    """

    class _StubPlan:
        def __init__(self):
            self.root = make_node("SequentialNode", children=[])

        def perform(self, *args, **kwargs):
            return "performed"

    return _StubPlan


def _make_stub_executable():
    """
    A fresh ``GiskardExecutable``-shaped class (see :func:`_make_stub_executor`).
    """

    class _StubExecutable:
        def __init__(self, motion_mappings=None):
            self.motion_mappings = motion_mappings or {}
            self.pre_condition_node = None
            self.post_condition_node = None

        def execute(self, *args, **kwargs):
            return "executed"

    return _StubExecutable


class TestPlanHooks:
    def test_perform_captures_plan_and_snapshots(self, monkeypatch):
        stub_plan = _make_stub_plan()
        monkeypatch.setattr(hooks_module, "Plan", stub_plan)
        monkeypatch.setattr(hooks_module, "GiskardExecutable", _make_stub_executable())
        bridge = Bridge()
        hooks_module.install_plan_hooks(bridge)
        plan = stub_plan()
        assert plan.perform() == "performed"
        assert bridge._plan is plan

    def test_execute_freezes_status_on_success(self, monkeypatch):
        monkeypatch.setattr(hooks_module, "Plan", _make_stub_plan())
        stub_executable = _make_stub_executable()
        monkeypatch.setattr(hooks_module, "GiskardExecutable", stub_executable)
        bridge = Bridge()
        hooks_module.install_plan_hooks(bridge)
        motion = make_node("MotionNode")
        executable = stub_executable(motion_mappings={motion: _LifeCycleTask(1)})
        assert executable.execute() == "executed"
        assert bridge._frozen[id(motion)] == TaskStatus.SUCCEEDED

    def test_execute_freezes_status_as_failed_and_reraises(self, monkeypatch):
        monkeypatch.setattr(hooks_module, "Plan", _make_stub_plan())

        class _RaisingExecutable(_make_stub_executable()):
            def execute(self, *args, **kwargs):
                raise RuntimeError("goal aborted")

        monkeypatch.setattr(hooks_module, "GiskardExecutable", _RaisingExecutable)
        bridge = Bridge()
        hooks_module.install_plan_hooks(bridge)
        motion = make_node("MotionNode")
        executable = _RaisingExecutable(motion_mappings={motion: _LifeCycleTask(1)})
        with pytest.raises(RuntimeError):
            executable.execute()
        assert bridge._frozen[id(motion)] == TaskStatus.FAILED

    def test_plan_hooks_reject_double_install(self, monkeypatch):
        monkeypatch.setattr(hooks_module, "Plan", _make_stub_plan())
        monkeypatch.setattr(hooks_module, "GiskardExecutable", _make_stub_executable())
        hooks_module.install_plan_hooks(Bridge())
        with pytest.raises(hooks_module.HookAlreadyInstalledError):
            hooks_module.install_plan_hooks(Bridge())


# %% mesh hook ------------------------------------------------------------------
def _make_stub_mesh_parser():
    """
    A fresh ``MeshParser``-shaped class (see :func:`_make_stub_executor`).
    """

    class _StubMeshParser:
        def __init__(self, file_path):
            self.file_path = file_path

        def parse(self, *args, **kwargs):
            return "world"

    return _StubMeshParser


class TestMeshHook:
    def test_parse_remembers_mesh_path(self, monkeypatch):
        stub_mesh_parser = _make_stub_mesh_parser()
        monkeypatch.setattr(hooks_module, "MeshParser", stub_mesh_parser)
        bridge = Bridge()
        hooks_module.install_mesh_hook(bridge)
        parser = stub_mesh_parser("/tmp/scenes/Milk.STL")
        assert parser.parse() == "world"
        assert bridge._mesh_files["milk.stl"] == "/tmp/scenes/Milk.STL"

    def test_mesh_hook_rejects_double_install(self, monkeypatch):
        monkeypatch.setattr(hooks_module, "MeshParser", _make_stub_mesh_parser())
        hooks_module.install_mesh_hook(Bridge())
        with pytest.raises(hooks_module.HookAlreadyInstalledError):
            hooks_module.install_mesh_hook(Bridge())


# %% runner ---------------------------------------------------------------------
class TestStart:
    def test_start_installs_hooks_in_order_and_binds_the_world(
        self, monkeypatch, shelved_object_world
    ):
        world, milk = shelved_object_world
        calls = []
        captured = {}
        monkeypatch.setattr(
            hooks_module, "install_mesh_hook", lambda bridge: calls.append("mesh")
        )
        monkeypatch.setattr(
            hooks_module, "install_plan_hooks", lambda bridge: calls.append("plan")
        )
        monkeypatch.setattr(
            hooks_module, "install_tick_hook", lambda bridge: calls.append("tick")
        )

        def fake_serve(bridge, port):
            calls.append("serve")
            captured["bridge"] = bridge
            return "server-stub"

        monkeypatch.setattr(runner_module, "serve", fake_serve)
        server = runner_module.start(world=world, port=0)
        assert server == "server-stub"
        assert calls == ["mesh", "plan", "tick", "serve"]
        assert captured["bridge"].world is world
        assert captured["bridge"].seq >= 1  # snapshot() ran before the tick hook

    def test_main_requires_a_demo_path_argument(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["cram-viz-live"])
        with pytest.raises(SystemExit):
            runner_module.main()


# %% http -------------------------------------------------------------------------
@pytest.fixture()
def running_bridge_server():
    """
    A bridge with a live HTTP server on an ephemeral port.
    """
    bridge = Bridge()
    server = serve(bridge, port=0)
    try:
        yield bridge, server
    finally:
        server.shutdown()


def _url(server, path):
    return "http://127.0.0.1:%d%s" % (server.server_address[1], path)


def _get(server, path):
    with urllib.request.urlopen(_url(server, path), timeout=5) as response:
        return response.status, json_module.loads(response.read())


def _post(server, path, payload):
    request = urllib.request.Request(
        _url(server, path), data=json_module.dumps(payload).encode(), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json_module.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json_module.loads(error.read())


class TestHttpRoutes:
    def test_state_route_returns_bridge_state(self, running_bridge_server):
        bridge, server = running_bridge_server
        bridge.state = {"seq": 3, "frames": {}, "base": None, "objects": {}}
        status, payload = _get(server, "/state")
        assert status == 200
        assert payload["seq"] == 3

    def test_unknown_route_is_404(self, running_bridge_server):
        _, server = running_bridge_server
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(_url(server, "/nope"), timeout=5)
        assert excinfo.value.code == 404

    def test_prefixed_path_is_not_confused_with_a_known_route(
        self, running_bridge_server
    ):
        # a naive `startswith("/state")` route match would wrongly accept this
        _, server = running_bridge_server
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(_url(server, "/statechart"), timeout=5)
        assert excinfo.value.code == 404

    def test_mesh_route_404s_when_key_is_unknown(self, running_bridge_server):
        _, server = running_bridge_server
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(_url(server, "/mesh?key=nope"), timeout=5)
        assert excinfo.value.code == 404

    def test_objects_route_returns_the_catalog(self, running_bridge_server):
        bridge, server = running_bridge_server
        bridge.object_meta = [
            {
                "key": "milk.stl",
                "id": "milk",
                "kind": "box",
                "size": [0.1, 0.1, 0.1],
                "color": "#fff",
            }
        ]
        status, payload = _get(server, "/objects")
        assert status == 200
        assert payload["objects"][0]["key"] == "milk.stl"

    def test_info_route_summarizes_attachment_state(self, running_bridge_server):
        _, server = running_bridge_server
        status, payload = _get(server, "/info")
        assert status == 200
        assert payload["running"] is False
        assert payload["movable"] is True

    def test_move_with_valid_payload_is_queued(self, running_bridge_server):
        bridge, server = running_bridge_server
        status, payload = _post(
            server, "/move", {"object": "milk.stl", "position": [1.0, 2.0, 3.0]}
        )
        assert status == 200
        assert payload["ok"] is True
        moves = bridge._moves.drain()
        assert len(moves) == 1
        assert moves[0].object_key == "milk.stl"
        assert moves[0].position == [1.0, 2.0, 3.0]

    def test_move_with_malformed_json_is_a_400(self, running_bridge_server):
        _, server = running_bridge_server
        request = urllib.request.Request(
            _url(server, "/move"), data=b"not json", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=5)
        assert excinfo.value.code == 400
        assert json_module.loads(excinfo.value.read())["ok"] is False

    def test_move_missing_position_is_a_400(self, running_bridge_server):
        _, server = running_bridge_server
        status, payload = _post(server, "/move", {"object": "milk.stl"})
        assert status == 400
        assert payload["ok"] is False

    def test_move_to_unknown_object_is_still_accepted_and_skipped_on_apply(
        self, running_bridge_server
    ):
        # the /move endpoint only validates shape; unknown-body skipping is
        # apply_moves()'s job (see TestApplyMove.test_unknown_object_key_is_skipped)
        bridge, server = running_bridge_server
        status, payload = _post(
            server, "/move", {"object": "nope.stl", "position": [0.0, 0.0, 0.0]}
        )
        assert status == 200
        assert payload["ok"] is True

    def test_options_preflight_sends_cors_headers(self, running_bridge_server):
        _, server = running_bridge_server
        request = urllib.request.Request(_url(server, "/move"), method="OPTIONS")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == "*"
