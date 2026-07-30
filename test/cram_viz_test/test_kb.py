"""
Tests for the scene-driven knowledge base and its graph-panel payloads.
"""

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from cram_viz import kb  # noqa: E402  (importable once krrood is present)


@pytest.fixture()
def fresh_kb(fixture_scene):
    kb.reset_kb()
    return kb.get_kb()


# %% entity construction
class TestKB:
    def test_scene_entities(self, fresh_kb):
        assert [o.name for o in fresh_kb.objects] == ["milk", "place_area"]
        assert fresh_kb.robot.name == "pr2"
        assert [a.side for a in fresh_kb.arms] == [kb.BodySide.LEFT]
        assert fresh_kb.arms[0].gripper.name == "left_gripper"

    def test_episodes_link_objects(self, fresh_kb):
        transport = next(e for e in fresh_kb.episodes if e.name == "transport_milk")
        assert transport.picks is fresh_kb.objects[0]
        assert transport.places_at.name == "place_area"
        assert transport.performed_by.side is kb.BodySide.LEFT

    def test_prepare_episode_has_no_manipulation(self, fresh_kb):
        prepare = next(e for e in fresh_kb.episodes if e.name == "prepare")
        assert prepare.performed_by is None
        assert prepare.picks is None
        assert prepare.places_at is None

    def test_joint_motion_ranges(self, fresh_kb):
        torso = next(j for j in fresh_kb.joints if j.name == "torso_lift_joint")
        assert torso.min_rad == 0.0 and torso.max_rad == 0.3

    def test_architecture_scan(self, fresh_kb):
        names = {p.name for p in fresh_kb.packages}
        assert {"coraplex", "krrood"} <= names
        assert any(c.name == "Plan" for c in fresh_kb.classes)


# %% per-joint arm-side resolution (KB._build_joint_motions is a pure staticmethod,
# so each branch is exercised directly without needing a full scene bundle)
class TestJointMotionSides:
    def test_foreign_prefix_is_environment(self):
        trajectory = {"frames": [{"other/some_joint": 0.0}, {"other/some_joint": 1.0}]}
        joints = kb.KB._build_joint_motions(trajectory, parts={}, robot_prefix="pr2")
        assert joints[0].arm_side is kb.BodySide.ENVIRONMENT

    def test_part_lookup_wins_over_name_heuristic(self):
        trajectory = {"frames": [{"pr2/some_joint": 0.0}, {"pr2/some_joint": 1.0}]}
        parts = {"left_arm": ["some_link"]}
        joints = kb.KB._build_joint_motions(trajectory, parts, robot_prefix="pr2")
        assert joints[0].arm_side is kb.BodySide.LEFT

    def test_name_heuristic_when_no_part_matches(self):
        trajectory = {"frames": [{"pr2/r_elbow_joint": 0.0}, {"pr2/r_elbow_joint": 1.0}]}
        joints = kb.KB._build_joint_motions(trajectory, parts={}, robot_prefix="pr2")
        assert joints[0].arm_side is kb.BodySide.RIGHT

    def test_defaults_to_body(self):
        trajectory = {
            "frames": [{"pr2/torso_lift_joint": 0.0}, {"pr2/torso_lift_joint": 1.0}]
        }
        joints = kb.KB._build_joint_motions(trajectory, parts={}, robot_prefix="pr2")
        assert joints[0].arm_side is kb.BodySide.BODY


# %% queries
class TestQueries:
    def test_entity_query(self, fixture_scene):
        result = kb.run_query("the(entity(object).where(object.name == 'milk'))")
        assert result["ok"] and result["count"] == 1
        assert result["rows"][0]["__entity__"] == "milk"
        assert "milk" in result["highlight"]

    def test_class_query_highlights_subpackage_not_class_name(self, fixture_scene):
        # regression test: a PythonClass result must not highlight its own (non
        # graph-node) name — only its subpackage and package, which are real nodes
        result = kb.run_query(
            "the(entity(python_class).where(python_class.name == 'Plan'))"
        )
        assert result["ok"]
        assert result["rows"][0]["__entity__"] == "Plan"
        assert "Plan" not in result["highlight"]
        assert "coraplex.plans" in result["highlight"]
        assert "coraplex" in result["highlight"]

    def test_error_is_reported_not_raised(self, fixture_scene):
        result = kb.run_query("this is not python")
        assert result["ok"] is False
        assert result["error"]

    def test_empty_query_is_reported_not_raised(self, fixture_scene):
        result = kb.run_query("")
        assert result["ok"] is False
        assert "EmptyEqlQueryError" in result["error"]


# %% view payloads
class TestViewPayloads:
    def test_knowledge_view(self, fixture_scene):
        payload = kb.view_payload("knowledge")
        assert payload["ok"]
        ids = {n["id"] for n in payload["nodes"]}
        assert {"pr2", "milk", "transport_milk", "plan"} <= ids
        assert payload["presets"]

    def test_knowledge_view_links_episode_to_architecture(self, fixture_scene):
        payload = kb.view_payload("knowledge")
        edges = {(e["from"], e["to"], e["label"]) for e in payload["edges"]}
        assert ("transport_milk", "coraplex.plans", "planned by") in edges

    def test_knowledge_view_without_architecture(self, fixture_scene, monkeypatch, tmp_path):
        monkeypatch.setenv("CRAM_VIZ_ARCHITECTURE", str(tmp_path / "does-not-exist"))
        kb.reset_kb()
        payload = kb.view_payload("knowledge")
        assert payload["ok"]
        assert "cram" not in {n["id"] for n in payload["nodes"]}

    def test_kinematics_view(self, fixture_scene):
        payload = kb.view_payload("kinematics")
        assert payload["ok"]
        ids = {n["id"] for n in payload["nodes"]}
        assert "urdf:base_link" in ids and "urdf:l_gripper_link" in ids
        # fixed joints render dashed ('type'), movable solid ('prop')
        kinds = {e["label"].split(" ")[0]: e["kind"] for e in payload["edges"]}
        assert kinds["torso_lift_joint"] == "prop"
        assert kinds["l_gripper_joint"] == "type"

    def test_plan_view_carries_status(self, fixture_scene):
        payload = kb.view_payload("plan")
        assert payload["ok"] and payload["layout"] == "hier"
        assert payload["live"] == "plan" and payload["statusLegend"]
        by_label = {n["label"]: n for n in payload["nodes"]}
        assert by_label["SequentialNode"]["status"] == "SUCCEEDED"
        # recorded inner nodes stay CREATED (only the root is performed)
        assert by_label["Transport"]["status"] == "CREATED"
        assert len(payload["edges"]) == len(payload["nodes"]) - 1

    def test_plan_view_always_includes_empty_hint(self, fixture_scene):
        payload = kb.view_payload("plan")
        assert payload["empty"]

    def test_chart_view_is_live_only(self, fixture_scene):
        payload = kb.view_payload("chart")
        assert payload["ok"] and payload["nodes"] == []
        assert payload["live"] == "chart" and payload["empty"]

    def test_unknown_view(self, fixture_scene):
        payload = kb.view_payload("bogus")
        assert not payload["ok"]


# %% drill-down
class TestExpandNode:
    def test_expand_package_shows_subpackages(self, fixture_scene):
        payload = kb.expand_node("coraplex")
        assert payload is not None and payload["ok"]
        assert "coraplex.plans" in {n["id"] for n in payload["nodes"]}

    def test_expand_subpackage_shows_its_classes(self, fixture_scene):
        subpackage_name = next(sp.name for sp in kb.get_kb().subpackages)
        payload = kb.expand_node(subpackage_name)
        assert payload is not None and payload["ok"]
        assert any(n["group"] == "pyclass" for n in payload["nodes"])

    def test_expand_class_shows_inheritance_crumb(self, fixture_scene):
        python_class = next(c for c in kb.get_kb().classes if c.name == "Plan")
        payload = kb.expand_node(kb._class_id(python_class))
        assert payload is not None and payload["ok"]
        assert payload["crumb"] == "Plan"

    def test_expand_unknown_node_returns_none(self, fixture_scene):
        assert kb.expand_node("does-not-exist") is None

    def test_subpackage_view_notes_truncation(self, monkeypatch, fixture_scene):
        monkeypatch.setattr(kb, "CLASS_CAP", 0)
        subpackage = next(
            sp for sp in kb.get_kb().subpackages if sp.name == "coraplex.plans"
        )
        payload = kb._subpackage_view(kb.get_kb(), subpackage)
        assert any(
            "showing" in line for line in payload["details"][subpackage.name]["lines"]
        )
