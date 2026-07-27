from ..giskardpy_test.test_ros_optional_imports import run_import_without_ros


class TestRosOptionalImports:
    """
    The plan and action modules a demo needs must stay importable when ROS is
    absent, so demos can run in plain simulation.
    """

    def test_demo_plan_modules_import_without_ros(self):
        result = run_import_without_ros(
            """
            import coraplex.plans.plan_node
            import coraplex.plans.factories
            import coraplex.robot_plans.actions.composite.transporting
            import coraplex.execution_environment
            """
        )
        assert result.returncode == 0, result.stderr
