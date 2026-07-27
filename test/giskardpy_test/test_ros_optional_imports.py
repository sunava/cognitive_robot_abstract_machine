import subprocess
import sys
import textwrap

ROS_IMPORT_BLOCKER = textwrap.dedent(
    """
    import sys

    class RosImportBlocker:
        \"\"\"Raises ImportError for every ROS module, mimicking a ROS-less interpreter.\"\"\"

        blocked_packages = (
            "rclpy",
            "geometry_msgs",
            "std_msgs",
            "visualization_msgs",
            "builtin_interfaces",
        )

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in self.blocked_packages:
                raise ImportError(f"blocked ROS module: {fullname}")
            return None

    sys.meta_path.insert(0, RosImportBlocker())
    """
)


def run_import_without_ros(statements: str) -> subprocess.CompletedProcess:
    """
    Run the given import statements in a fresh interpreter where every ROS
    module import fails.
    """
    return subprocess.run(
        [sys.executable, "-c", ROS_IMPORT_BLOCKER + textwrap.dedent(statements)],
        capture_output=True,
        text=True,
    )


class TestRosOptionalImports:
    """
    Modules with optional ROS visualization must stay importable when ROS is
    absent, so demos and the executor can run in plain simulation.
    """

    def test_debug_expression_publisher_falls_back_to_mocked_node(self):
        result = run_import_without_ros(
            """
            from giskardpy.motion_statechart.debug_expression_publisher import Node
            from semantic_digital_twin.utils import MockedNodeClass

            assert Node is MockedNodeClass
            """
        )
        assert result.returncode == 0, result.stderr

    def test_ros_executor_imports_without_ros(self):
        result = run_import_without_ros("import giskardpy.ros_executor")
        assert result.returncode == 0, result.stderr
