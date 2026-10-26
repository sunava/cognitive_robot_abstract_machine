"""
Running a demo file as if it were started directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cramera_bridge.demo_script import DemoScript

EXITING_SCRIPT = Path(__file__).parent / "dataset" / "scripts" / "exiting_script.py"
"""
A script exiting with the status it is given.
"""

EXIT_STATUS = 4
"""
The status the script is asked to exit with.
"""


def test_a_failing_demo_leaves_the_arguments_and_the_import_path_as_they_were():
    arguments_before, path_before = list(sys.argv), list(sys.path)
    with pytest.raises(SystemExit) as raised:
        DemoScript(EXITING_SCRIPT, [str(EXIT_STATUS)]).run()
    assert raised.value.code == EXIT_STATUS
    assert (sys.argv, sys.path) == (arguments_before, path_before)
