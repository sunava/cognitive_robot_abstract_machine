"""
Running a demo file unchanged, as if it were started directly.
"""

from __future__ import annotations

import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import List

MAIN_MODULE = "__main__"
"""
The module name a script runs under when it is started directly.
"""

# %% script


@dataclass(frozen=True)
class DemoScript:
    """
    A demo file and its arguments.
    """

    path: Path
    """
    The demo file.
    """

    arguments: List[str] = field(default_factory=list)
    """
    The arguments handed to the demo.
    """

    def run(self) -> None:
        """
        Run the demo as ``__main__`` with its arguments and its directory first on the
        import path; both are restored afterwards, also when the demo fails.
        """
        original_arguments = list(sys.argv)
        original_path = list(sys.path)
        sys.argv[:] = [str(self.path), *self.arguments]
        sys.path.insert(0, str(self.path.parent))
        try:
            runpy.run_path(str(self.path), run_name=MAIN_MODULE)
        finally:
            sys.argv[:] = original_arguments
            sys.path[:] = original_path
