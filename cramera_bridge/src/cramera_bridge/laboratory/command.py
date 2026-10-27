"""
The laboratory transfer's process boundary: a bundle path enters, a recorded scene comes
back.

Dependency discovery imports no CRAM modules.
"""

from __future__ import annotations

import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path

from typing_extensions import List, Optional, Sequence

from cramera_protocol.laboratory.runs import TransferOption
from cramera_protocol.live_messages import SavedScene

CRAM_PACKAGES = (
    "semantic_digital_twin",
    "coraplex",
    "giskardpy",
    "krrood",
    "segmind",
)
"""
The workspace libraries the transfer and the live recording need.
"""


def missing_libraries() -> List[str]:
    """
    :return: The missing workspace modules, without importing native classes.
    """
    return [name for name in CRAM_PACKAGES if importlib.util.find_spec(name) is None]


def run_transfer(bundle: str, arguments: Sequence[str]) -> SavedScene:
    """
    :param bundle: The caller's laboratory bundle directory.
    :param arguments: The transfer's remaining command-line options.
    :return: The saved or published scene, as a protocol document.
    """
    from cramera_bridge.laboratory.transfer import VesselTransferDemo
    from cramera_bridge.robot_task import TaskCommand

    return TaskCommand(VesselTransferDemo.at_the_bench(Path(bundle))).main(arguments)


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """
    :param arguments: The command arguments, or this process's arguments by default.
    """
    parser = ArgumentParser(description="Run the laboratory's PR2 vessel transfer.")
    parser.add_argument(TransferOption.BUNDLE, required=True, type=Path)
    options, remaining = parser.parse_known_args(
        sys.argv[1:] if arguments is None else list(arguments)
    )
    missing = missing_libraries()
    if missing:
        parser.error("The PR2 transfer needs the CRAM workspace: " + ", ".join(missing))
    run_transfer(str(options.bundle), remaining)


if __name__ == "__main__":
    main()
