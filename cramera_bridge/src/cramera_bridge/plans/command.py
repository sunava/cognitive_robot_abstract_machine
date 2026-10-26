"""
The command line of a generated demo: perform the plan and save its recording, publish a
saved recording, or only stage the plan's world so its objects can be placed by hand.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from argparse import ArgumentParser
from contextlib import ExitStack
from dataclasses import dataclass, field

from typing_extensions import Any, Optional, Sequence

from cramera_bridge.robot_task import TaskArgument, TaskCommand
from cramera_bridge.visualization import LiveVisualization
from cramera_bridge.plans.demo import PlannedTaskDemo
from cramera.lifeline import Lifeline
from cramera_protocol.live import LIVE_PORT
from cramera_protocol.live_messages import SavedScene
from cramera_protocol.plan_builder import DemoDocument, DemoOption

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)
"""
The signals that end a staged world.
"""

# %% staging


@dataclass
class StagedWorld:
    """
    The world of a plan, published until a stop signal arrives, without performing
    anything.
    """

    demo: PlannedTaskDemo
    """
    The demo whose world is staged.
    """

    stopped: threading.Event = field(default_factory=threading.Event)
    """
    Set once the world is to be taken down.
    """

    def run(self) -> None:
        """
        Build and publish the world, wait for a stop signal, then stop publishing and
        restore the signals' earlier handlers.
        """
        world = self.demo.build_world()
        with ExitStack() as published:
            visualization = LiveVisualization(
                world=world, port=self.demo.port, locations=self.demo.locations
            ).start()
            published.callback(visualization.stop)
            for number in STOP_SIGNALS:
                published.callback(signal.signal, number, signal.getsignal(number))
                signal.signal(number, self.stop)
            self.stopped.wait()

    def stop(self, number: int, frame: Any) -> None:
        """
        :param number: The signal that arrived.
        :param frame: The frame it interrupted.
        """
        self.stopped.set()


# %% command


@dataclass(frozen=True)
class PlanDemoCommand:
    """
    Runs the plan a generated demo carries.
    """

    document: DemoDocument
    """
    The plan and where its environment was found.
    """

    @classmethod
    def of_text(cls, text: str) -> PlanDemoCommand:
        """
        :param text: The JSON of a :class:`DemoDocument`.
        :return: The command running it.
        """
        return cls(document=DemoDocument.from_json(json.loads(text)))

    def main(self, arguments: Sequence[str]) -> Optional[SavedScene]:
        """
        :param arguments: The command's arguments.
        :return: The saved or published recording, or None for a staged world.
        """
        plan = self.document.plan
        parser = ArgumentParser(description=plan.task or plan.name)
        parser.add_argument(
            DemoOption.STAGE,
            action="store_true",
            help="only build and publish the world until the process is stopped",
        )
        parser.add_argument(
            DemoOption.LIFELINE,
            action="store_true",
            help="end when standard input closes",
        )
        parser.add_argument(
            DemoOption.PORT,
            type=int,
            default=LIVE_PORT,
            help="the port the world is published on",
        )
        options, remaining = parser.parse_known_args(list(arguments))
        if options.lifeline:
            Lifeline(sys.stdin.fileno()).watch()
        demo = PlannedTaskDemo.of_document(self.document, port=options.port)
        if options.stage:
            StagedWorld(demo).run()
            return None
        return TaskCommand(demo).main(
            [*remaining, TaskArgument.PORT, str(options.port)]
        )
