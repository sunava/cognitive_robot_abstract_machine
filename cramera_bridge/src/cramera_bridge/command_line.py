"""
The commands of cramera-bridge.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import List, Optional, Sequence

from cramera.locations import DataLocations
from cramera_bridge.coraplex_backend import CoraplexBackendRegistration
from cramera_bridge.demo_script import DemoScript
from cramera_bridge.onboarding import DemoOnboarding
from cramera_protocol.live_messages import SceneDestination

DEMO_ARGUMENTS_SEPARATOR = "--"
"""
Everything after it on a command line is handed to the demo unread.
"""

# %% arguments


@dataclass(frozen=True)
class CommandLine:
    """
    A command line split into the command's own arguments and those after
    :data:`DEMO_ARGUMENTS_SEPARATOR`, which belong to the demo, such as its own options.
    """

    own: List[str]
    """
    The arguments the command reads.
    """

    demo: List[str]
    """
    The arguments handed to the demo unread.
    """

    @classmethod
    def split(cls, arguments: Sequence[str]) -> CommandLine:
        """
        :param arguments: A command's arguments.
        :return: The arguments split at the first separator.
        """
        arguments = list(arguments)
        if DEMO_ARGUMENTS_SEPARATOR not in arguments:
            return cls(own=arguments, demo=[])
        separator = arguments.index(DEMO_ARGUMENTS_SEPARATOR)
        return cls(own=arguments[:separator], demo=arguments[separator + 1 :])


# %% live


class LiveArgument(StrEnum):
    """
    The arguments of ``cramera-live``.
    """

    DEMO = "demo"
    """
    The demo file to run.
    """

    DEMO_ARGUMENTS = "arguments"
    """
    The arguments handed to the demo.
    """


@dataclass(frozen=True)
class LiveCommand:
    """
    Runs a demo file unchanged with the cramera backend selected, then keeps the
    process, and with it the live publisher, up until it is interrupted.

    The viewer is started separately with ``cramera``; this command only publishes.
    """

    demo: DemoScript
    """
    The demo file and its arguments.
    """

    @classmethod
    def parse(cls, arguments: Sequence[str]) -> LiveCommand:
        """
        :param arguments: The command's arguments.
        :return: The command.
        """
        parser = ArgumentParser(
            prog="cramera-live",
            description="Run a CRAM demo and publish it to the cramera viewer.",
        )
        parser.add_argument(LiveArgument.DEMO, type=Path, help="the demo file")
        parser.add_argument(
            LiveArgument.DEMO_ARGUMENTS, nargs="*", help="arguments for the demo"
        )
        command_line = CommandLine.split(arguments)
        options = vars(parser.parse_args(command_line.own))
        return cls(
            DemoScript(
                options[LiveArgument.DEMO].resolve(),
                options[LiveArgument.DEMO_ARGUMENTS] + command_line.demo,
            )
        )

    def run_demo(self) -> None:
        """
        Run the demo as its own ``__main__`` with the cramera backend selected.
        """
        with CoraplexBackendRegistration():
            self.demo.run()


def live(arguments: Optional[Sequence[str]] = None) -> None:
    """
    ``cramera-live demo.py [arguments] [-- demo options]``: run a demo and keep
    publishing until Ctrl-C.

    :param arguments: The arguments, or None for the process's.
    """
    logging.basicConfig(level=logging.INFO)
    command = LiveCommand.parse(sys.argv[1:] if arguments is None else arguments)
    command.run_demo()
    stop_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, stop_signals)
    logging.getLogger(__name__).info(
        "The demo finished; the live publisher stays up until interrupted."
    )
    signal.sigwait(stop_signals)
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


# %% onboard


class OnboardArgument(StrEnum):
    """
    The arguments of ``cramera-onboard``.
    """

    DEMO = "demo"
    """
    The demo file to run.
    """

    NAME = "--name"
    """
    The bundle name the recording is saved under.
    """

    SHARED = "--shared"
    """
    Save into the shared scenes directory instead of the local one.
    """

    TASK = "--task"
    """
    What the demo does.
    """

    DETECT_EVENTS = "--detect-events"
    """
    Let segmind's detectors watch the run.
    """

    DEMO_ARGUMENTS = "arguments"
    """
    The arguments handed to the demo.
    """


@dataclass(frozen=True)
class OnboardCommand:
    """
    Runs a demo file with the cramera backend and saves its recording as a bundle.
    """

    onboarding: DemoOnboarding
    """
    The onboarding the arguments describe.
    """

    @classmethod
    def parse(
        cls, arguments: Sequence[str], locations: DataLocations
    ) -> OnboardCommand:
        """
        :param arguments: The command's arguments.
        :param locations: Where cramera keeps its scenes.
        :return: The command.
        """
        parser = ArgumentParser(
            prog="cramera-onboard",
            description="Run a CRAM demo and save its run as a cramera scene bundle.",
        )
        parser.add_argument(OnboardArgument.DEMO, type=Path, help="the demo file")
        parser.add_argument(OnboardArgument.NAME, required=True, help="the bundle name")
        parser.add_argument(
            OnboardArgument.SHARED,
            action="store_true",
            help="save into the shared scenes directory",
        )
        parser.add_argument(OnboardArgument.TASK, help="what the demo does")
        parser.add_argument(
            OnboardArgument.DETECT_EVENTS,
            action="store_true",
            help="let segmind's detectors watch the run",
        )
        parser.add_argument(
            OnboardArgument.DEMO_ARGUMENTS, nargs="*", help="arguments for the demo"
        )
        command_line = CommandLine.split(arguments)
        options = parser.parse_args(command_line.own)
        return cls(
            DemoOnboarding(
                demo=DemoScript(
                    options.demo.resolve(), options.arguments + command_line.demo
                ),
                name=options.name,
                locations=locations,
                destination=(
                    SceneDestination.SHARED
                    if options.shared
                    else SceneDestination.LOCAL
                ),
                task=options.task,
                detect_events=options.detect_events,
            )
        )


def onboard(arguments: Optional[Sequence[str]] = None) -> None:
    """
    ``cramera-onboard demo.py --name NAME [--shared] [--task TEXT] [--detect-events]
    [arguments] [-- demo options]``: run a demo and save its run as a scene bundle.

    :param arguments: The arguments, or None for the process's.
    """
    logging.basicConfig(level=logging.INFO)
    command = OnboardCommand.parse(
        sys.argv[1:] if arguments is None else arguments,
        DataLocations.from_environment(dict(os.environ), Path.home()),
    )
    saved = command.onboarding.run()
    logging.getLogger(__name__).info("Saved the run as %s.", saved.scene)
