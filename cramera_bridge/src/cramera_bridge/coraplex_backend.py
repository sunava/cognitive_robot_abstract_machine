"""
Selecting this package's live visualization as coraplex's cramera backend.

coraplex's ``WorldVisualization`` starts its cramera backend by loading the provider
installed under the name ``cramera`` in the entry point group
``coraplex.visualizations``, creating it with ``world=`` and starting it. This package
installs :class:`CoraplexProvider` there, which serves the world through a
:class:`LiveVisualization`, so a demo that selects the backend runs unchanged.

A registration selects the backend while it is entered and lets the visualizations the
provider serves be created by a factory of its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Callable, ClassVar, List, Optional

from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.visualization import PlanVisualization
from cramera_bridge.visualization import LiveVisualization
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan

# %% names


class CoraplexBackend(StrEnum):
    """
    The names coraplex selects and loads its cramera backend by.
    """

    VARIABLE = "CORAPLEX_VISUALIZATION"
    """
    The environment variable naming the backend.
    """

    VALUE = "cramera"
    """
    The value selecting the cramera backend, and the name of its entry point.
    """

    GROUP = "coraplex.visualizations"
    """
    The entry point group coraplex loads the backend's provider from.
    """


VisualizationFactory = Callable[..., LiveVisualization]
"""
Creates the visualization of a world, called as ``factory(world=world)``.
"""

# %% registration


@dataclass
class CoraplexBackendRegistration:
    """
    Selects the cramera backend and makes the provider create its visualizations through
    a factory while entered; leaving restores the environment variable and the factory
    before.
    """

    factory: VisualizationFactory = LiveVisualization
    """
    Creates each visualization coraplex asks for.
    """

    entered: ClassVar[List[VisualizationFactory]] = []
    """
    The factories of the registrations entered, innermost last.
    """

    previous_selection: Optional[str] = field(default=None, init=False)
    """
    The value the environment variable had before, if any.
    """

    @classmethod
    def current_factory(cls) -> VisualizationFactory:
        """
        :return: The factory of the innermost registration entered, else the plain
            live visualization, for a demo that selects the backend itself.
        """
        return cls.entered[-1] if cls.entered else LiveVisualization

    def __enter__(self) -> CoraplexBackendRegistration:
        self.previous_selection = os.environ.get(CoraplexBackend.VARIABLE)
        os.environ[CoraplexBackend.VARIABLE] = CoraplexBackend.VALUE
        self.entered.append(self.factory)
        return self

    def __exit__(self, *exception: object) -> None:
        self.entered.remove(self.factory)
        os.environ.pop(CoraplexBackend.VARIABLE, None)
        if self.previous_selection is not None:
            os.environ[CoraplexBackend.VARIABLE] = self.previous_selection


# %% provider


@dataclass
class CoraplexProvider(PlanVisualization):
    """
    The provider coraplex loads for the cramera backend: serves its world through a
    visualization the current registration's factory creates.
    """

    world: World
    """
    The world served.
    """

    visualization: Optional[LiveVisualization] = field(default=None, init=False)
    """
    The visualization serving the world, while started.
    """

    def start(self) -> CoraplexProvider:
        """
        :return: This provider, serving its world.
        """
        factory = CoraplexBackendRegistration.current_factory()
        self.visualization = factory(world=self.world).start()
        return self

    def stop(self) -> None:
        """
        Stop the visualization, which writes its recording; stopping a provider that
        serves nothing changes nothing.
        """
        if self.visualization is not None:
            self.visualization.stop()
            self.visualization = None

    def plan_callback(self, plan: Plan) -> PlanCallback:
        """
        :param plan: The plan about to perform.
        :return: The callback publishing its progress.
        """
        return self.visualization.plan_callback(plan)
