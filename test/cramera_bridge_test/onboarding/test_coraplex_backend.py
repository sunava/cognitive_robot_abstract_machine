"""
Selecting this package's live visualization as coraplex's cramera backend.
"""

from __future__ import annotations

import os
from importlib.metadata import entry_points

from cramera_bridge.coraplex_backend import (
    CoraplexBackend,
    CoraplexBackendRegistration,
    CoraplexProvider,
)
from cramera_bridge.visualization import LiveVisualization
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex

PREVIOUS_SELECTION = "rviz"
"""
A backend a process selected before the registration.
"""


def test_the_provider_is_installed_where_coraplex_loads_it():
    (installed,) = entry_points(group=CoraplexBackend.GROUP, name=CoraplexBackend.VALUE)
    assert installed.load() is CoraplexProvider


def test_a_registration_selects_the_backend_while_entered(monkeypatch):
    monkeypatch.delenv(CoraplexBackend.VARIABLE, raising=False)
    with CoraplexBackendRegistration():
        selection = os.environ[CoraplexBackend.VARIABLE]
        factory = CoraplexBackendRegistration.current_factory()
    assert selection == CoraplexBackend.VALUE
    assert factory is LiveVisualization
    assert CoraplexBackend.VARIABLE not in os.environ


def test_leaving_restores_what_the_registration_replaced(monkeypatch):
    def factory(world) -> LiveVisualization:
        return LiveVisualization(world=world)

    monkeypatch.setenv(CoraplexBackend.VARIABLE, PREVIOUS_SELECTION)
    with CoraplexBackendRegistration(factory=factory):
        with CoraplexBackendRegistration():
            pass
        inner_left = CoraplexBackendRegistration.current_factory()
    assert inner_left is factory
    assert CoraplexBackendRegistration.current_factory() is LiveVisualization
    assert os.environ[CoraplexBackend.VARIABLE] == PREVIOUS_SELECTION


def test_coraplex_creates_its_visualization_through_the_factory(
    sliding_box, locations, monkeypatch, coraplex_visualization
):
    created = []

    def factory(world) -> LiveVisualization:
        visualization = LiveVisualization(
            world=world,
            port=0,
            locations=locations,
            read_ros_markers=False,
            class_index=WorkspaceClassIndex(),
        )
        created.append(visualization)
        return visualization

    monkeypatch.delenv(CoraplexBackend.VARIABLE, raising=False)
    with CoraplexBackendRegistration(factory=factory):
        visualization = coraplex_visualization.WorldVisualization.from_environment(
            sliding_box.world
        ).start()
    shown = visualization.provider.visualization
    visualization.stop()
    assert created == [shown]
    assert shown.world is sliding_box.world
