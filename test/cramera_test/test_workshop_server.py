"""
The local viewer exposes the same physical workshop controls as its gateway.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cramera import workshop_physics
from cramera.workshop_session import WorkshopRoute
from .test_workshop_session import ContactWorld
from .test_server import server


# %% shared local HTTP fixture
@pytest.fixture
def contact_world(monkeypatch) -> ContactWorld:
    """
    Replace only workshop dynamics while preserving the real HTTP listener.
    """
    world = ContactWorld()
    monkeypatch.setattr(workshop_physics, "WorkshopPhysics", lambda: world)
    return world


def test_local_workshop_commands_and_state_share_the_engine(
    contact_world, server
) -> None:
    """
    A local client sees time advance after starting its physical world.
    """
    request = Request(
        server + WorkshopRoute.COMMAND,
        data=json.dumps({"action": "start"}).encode(),
        headers={"Content-Type": "application/json", "Origin": server},
    )
    with urlopen(request) as response:
        assert response.status == 200
        assert json.load(response)["running"] is True
    assert contact_world.stepped.wait(1)
    with urlopen(server + WorkshopRoute.STATE) as response:
        assert json.load(response)["time"] > 0


def test_local_workshop_rejects_an_external_origin(contact_world, server) -> None:
    """
    A webpage on another host cannot move local workshop objects.
    """
    request = Request(
        server + WorkshopRoute.COMMAND,
        data=json.dumps({"action": "start"}).encode(),
        headers={
            "Content-Type": "application/json",
            "Origin": "https://untrusted.example.test",
        },
    )
    with pytest.raises(HTTPError) as failure:
        urlopen(request)
    assert failure.value.code == 403
    assert contact_world.time == 0.0
