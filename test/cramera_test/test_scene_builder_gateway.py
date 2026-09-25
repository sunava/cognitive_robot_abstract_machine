"""
The protected scene editor stores drafts without controlling a simulation.
"""

import json
from urllib.parse import urlencode

import pytest

from cramera.scene_builder_http import SceneBuilderRoute
from .test_workshop_share import (
    shared_laboratory,
    sharing_scene,
    fixture_scene,
    sharing_replay,
)


# %% protected authoring workspace
@pytest.fixture
def sharing_prefix() -> str:
    """
    Serve the editor beneath the existing authenticated workshop mount.
    """
    return "/workshop"


def test_builder_catalog_requires_login(shared_laboratory):
    assert (
        shared_laboratory.request(
            shared_laboratory.options.public_path(SceneBuilderRoute.CATALOG)
        ).status
        == 401
    )
    shared_laboratory.login()
    response = shared_laboratory.request(
        shared_laboratory.options.public_path(SceneBuilderRoute.CATALOG)
    )
    assert response.status == 200
    assets = json.loads(response.body)["assets"]
    assert assets
    assert all(
        asset["url"].startswith("/workshop" + SceneBuilderRoute.ASSETS)
        for asset in assets
    )


def test_draft_round_trip_is_isolated_from_robot_commands(shared_laboratory):
    shared_laboratory.login()
    document = {"version": 1, "name": "New furniture layout", "instances": []}
    response = shared_laboratory.request(
        shared_laboratory.options.public_path(SceneBuilderRoute.DRAFTS),
        method="POST",
        body=json.dumps({"document": document}).encode(),
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/json",
        },
    )
    assert response.status == 200
    saved = json.loads(response.body)
    loaded = shared_laboratory.request(
        shared_laboratory.options.public_path(
            SceneBuilderRoute.DRAFTS + "/" + saved["id"]
        )
    )
    assert json.loads(loaded.body) == saved
    assert saved["document"]["instances"] == []
    assert shared_laboratory.backend.requests == []


def test_cross_origin_draft_save_is_rejected(shared_laboratory):
    shared_laboratory.login()
    response = shared_laboratory.request(
        shared_laboratory.options.public_path(SceneBuilderRoute.DRAFTS),
        method="POST",
        body=b"{}",
        headers={
            "Origin": "https://untrusted.example",
            "Content-Type": "application/json",
        },
    )
    assert response.status == 403


def test_direct_builder_login_returns_to_the_editor(shared_laboratory):
    page = shared_laboratory.request(
        shared_laboratory.options.public_path(SceneBuilderRoute.PAGE)
    )
    assert page.status == 200
    assert b'name="next"' in page.body
    result = shared_laboratory.request(
        shared_laboratory.options.public_path("/login"),
        method="POST",
        body=urlencode(
            {"password": shared_laboratory.password, "next": SceneBuilderRoute.PAGE}
        ).encode(),
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert result.status == 303
    assert result.headers["Location"] == "/workshop" + SceneBuilderRoute.PAGE
