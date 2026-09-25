"""
A workshop share remains isolated from the laboratory and its live controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cramera.laboratory_robot_session import RobotPhysicsRoute
from cramera.laboratory_share import (
    LaboratoryAssets,
    SharedScene,
    ShareOptions,
    ShareRoute,
)
from cramera.offline_assets import (
    AssetDependencies,
    OfflineAssetError,
    OfflineAssetProblem,
)
from cramera.workshop_session import WorkshopRoute, WorkshopSession
from .test_workshop_session import ContactWorld
from .test_laboratory_share import (
    SharedLaboratory,
    shared_laboratory,
    sharing_prefix,
    sharing_replay,
)


# %% scene isolation
@pytest.fixture
def fixture_scene(fixture_scene: Path) -> Path:
    """
    Extend the miniature bundle with explicitly declared root and mesh notices.
    """
    directory = fixture_scene / "scenes" / "fixture"
    manifest = directory / "scene.json"
    document = json.loads(manifest.read_text())
    document["assetRights"] = {"licenseFile": "LICENSE.txt", "noticeFile": "NOTICE.md"}
    document["objects"].append(
        {"key": "rights_mesh", "mesh": "meshes/recording/rights.glb"}
    )
    manifest.write_text(json.dumps(document))
    mesh_directory = directory / "meshes" / "recording"
    mesh_directory.mkdir(parents=True)
    mesh_document = {
        "asset": {
            "version": "2.0",
            "extras": {
                "assetRights": {
                    "licenseFile": "../LICENSE.txt",
                    "noticeFile": "../NOTICE.md",
                }
            },
        }
    }
    payload = json.dumps(mesh_document).encode()
    payload += b" " * (-len(payload) % 4)
    header = AssetDependencies.GLB_HEADER.pack(
        AssetDependencies.GLB_MAGIC,
        2,
        AssetDependencies.GLB_HEADER.size
        + AssetDependencies.GLB_CHUNK.size
        + len(payload),
    )
    (mesh_directory / "rights.glb").write_bytes(
        header
        + AssetDependencies.GLB_CHUNK.pack(len(payload), AssetDependencies.GLB_JSON)
        + payload
    )
    for parent in (directory, directory / "meshes"):
        (parent / "LICENSE.txt").write_text("Fixture asset licence.")
        (parent / "NOTICE.md").write_text("Fixture author attribution.")
    (directory / "undeclared.txt").write_text("Not part of the shared asset graph.")
    return fixture_scene


@pytest.fixture
def sharing_scene() -> SharedScene:
    """
    Select the independently authenticated workshop.
    """
    return SharedScene.WORKSHOP


def test_workshop_login_selects_workshop(shared_laboratory: SharedLaboratory) -> None:
    """
    Logging in opens the workshop without replacing the laboratory cookie.
    """
    response = shared_laboratory.login()
    options = shared_laboratory.options
    assert response.headers["Location"] == options.viewer_route
    assert options.cookie_name != ShareOptions.COOKIE_NAME
    assert response.headers["Set-Cookie"].startswith(options.cookie_name + "=")
    assert shared_laboratory.request(options.viewer_route).status == 200


def test_workshop_exposes_only_its_own_assets(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    A configured workshop has one scene and cannot read laboratory files.
    """
    shared_laboratory.login()
    response = shared_laboratory.request(ShareRoute.SCENE_INDEX)
    document = json.loads(response.body)
    assert document["default"] == SharedScene.WORKSHOP
    assert [entry["name"] for entry in document["scenes"]] == [SharedScene.WORKSHOP]
    assert (
        shared_laboratory.request(f"/scenes/{SharedScene.WORKSHOP}/robot.urdf").status
        == 200
    )
    assert (
        shared_laboratory.request(f"/scenes/{SharedScene.LABORATORY}/robot.urdf").status
        == 404
    )


@pytest.mark.parametrize(
    "method,route",
    [("GET", RobotPhysicsRoute.STATE), ("POST", RobotPhysicsRoute.RESET)],
)
def test_workshop_cannot_control_laboratory(
    shared_laboratory: SharedLaboratory, method: str, route: str
) -> None:
    """
    The workshop gateway never forwards requests to the laboratory robot.
    """
    shared_laboratory.login()
    response = shared_laboratory.request(
        route,
        method=method,
        body=b"{}" if method == "POST" else None,
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/json",
        },
    )
    assert response.status == 404
    assert shared_laboratory.backend.requests == []


def test_workshop_login_describes_workshop(shared_laboratory: SharedLaboratory) -> None:
    """
    The separate entry point explains the workshop before authentication.
    """
    response = shared_laboratory.request(ShareRoute.ROOT)
    assert b"AICOR Workshop" in response.body
    assert b"contact physics and liquids" not in response.body


@pytest.mark.parametrize("sharing_replay", [True])
def test_recorded_workshop_is_isolated_and_authenticated(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    An explicit second workshop scene shares authentication but no laboratory API.
    """
    route = f"/scenes/{SharedScene.WORKSHOP_REPLAY}/robot.urdf"
    assert shared_laboratory.request(route).status == 401
    shared_laboratory.login()
    assert shared_laboratory.request(route).status == 200
    viewer = f"/index.html?scene={SharedScene.WORKSHOP_REPLAY}&layout=scene&offline=1"
    assert shared_laboratory.request(viewer).status == 200
    index = json.loads(shared_laboratory.request(ShareRoute.SCENE_INDEX).body)
    assert [entry["name"] for entry in index["scenes"]] == [
        SharedScene.WORKSHOP,
        SharedScene.WORKSHOP_REPLAY,
    ]
    assert (
        shared_laboratory.request(f"/scenes/{SharedScene.LABORATORY}/robot.urdf").status
        == 404
    )


@pytest.mark.parametrize("sharing_replay", [False, True])
def test_replay_button_only_targets_an_installed_recording(
    shared_laboratory: SharedLaboratory, sharing_replay: bool
) -> None:
    """
    The gateway supplies a usable replay link only when its assets are present.
    """
    shared_laboratory.login()
    route = f"/scenes/{SharedScene.WORKSHOP}/scene.json"
    scene = json.loads(shared_laboratory.request(route).body)
    link = scene.get("workshop", {}).get("demoUrl")
    if sharing_replay:
        assert (
            link
            == f"index.html?scene={SharedScene.WORKSHOP_REPLAY}&layout=scene&offline=1"
        )
    else:
        assert link is None


# %% explicit asset rights dependencies
@pytest.mark.parametrize("sharing_replay", [True])
def test_declared_asset_rights_require_authentication_at_root_and_mesh_paths(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    Only declared notice files accompany each authenticated workshop bundle.
    """
    files = ("LICENSE.txt", "NOTICE.md", "meshes/LICENSE.txt", "meshes/NOTICE.md")
    routes = [
        f"/scenes/{scene}/{filename}"
        for scene in (SharedScene.WORKSHOP, SharedScene.WORKSHOP_REPLAY)
        for filename in files
    ]
    for route in routes:
        assert shared_laboratory.request(route).status == 401
    shared_laboratory.login()
    for route in routes:
        response = shared_laboratory.request(route)
        assert response.status == 200
        assert response.body in (
            b"Fixture asset licence.",
            b"Fixture author attribution.",
        )
    assert (
        shared_laboratory.request(
            f"/scenes/{SharedScene.WORKSHOP}/undeclared.txt"
        ).status
        == 404
    )


@pytest.mark.parametrize("mesh_notice", [False, True])
def test_asset_rights_cannot_escape_the_scene_directory(
    fixture_scene: Path, mesh_notice: bool
) -> None:
    """
    A declared notice outside the bundle prevents an unsafe allowlist from opening.
    """
    directory = fixture_scene / "scenes" / "fixture"
    (directory.parent / "outside.txt").write_text("Private file outside the scene.")
    if mesh_notice:
        (directory / "meshes" / "LICENSE.txt").unlink()
        (directory / "meshes" / "LICENSE.txt").symlink_to(
            directory.parent / "outside.txt"
        )
    else:
        manifest = directory / "scene.json"
        document = json.loads(manifest.read_text())
        document["assetRights"]["licenseFile"] = "../outside.txt"
        manifest.write_text(json.dumps(document))
    with pytest.raises(OfflineAssetError) as failure:
        LaboratoryAssets(directory, SharedScene.WORKSHOP)
    assert failure.value.problem is OfflineAssetProblem.OUTSIDE_BUNDLE


def test_asset_rights_do_not_expand_the_existing_laboratory_allowlist(
    fixture_scene: Path,
) -> None:
    """
    Laboratory sharing retains its original dependency boundary.
    """
    assets = LaboratoryAssets(
        fixture_scene / "scenes" / "fixture", SharedScene.LABORATORY
    )
    assert f"/scenes/{SharedScene.LABORATORY}/LICENSE.txt" not in assets.files
    assert f"/scenes/{SharedScene.LABORATORY}/meshes/NOTICE.md" not in assets.files


# %% dedicated contact controls
def test_workshop_physics_requires_login_and_runs_only_in_its_gateway(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    Authenticated workshop requests never reach the laboratory backend.
    """
    assert shared_laboratory.request(WorkshopRoute.STATE).status == 401
    shared_laboratory.login()
    result = shared_laboratory.request(WorkshopRoute.STATE)
    assert result.status == 200
    assert json.loads(result.body)["running"] is False
    assert shared_laboratory.backend.requests == []


@pytest.mark.parametrize("sharing_prefix", ["", "/workshop"])
def test_workshop_physics_commands_advance_the_owned_session(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    Mounted and root gateways use one physical session for control and reads.
    """
    world = ContactWorld()
    shared_laboratory.server.workshop_physics = WorkshopSession(factory=lambda: world)
    shared_laboratory.login()
    response = shared_laboratory.request(
        shared_laboratory.options.public_path(WorkshopRoute.COMMAND),
        method="POST",
        body=json.dumps({"action": "start"}).encode(),
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/json",
        },
    )
    assert response.status == 200
    assert json.loads(response.body)["running"] is True
    assert world.stepped.wait(1)
    state = shared_laboratory.request(
        shared_laboratory.options.public_path(WorkshopRoute.STATE)
    )
    assert json.loads(state.body)["time"] > 0
    assert shared_laboratory.backend.requests == []


@pytest.mark.parametrize("origin", [None, "https://untrusted.example.test"])
def test_workshop_physics_rejects_cross_origin_commands(
    shared_laboratory: SharedLaboratory,
    origin: str | None,
) -> None:
    """
    A workshop cookie cannot authorize another site's force commands.
    """
    shared_laboratory.login()
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    result = shared_laboratory.request(
        WorkshopRoute.COMMAND,
        method="POST",
        body=b'{"action":"start"}',
        headers=headers,
    )
    assert result.status == 403


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b'{"action":"drive","direction":2}',
        b'{"action":"move","pose":[0,0,1,0,0,0,NaN]}',
    ],
)
def test_workshop_physics_reports_invalid_commands_as_json(
    shared_laboratory: SharedLaboratory,
    payload: bytes,
) -> None:
    """
    Malformed controls are rejected before any engine is constructed.
    """
    shared_laboratory.login()
    result = shared_laboratory.request(
        WorkshopRoute.COMMAND,
        method="POST",
        body=payload,
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/json",
        },
    )
    assert result.status == 400
    assert "error" in json.loads(result.body)


@pytest.mark.parametrize("sharing_scene", [SharedScene.LABORATORY])
def test_laboratory_does_not_expose_workshop_controls(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    Enabling contact controls leaves the medical lab's API boundary unchanged.
    """
    shared_laboratory.login()
    assert shared_laboratory.request(WorkshopRoute.STATE).status == 404
    result = shared_laboratory.request(
        WorkshopRoute.COMMAND,
        method="POST",
        body=b'{"action":"start"}',
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/json",
        },
    )
    assert result.status == 404
