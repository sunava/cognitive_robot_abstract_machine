"""
Standalone instrument packages remain inside the authenticated laboratory.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pytest

from cramera.laboratory_devices import LaboratoryDeviceResource, LaboratoryInstrument
from cramera.laboratory_share import (
    InstrumentAssets,
    InstrumentFile,
    InstrumentRoute,
    SharedScene,
    ShareRoute,
)

from .test_laboratory_share import (
    SharedLaboratory,
    shared_laboratory,
    sharing_prefix,
    sharing_replay,
    sharing_scene,
)


# %% isolated resource library
@pytest.fixture(autouse=True)
def instrument_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Publish a miniature explicit instrument without touching generated assets.
    """
    root = tmp_path / "instruments"
    fixture = Path(__file__).parent / "dataset/instrument_share"
    catalog = root / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "index.html").write_bytes((fixture / "index.html").read_bytes())
    device = root / LaboratoryInstrument.VORTEX_MIXER
    (device / "assets").mkdir(parents=True)
    for filename in InstrumentFile:
        (device / filename).write_bytes(filename.encode())
    (device / InstrumentFile.MODEL).write_bytes((fixture / "model.urdf").read_bytes())
    (device / "assets/body.glb").write_bytes(b"fixture-mesh")
    for filename in (
        InstrumentFile.README,
        InstrumentFile.LICENSE,
        InstrumentFile.NOTICE,
        InstrumentFile.RIGHTS,
    ):
        (root / filename).write_bytes(filename.encode())
    (device / "private.txt").write_bytes(b"private-test-content")
    (device / "assets/unlisted.glb").write_bytes(b"unlisted-test-content")
    unrelated = root / "confidential"
    unrelated.mkdir()
    (unrelated / InstrumentFile.MODEL).write_bytes(b"unrelated-test-content")
    monkeypatch.setattr(
        LaboratoryDeviceResource,
        "directory",
        property(lambda resource: root / resource.device),
    )
    return root


@pytest.fixture
def instrument_route() -> str:
    """
    Address the known resource independently of its mounted laboratory prefix.
    """
    return f"{InstrumentRoute.ROOT}/{LaboratoryInstrument.VORTEX_MIXER}"


# %% password boundary and continuation
@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
def test_instrument_gallery_login_returns_to_requested_page(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    The existing password form retains the exact gallery destination.
    """
    path = shared_laboratory.options.public_path(InstrumentRoute.CATALOG)
    page = shared_laboratory.request(path)
    assert page.status == 200
    assert f'name="next" value="{InstrumentRoute.CATALOG}"'.encode() in page.body
    assert b"Instrument gallery test fixture" not in page.body
    login = shared_laboratory.request(
        shared_laboratory.options.public_path(ShareRoute.LOGIN),
        method="POST",
        body=urlencode(
            {"password": shared_laboratory.password, "next": InstrumentRoute.CATALOG}
        ).encode(),
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert login.status == 303
    assert login.headers["Location"] == path
    shared_laboratory.cookie = login.headers["Set-Cookie"].split(";", 1)[0]
    page = shared_laboratory.request(path)
    assert page.status == 200
    assert b"Instrument gallery test fixture" in page.body


@pytest.mark.parametrize("filename", list(InstrumentFile) + ["assets/body.glb"])
def test_instrument_files_require_existing_password(
    shared_laboratory: SharedLaboratory,
    instrument_route: str,
    filename: str,
    instrument_library: Path,
) -> None:
    """
    Every source, preview, mesh and rights file is withheld until authentication.
    """
    route = f"{instrument_route}/{filename}"
    assert shared_laboratory.request(route).status == 401
    shared_laboratory.login()
    response = shared_laboratory.request(route)
    assert response.status == 200
    assert (
        response.body
        == (
            instrument_library / LaboratoryInstrument.VORTEX_MIXER / filename
        ).read_bytes()
    )
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    "filename",
    [
        InstrumentFile.README,
        InstrumentFile.LICENSE,
        InstrumentFile.NOTICE,
        InstrumentFile.RIGHTS,
    ],
)
def test_collection_notices_require_login(
    shared_laboratory: SharedLaboratory,
    instrument_library: Path,
    filename: InstrumentFile,
) -> None:
    """
    The gallery's top-level notice links share the same password boundary.
    """
    route = f"{InstrumentRoute.ROOT}/{filename}"
    assert shared_laboratory.request(route).status == 401
    shared_laboratory.login()
    response = shared_laboratory.request(route)
    assert response.status == 200
    assert response.body == (instrument_library / filename).read_bytes()


# %% exact package confinement
@pytest.mark.parametrize(
    "suffix",
    [
        "private.txt",
        "model.blend1",
        "assets/unlisted.glb",
        "assets/body.glb?download=1",
        "../confidential/model.urdf",
        "%2e%2e/confidential/model.urdf",
    ],
)
def test_unlisted_instrument_assets_are_not_exposed(
    shared_laboratory: SharedLaboratory, instrument_route: str, suffix: str
) -> None:
    """
    A known device does not grant access to neighboring or undeclared files.
    """
    shared_laboratory.login()
    assert shared_laboratory.request(f"{instrument_route}/{suffix}").status == 404


def test_unknown_instrument_directory_is_not_exposed(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    A directory outside the instrument enum is never a public package.
    """
    shared_laboratory.login()
    assert (
        shared_laboratory.request(
            f"{InstrumentRoute.ROOT}/confidential/{InstrumentFile.MODEL}"
        ).status
        == 404
    )


def test_replaced_instrument_asset_cannot_escape_its_package(
    shared_laboratory: SharedLaboratory,
    instrument_library: Path,
    instrument_route: str,
    tmp_path: Path,
) -> None:
    """
    Confinement is checked again if a file becomes a symlink after startup.
    """
    shared_laboratory.login()
    outside = tmp_path / "secret"
    outside.write_bytes(b"outside-secret")
    asset = (
        instrument_library / LaboratoryInstrument.VORTEX_MIXER / InstrumentFile.BLENDER
    )
    asset.unlink()
    asset.symlink_to(outside)
    response = shared_laboratory.request(f"{instrument_route}/{InstrumentFile.BLENDER}")
    assert response.status == 404
    assert outside.read_bytes() not in response.body


@pytest.mark.parametrize(
    "sharing_scene", [SharedScene.WORKSHOP, SharedScene.WORKSHOP_REPLAY]
)
def test_nonlaboratory_gateway_has_no_instrument_files(
    shared_laboratory: SharedLaboratory, instrument_route: str
) -> None:
    """
    The resource library is absent from gateways for other scenes.
    """
    shared_laboratory.login()
    assert shared_laboratory.request(InstrumentRoute.CATALOG).status == 404
    assert (
        shared_laboratory.request(f"{instrument_route}/{InstrumentFile.MODEL}").status
        == 404
    )
    login = shared_laboratory.request(
        ShareRoute.LOGIN,
        method="POST",
        body=urlencode(
            {"password": shared_laboratory.password, "next": InstrumentRoute.CATALOG}
        ).encode(),
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert login.status == 401


@pytest.mark.parametrize(
    "destination",
    [
        "https://outside.example.test/",
        "//outside.example.test/",
        "/instruments/catalog/index.html?next=outside",
        "/instruments/vortex_mixer/model.blend",
    ],
)
def test_gallery_login_continuation_rejects_other_destinations(
    shared_laboratory: SharedLaboratory, destination: str
) -> None:
    """
    The gallery login cannot turn a correct password into an arbitrary redirect.
    """
    response = shared_laboratory.request(
        ShareRoute.LOGIN,
        method="POST",
        body=urlencode(
            {"password": shared_laboratory.password, "next": destination}
        ).encode(),
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert response.status == 401
    assert "Location" not in response.headers


def test_instrument_symlink_aliases_are_excluded_at_startup(
    instrument_library: Path,
) -> None:
    """
    Even an in-package symlink cannot alias an unlisted private resource.
    """
    package = instrument_library / LaboratoryInstrument.VORTEX_MIXER
    preview = package / InstrumentFile.PREVIEW
    preview.unlink()
    preview.symlink_to(package / "private.txt")
    files = InstrumentAssets(instrument_library).files()
    assert (
        f"{InstrumentRoute.ROOT}/{LaboratoryInstrument.VORTEX_MIXER}/{InstrumentFile.PREVIEW}"
        not in files
    )


def test_changed_mesh_directory_cannot_alias_another_resource(
    shared_laboratory: SharedLaboratory, instrument_library: Path, instrument_route: str
) -> None:
    """
    Replacing an allowed asset directory cannot enlarge its startup allowlist.
    """
    shared_laboratory.login()
    package = instrument_library / LaboratoryInstrument.VORTEX_MIXER
    assets = package / "assets"
    assets.rename(package / "original_assets")
    private = package / "private_assets"
    private.mkdir()
    (private / "body.glb").write_bytes(b"private-replacement")
    assets.symlink_to(private, target_is_directory=True)
    response = shared_laboratory.request(f"{instrument_route}/assets/body.glb")
    assert response.status == 404
    assert b"private-replacement" not in response.body
