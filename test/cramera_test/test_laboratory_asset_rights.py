"""
Exported laboratory models retain attribution without changing geometry.
"""

from __future__ import annotations

import copy
import json
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing_extensions import Any

import pytest

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_device_resources import device_resources

# %% isolated export fixtures


@dataclass
class AssetDocument:
    """
    Editable JSON and untouched trailing chunks of an actual authored mesh.
    """

    document: dict[str, Any]
    """Decoded glTF description including geometry and material references."""
    trailing: bytes
    """
    Complete original binary chunks, including their headers and padding.
    """

    @classmethod
    def read(cls, path: Path) -> AssetDocument:
        """
        Read the leading JSON chunk independently of the rights installer.
        """
        content = path.read_bytes()
        magic, version, length = struct.unpack_from("<4sII", content)
        assert (magic, version, length) == (b"glTF", 2, len(content))
        size, kind = struct.unpack_from("<I4s", content, 12)
        assert kind == b"JSON"
        return cls(json.loads(content[20 : 20 + size]), content[20 + size :])

    def write(self, path: Path) -> None:
        """
        Encode a fixture while retaining every original binary chunk.
        """
        content = json.dumps(self.document).encode()
        content += b" " * (-len(content) % 4)
        path.write_bytes(
            struct.pack("<4sII", b"glTF", 2, 20 + len(content) + len(self.trailing))
            + struct.pack("<I4s", len(content), b"JSON")
            + content
            + self.trailing
        )


@pytest.fixture
def rights_module() -> ModuleType:
    """
    Load the packaging helper directly without importing a CRAM package.
    """
    path = Path(__file__).parents[2] / "cramera/tools/laboratory/rights.py"
    specification = spec_from_file_location("laboratory_asset_rights_tool", path)
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def canonical_rights() -> Path:
    """
    Find the authoritative attribution accompanying the source generator.
    """
    return Path(__file__).parents[2] / "cramera/tools/laboratory/asset_rights"


@pytest.fixture
def standalone_device(tmp_path: Path) -> Path:
    """
    Copy one real laboratory mesh and description into an isolated export.
    """
    source = (
        Path(__file__).parents[2] / "cramera/src/cramera/resources/aicor_medical_lab"
    )
    directory = tmp_path / "device"
    (directory / "assets").mkdir(parents=True)
    shutil.copyfile(source / "stopper.urdf", directory / "device.urdf")
    shutil.copyfile(source / "assets/stopper.glb", directory / "assets/stopper.glb")
    (directory / "metadata.json").write_text(json.dumps({"device": "fixture"}))
    return directory


def assert_local_notices(directory: Path, metadata: dict[str, str]) -> None:
    """
    Check that embedded relative references resolve inside a copied export.
    """
    for key in ("licenseFile", "noticeFile"):
        reference = directory / metadata[key]
        assert reference.is_file()
        assert not Path(metadata[key]).is_absolute()


# %% standalone metadata and binary preservation


def test_install_preserves_geometry_binary_chunks_and_prior_attribution(
    standalone_device: Path, rights_module: ModuleType
) -> None:
    """
    Only asset metadata changes, retaining foreign attribution and extensions.
    """
    path = standalone_device / "assets/stopper.glb"
    original = AssetDocument.read(path)
    original.document["asset"]["copyright"] = "Earlier mesh attribution"
    original.document["asset"]["extras"] = {"calibration": [1, 2, 3]}
    original.trailing += struct.pack("<I4s", 8, b"TEST") + b"opaque12"
    original.write(path)
    geometry = copy.deepcopy(original.document)
    geometry.pop("asset")
    description = ElementTree.tostring(
        ElementTree.parse(standalone_device / "device.urdf").getroot()
    )

    rights = rights_module.LaboratoryAssetRights()
    rights.install(standalone_device)

    installed = AssetDocument.read(path)
    asset = installed.document.pop("asset")
    assert installed.document == geometry
    assert installed.trailing == original.trailing
    assert asset["copyright"] == rights.metadata["copyright"]
    assert asset["extras"]["priorCopyright"] == [
        original.document["asset"]["copyright"]
    ]
    assert (
        asset["extras"]["calibration"]
        == original.document["asset"]["extras"]["calibration"]
    )
    expected = dict(
        rights.metadata, licenseFile="../LICENSE.txt", noticeFile="../NOTICE.md"
    )
    assert asset["extras"]["assetRights"] == expected
    assert_local_notices(path.parent, asset["extras"]["assetRights"])
    assert (
        ElementTree.tostring(
            ElementTree.parse(standalone_device / "device.urdf").getroot()
        )
        == description
    )


def test_repeated_install_is_byte_identical_and_copied_folder_is_self_contained(
    standalone_device: Path,
    rights_module: ModuleType,
    canonical_rights: Path,
    tmp_path: Path,
) -> None:
    """
    Repeated packaging neither duplicates notices nor relies on source paths.
    """
    rights = rights_module.LaboratoryAssetRights()
    rights.install(standalone_device)
    first = {
        path.relative_to(standalone_device): path.read_bytes()
        for path in standalone_device.rglob("*")
        if path.is_file()
    }
    rights.install(standalone_device)
    assert {
        path.relative_to(standalone_device): path.read_bytes()
        for path in standalone_device.rglob("*")
        if path.is_file()
    } == first

    copied = tmp_path / "copied"
    shutil.copytree(standalone_device, copied)
    shutil.rmtree(standalone_device)
    for filename in ("rights.json", "LICENSE.txt", "NOTICE.md"):
        assert (copied / filename).read_bytes() == (
            canonical_rights / filename
        ).read_bytes()
    metadata = json.loads((copied / "metadata.json").read_text())
    assert metadata == {"device": "fixture", "assetRights": rights.metadata}
    assert_local_notices(copied, metadata["assetRights"])
    parser = ElementTree.XMLParser(target=ElementTree.TreeBuilder(insert_comments=True))
    comments = [
        element.text
        for element in ElementTree.parse(copied / "device.urdf", parser).getroot()
        if element.tag is ElementTree.Comment
    ]
    matching = [
        comment for comment in comments if rights.metadata["principalAuthor"] in comment
    ]
    assert len(matching) == 1
    for field in ("institution", "license", "licenseFile", "noticeFile"):
        assert rights.metadata[field] in matching[0]
    assert_local_notices(
        copied / "assets",
        AssetDocument.read(copied / "assets/stopper.glb").document["asset"]["extras"][
            "assetRights"
        ],
    )


def test_invalid_glb_is_rejected_without_rewriting_it(
    standalone_device: Path, rights_module: ModuleType
) -> None:
    """
    A truncated binary asset cannot silently acquire misleading metadata.
    """
    path = standalone_device / "assets/stopper.glb"
    content = path.read_bytes()[:-1]
    path.write_bytes(content)
    with pytest.raises(rights_module.InvalidLaboratoryAsset):
        rights_module.LaboratoryAssetRights().install(standalone_device)
    assert path.read_bytes() == content


def test_non_object_asset_extras_are_not_discarded(
    standalone_device: Path, rights_module: ModuleType
) -> None:
    """
    Existing arbitrary glTF extras cause an explicit failure instead of loss.
    """
    path = standalone_device / "assets/stopper.glb"
    original = AssetDocument.read(path)
    original.document["asset"]["extras"] = ["existing external attribution"]
    original.write(path)
    content = path.read_bytes()
    with pytest.raises(rights_module.InvalidLaboratoryAsset):
        rights_module.LaboratoryAssetRights().install(standalone_device)
    assert path.read_bytes() == content


def test_helper_runs_without_site_packages_and_leaves_nested_robots_untouched(
    standalone_device: Path,
) -> None:
    """
    Blender's separate Python can package assets without any CRAM dependency.
    """
    nested = standalone_device / "robots/pr2"
    nested.mkdir(parents=True)
    shutil.copyfile(standalone_device / "device.urdf", nested / "robot.urdf")
    original = (nested / "robot.urdf").read_bytes()
    repository = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("dataset") / "install_laboratory_rights.py"),
            str(repository / "cramera/tools/laboratory/rights.py"),
            str(standalone_device),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (standalone_device / "rights.json").is_file()
    assert (nested / "robot.urdf").read_bytes() == original


# %% generator propagation


def test_new_bundle_contains_rights_in_scene_and_every_root_urdf(
    laboratory_directory: Path, canonical_rights: Path
) -> None:
    """
    Generating a fresh scene already produces attributable portable models.
    """
    expected = json.loads((canonical_rights / "rights.json").read_text())
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    assert scene["assetRights"] == expected
    for filename in ("rights.json", "LICENSE.txt", "NOTICE.md"):
        assert (laboratory_directory / filename).read_bytes() == (
            canonical_rights / filename
        ).read_bytes()
    for path in laboratory_directory.glob("*.urdf"):
        content = path.read_text()
        assert expected["principalAuthor"] in content
        assert expected["licenseFile"] in content
        ElementTree.parse(path)


def test_device_reexport_keeps_each_folder_and_catalog_self_contained(
    device_resources: tuple[Path, Path], canonical_rights: Path
) -> None:
    """
    Re-extracting devices retains notices even when the source is removed.
    """
    from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource

    output, source = device_resources
    repository = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "cramera/tools/laboratory/device_resources.py"),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    shutil.rmtree(source / "assets")
    expected = json.loads((canonical_rights / "rights.json").read_text())
    for directory in (
        output,
        *(
            LaboratoryDeviceResource(device, output).directory
            for device in LaboratoryDevice
        ),
    ):
        for filename in ("rights.json", "LICENSE.txt", "NOTICE.md"):
            assert (directory / filename).read_bytes() == (
                canonical_rights / filename
            ).read_bytes()
        assert_local_notices(
            directory, json.loads((directory / "rights.json").read_text())
        )
        if directory == output:
            continue
        assert (
            json.loads((directory / "metadata.json").read_text())["assetRights"]
            == expected
        )
        for asset in (directory / "assets").glob("*.glb"):
            metadata = AssetDocument.read(asset).document["asset"]
            assert metadata["copyright"] == expected["copyright"]
            assert_local_notices(asset.parent, metadata["extras"]["assetRights"])
