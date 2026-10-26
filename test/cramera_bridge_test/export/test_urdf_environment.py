"""
Existing URDF files written as environment bundles.
"""

from __future__ import annotations

import json
import shutil

import pytest

from cramera.scenes import SceneCatalog, SceneDirectory, SceneSource
from cramera_bridge.exceptions import UrdfSourceNotFound
from cramera_bridge.export.urdf_environment import UrdfEnvironmentBundle
from cramera_protocol.bundle.files import BundleFile, ScenesDirectoryFile
from cramera_protocol.bundle.index import SceneIndex
from cramera_protocol.bundle.scene import SceneDescription

from ..dataset.bundles import BUNDLED_CUP_URDF


def test_a_urdf_becomes_an_environment_bundle_the_index_lists(tmp_path):
    loaded = UrdfEnvironmentBundle(tmp_path).write(BUNDLED_CUP_URDF, set())
    bundle = tmp_path / "bundled_cup"
    description = SceneDescription.from_json(
        json.loads((bundle / BundleFile.SCENE).read_text(encoding="utf-8"))
    )
    [model] = description.models
    assert loaded.entry.name == "bundled_cup"
    assert loaded.entry.environment == "bundled_cup"
    assert loaded.missing_meshes == ["package://no_such_cramera_package/meshes/lid.stl"]
    assert description.robot is None and not model.robot
    assert (bundle / model.urdf).is_file()
    assert model.links == 2 and model.movable_joints == ["lid_hinge"]
    index = SceneIndex.from_json(
        json.loads((tmp_path / ScenesDirectoryFile.INDEX).read_text(encoding="utf-8"))
    )
    assert [entry.name for entry in index.scenes] == ["bundled_cup"]
    catalog = SceneCatalog([SceneDirectory(tmp_path, SceneSource.LOCAL)])
    assert catalog.index().scenes == [loaded.entry]


def test_a_name_in_use_is_numbered(tmp_path):
    bundle = UrdfEnvironmentBundle(tmp_path)
    first = bundle.write(BUNDLED_CUP_URDF, set())
    second = bundle.write(BUNDLED_CUP_URDF, set())
    third = bundle.write(BUNDLED_CUP_URDF, {"bundled_cup_3"})
    assert [first.entry.name, second.entry.name, third.entry.name] == [
        "bundled_cup",
        "bundled_cup_2",
        "bundled_cup_4",
    ]


@pytest.mark.parametrize(
    "file_name, bundle_name, label",
    [
        ("my kitchen.urdf.xacro", "my_kitchen", "my kitchen"),
        ("küche.urdf", "k_che", "küche"),
        ("...urdf", "environment", ".."),
    ],
)
def test_a_files_name_becomes_a_valid_bundle_name(
    tmp_path, file_name, bundle_name, label
):
    source = tmp_path / "sources" / file_name
    source.parent.mkdir()
    shutil.copyfile(BUNDLED_CUP_URDF, source)
    bundle = UrdfEnvironmentBundle(tmp_path / "scenes")
    assert bundle.free_name(source, set()) == bundle_name
    assert bundle.label_of(source) == label


def test_a_missing_urdf_writes_nothing(tmp_path):
    with pytest.raises(UrdfSourceNotFound):
        UrdfEnvironmentBundle(tmp_path).write(tmp_path / "none.urdf", set())
    assert list(tmp_path.iterdir()) == []
