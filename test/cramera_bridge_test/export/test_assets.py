"""
Copying mesh files and the files they reference into a bundle.
"""

from __future__ import annotations

import pytest

from cramera_bridge.exceptions import PathOutsideBundle
from cramera_bridge.export.assets import (
    BundledAssets,
    MaterialLibrary,
    MeshFormat,
    MeshReference,
)

from ..dataset.worlds import CUP_MESH, MESHES

TEXTURED_MESHES = MESHES / "textured"
"""
An OBJ whose material library names a texture that does not exist.
"""

COLLADA_MESHES = MESHES / "collada"
"""
A COLLADA document naming a texture beside it.
"""

# %% formats


@pytest.mark.parametrize("member", list(MeshFormat))
def test_a_mesh_format_is_read_from_the_suffix(member):
    assert MeshFormat.of_path(f"package://kitchen/meshes/cup{member.upper()}") is member


def test_a_file_without_a_mesh_suffix_is_no_mesh():
    assert MeshFormat.of_path("libplugin.so") is None


def test_the_material_library_of_an_obj_is_found():
    assert MaterialLibrary.companion_of(CUP_MESH) == MESHES / "cup.mtl"
    assert MaterialLibrary.declared_by(MESHES / "cup.mtl") == []


# %% copying


def test_equal_file_names_of_different_directories_get_different_folders(tmp_path):
    assets = BundledAssets(bundle_root=tmp_path / "bundle")
    first = tmp_path / "first" / "cup.obj"
    second = tmp_path / "second" / "cup.obj"
    assert assets.folder_for(first, "x") != assets.folder_for(second, "x")
    assert assets.folder_for(first, "x") == assets.folder_for(first.with_name("a"), "x")


def test_a_missing_source_is_reported_not_replaced(tmp_path):
    assets = BundledAssets(bundle_root=tmp_path)
    assert assets.copy(tmp_path / "gone.stl", tmp_path / "meshes" / "gone.stl") is False
    assert assets.missing == [str(tmp_path / "gone.stl")]
    assert not (tmp_path / "meshes").exists()


def test_a_destination_outside_the_bundle_is_refused(tmp_path):
    assets = BundledAssets(bundle_root=tmp_path / "bundle")
    with pytest.raises(PathOutsideBundle):
        assets.copy(CUP_MESH, tmp_path / "elsewhere" / "cup.obj")


def test_a_texture_named_by_a_material_library_is_missing_when_absent(tmp_path):
    source = TEXTURED_MESHES / "box.obj"
    assets = BundledAssets(bundle_root=tmp_path)
    destination = tmp_path / "meshes" / "box.obj"
    assert assets.copy_mesh(source, destination)
    assert (destination.parent / "box.mtl").is_file()
    assert assets.missing == [str(TEXTURED_MESHES / "textures" / "wood.png")]


def test_a_collada_texture_is_copied_beside_its_mesh(tmp_path):
    texture = COLLADA_MESHES / "textures" / "oak.png"
    assets = BundledAssets(bundle_root=tmp_path)
    destination = tmp_path / "meshes" / "door.dae"
    assets.copy_mesh(COLLADA_MESHES / "door.dae", destination)
    copied = destination.parent / "textures" / "oak.png"
    assert copied.read_bytes() == texture.read_bytes()


# %% references


def test_a_file_reference_resolves_to_its_file():
    assert MeshReference(CUP_MESH.as_uri()).resolve(MESHES) == CUP_MESH


def test_a_package_reference_of_an_absent_package_resolves_to_nothing():
    reference = MeshReference("package://no_such_cramera_package/cup.obj")
    assert reference.resolve(MESHES) is None
