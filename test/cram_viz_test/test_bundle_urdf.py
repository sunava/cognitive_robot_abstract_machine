"""
Tests for the URDF/mesh bundler used by the onboarder and the standalone CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cram_viz.onboard.bundle_urdf import (
    BundleReport,
    _copy_side_assets,
    _ref_to_relpath,
    bundle_urdf,
    resolve_uri,
)

# %% resolve_uri


class TestResolveUri:
    def test_hints_take_priority(self, tmp_path: Path):
        target = tmp_path / "hinted.stl"
        target.write_text("solid")
        assert resolve_uri("package://foo/bar.stl", hints={"package://foo/bar.stl": str(target)}) == str(target)

    def test_file_scheme(self, tmp_path: Path):
        target = tmp_path / "mesh.stl"
        target.write_text("solid")
        assert resolve_uri("file://%s" % target) == str(target)

    def test_file_scheme_missing_returns_none(self, tmp_path: Path):
        assert resolve_uri("file://%s" % (tmp_path / "missing.stl")) is None

    def test_absolute_path(self, tmp_path: Path):
        target = tmp_path / "mesh.stl"
        target.write_text("solid")
        assert resolve_uri(str(target)) == str(target)

    def test_relative_path_with_base_dir(self, tmp_path: Path):
        (tmp_path / "meshes").mkdir()
        target = tmp_path / "meshes" / "mesh.stl"
        target.write_text("solid")
        assert resolve_uri("meshes/mesh.stl", base_dir=str(tmp_path)) == str(target)

    def test_relative_path_without_base_dir_returns_none(self):
        assert resolve_uri("meshes/mesh.stl") is None

    def test_unresolvable_package_uri_returns_none(self):
        assert resolve_uri("package://does-not-exist/mesh.stl") is None


# %% _ref_to_relpath


class TestRefToRelpath:
    def test_package_uri_keeps_package_structure(self):
        assert _ref_to_relpath("package://mypkg/meshes/x.dae", None) == "mypkg/meshes/x.dae"

    def test_unresolved_local_ref_falls_back_to_basename(self):
        assert _ref_to_relpath("book.stl", None) == "_local/book.stl"

    def test_distinct_sources_with_same_basename_do_not_collide(self):
        first = _ref_to_relpath("book.stl", "/scene_a/meshes/book.stl")
        second = _ref_to_relpath("book.stl", "/scene_b/meshes/book.stl")
        assert first != second

    def test_same_resolved_source_maps_to_the_same_relpath(self):
        first = _ref_to_relpath("book.stl", "/scene_a/meshes/book.stl")
        second = _ref_to_relpath("../meshes/book.stl", "/scene_a/meshes/book.stl")
        assert first == second


# %% _copy_side_assets


class TestCopySideAssets:
    def test_dae_texture_reference_is_copied(self, tmp_path: Path):
        source_dir = tmp_path / "source"
        destination_dir = tmp_path / "out"
        source_dir.mkdir()
        (source_dir / "diffuse.png").write_bytes(b"\x89PNG")
        (source_dir / "model.dae").write_text('<image><init_from>diffuse.png</init_from></image>')

        copied, missing = {}, []
        _copy_side_assets(
            str(source_dir / "model.dae"), str(destination_dir / "model.dae"), copied, missing
        )

        assert (destination_dir / "diffuse.png").is_file()
        assert not missing

    def test_obj_mtl_and_its_texture_are_copied(self, tmp_path: Path):
        source_dir = tmp_path / "source"
        destination_dir = tmp_path / "out"
        source_dir.mkdir()
        (source_dir / "diffuse.jpg").write_bytes(b"\xff\xd8")
        (source_dir / "model.mtl").write_text("newmtl m\nmap_Kd diffuse.jpg\n")
        (source_dir / "model.obj").write_text("mtllib model.mtl\n")

        copied, missing = {}, []
        _copy_side_assets(
            str(source_dir / "model.obj"), str(destination_dir / "model.obj"), copied, missing
        )

        assert (destination_dir / "model.mtl").is_file()
        assert (destination_dir / "diffuse.jpg").is_file()
        assert not missing


# %% bundle_urdf


class TestBundleUrdf:
    def _write_source_urdf(self, tmp_path: Path) -> Path:
        meshes = tmp_path / "source_meshes"
        meshes.mkdir()
        (meshes / "book.stl").write_text("solid book\nendsolid book\n")
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot name="r">'
            '<link name="base_link"/>'
            '<link name="book_link"/>'
            '<joint name="fixed_joint" type="fixed">'
            '<parent link="base_link"/><child link="book_link"/></joint>'
            '<joint name="hinge" type="revolute">'
            '<parent link="book_link"/><child link="base_link"/></joint>'
            '<visual><geometry><mesh filename="source_meshes/book.stl"/></geometry></visual>'
            "</robot>"
        )
        return urdf

    def test_bundles_links_joints_and_meshes(self, tmp_path: Path):
        urdf = self._write_source_urdf(tmp_path)
        out_dir = tmp_path / "out"

        report = bundle_urdf(str(urdf), "r", str(out_dir))

        assert isinstance(report, BundleReport)
        assert report.links == ["base_link", "book_link"]
        assert report.joints == ["fixed_joint", "hinge"]
        assert report.movable_joints == ["hinge"]
        assert report.meshes_copied == 1
        assert report.missing == []
        assert Path(report.urdf).is_file()
        assert (out_dir / "meshes" / "_local").is_dir()

    def test_missing_mesh_is_reported_not_raised(self, tmp_path: Path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot name="r"><link name="base_link"/>'
            '<visual><geometry><mesh filename="does_not_exist.stl"/></geometry></visual>'
            "</robot>"
        )
        out_dir = tmp_path / "out"

        report = bundle_urdf(str(urdf), "r", str(out_dir))

        assert report.meshes_copied == 0
        assert report.missing == ["<unresolved>"]

    def test_missing_source_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            bundle_urdf(str(tmp_path / "nope.urdf"), "r", str(tmp_path / "out"))
