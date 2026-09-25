"""
An existing Blender source remains usable after separating the sample carrier.
"""

from pathlib import Path
import shutil
import subprocess

import pytest
import trimesh


# %% editable authoring source migration
def test_branding_refresh_separates_carrier_in_an_existing_source(
    tmp_path: Path,
) -> None:
    """
    Selective source refresh preserves the independently movable rack contract.
    """
    source = Path.home() / ".cramera/scenes/precision_lab/laboratory.blend"
    blender = shutil.which("blender")
    if blender is None or not source.is_file():
        pytest.skip("Blender and the editable laboratory source are unavailable")
    shutil.copyfile(source, tmp_path / source.name)
    (tmp_path / "assets").mkdir()
    builder = (
        Path(__file__).resolve().parents[2]
        / "cramera/tools/laboratory/build_blender.py"
    )
    result = subprocess.run(
        [
            blender,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            str(builder),
            "--",
            "--output",
            str(tmp_path),
            "--branding-only",
            "--skip-render",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "assets/tube_rack.glb").is_file(), result.stdout + result.stderr
    room = trimesh.load_scene(
        tmp_path / "assets/environment.glb", merge_primitives=True
    )
    rack = trimesh.load_scene(tmp_path / "assets/tube_rack.glb", merge_primitives=True)
    assert not any(
        name.startswith(("Rack", "Socket identifier", "Tube rack"))
        for name in room.graph.nodes_geometry
    )
    assert "Rack carry handle" in rack.graph.nodes_geometry
