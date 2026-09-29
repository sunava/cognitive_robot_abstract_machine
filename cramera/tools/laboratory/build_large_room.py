"""
Build the expanded laboratory shell and furniture with Blender.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent / "instruments"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from build_blender import InstrumentBuilder
from large_layout import LargeLaboratoryLayout
from spec import Finish

# %% Room resource generation


def main() -> None:
    """
    Export editable architecture, embedded meshes and primitive contacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    builder = InstrumentBuilder(
        LargeLaboratoryLayout().room_definition(), options.output / "room", 64, 1400
    )
    builder.prepare()
    material = builder.materials[Finish.PALE_TEAL]
    material.diffuse_color = (0.38, 0.46, 0.45, 1)
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = material.diffuse_color
    shader.inputs["Roughness"].default_value = 0.76
    shader.inputs["Metallic"].default_value = 0
    builder.export_parts()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(options.output / "room/model.blend"))


if __name__ == "__main__":
    main()
