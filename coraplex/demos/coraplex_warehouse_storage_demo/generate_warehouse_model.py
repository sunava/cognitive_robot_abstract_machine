"""
Turn a Wavefront warehouse model into the URDF and meshes the storage demo runs in.

The source model is a Blender export of a storage warehouse consisting of 28 named
objects and 31 materials, roughly two million triangles in total. It is not part of the
repository; run this script once against a local copy to regenerate
``storage_warehouse.urdf`` and ``meshes/``, both of which are committed::

    python generate_warehouse_model.py ~/Downloads/warehouse6.obj

Each object is cut into one link per material, so every link carries a single flat
colour and can be written as a binary STL without a material library. The heavy objects
are thinned to the fractions in :data:`WAREHOUSE_PARTS` to keep the scene bundle small
enough for the browser viewer, and the roof is cut away so the viewer's orbit camera can
look into the hall.

..note:: Decimation needs ``open3d``, which the demo itself does not depend on.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import open3d
import trimesh

from warehouse_layout import FITTINGS, Fitting

# %% the source model's coordinate frame

FLOOR_SURFACE_HEIGHT = 0.02
"""
Height of the walkable floor surface in the source model.

Subtracted from every vertex so the exported warehouse has its floor at ``z = 0``.
"""

ROOF_CUT_HEIGHT = 3.40
"""
Height above which the hall is cut open.

Every triangle reaching above this is dropped, which leaves the walls, the racks (whose
top deck ends at 3.28 m) and everything the robot works with, but removes the roof, its
trusses and the lamps hanging from them. The browser viewer orbits the scene, so a
closed roof would hide the whole demo from any camera above the hall. Both other
warehouse demos in this repository are open at the top for the same reason.
"""

# %% what the warehouse is made of, and how much detail each part keeps

SOURCE_MATERIALS_ARE_LINEAR = True
"""
Whether the source ``Kd`` values are linear intensities rather than display colours.

Blender writes linear values, while URDF ``rgba`` is read as a display colour, so the
values are converted before they are written out. Without the conversion the whole
warehouse comes out far too dark: the floor's ``Kd`` of 0.021 is a mid grey, not black.
"""


class DecimationStrategy(Enum):
    """
    How a part's triangles are reduced.
    """

    WHOLE_GROUP = auto()
    """
    Reduce the part's triangles as a single mesh.

    Right for one continuous surface. The walls are 268000 tiny coplanar sheet-metal
    quads, and merging across their seams is invisible.
    """

    PER_COMPONENT = auto()
    """
    Reduce every connected piece of the part on its own.

    Right for a part that is a collection of separate objects standing apart. The stored
    goods are thousands of individual boxes and reducing them as one mesh merges
    vertices across the gaps between them, which turns neat stacks into spikes.
    """


@dataclass(frozen=True)
class WarehousePart:
    """
    One named object of the source model and how it is exported.
    """

    source_name: str
    """
    Name of the object in the Wavefront file.
    """

    link_name: str
    """
    Stem of the link names the part contributes to the URDF.
    """

    detail_fraction: float = 1.0
    """
    Share of the part's triangles to keep, where ``1.0`` leaves it untouched.

    A target rather than a guarantee: :data:`MINIMUM_COMPONENT_TRIANGLES` and what the
    decimation can actually collapse both hold the result above it.
    """

    decimation: DecimationStrategy = DecimationStrategy.PER_COMPONENT
    """
    How the part's triangles are reduced.
    """


WAREHOUSE_PARTS = (
    # The building shell, one continuous surface each. The walls carry a third of the
    # whole model because the corrugated sheet is modeled rather than textured.
    WarehousePart(
        "01._rangka_Cube.007", "hall_frame", 0.29, DecimationStrategy.WHOLE_GROUP
    ),
    WarehousePart(
        "02._dinding_depan_Cube", "front_wall", 0.32, DecimationStrategy.WHOLE_GROUP
    ),
    WarehousePart(
        "03._dinding_Cube.005", "walls", 0.05, DecimationStrategy.WHOLE_GROUP
    ),
    WarehousePart("04._besi_lis_Cube.004", "wall_trim"),
    WarehousePart("05._gerbang_Gerbang", "gate", 0.29, DecimationStrategy.WHOLE_GROUP),
    WarehousePart("06._lantai_Plane", "floor"),
    WarehousePart("08._cat_garis_Plane.007", "floor_markings"),
    # The racking, kept whole: the robot places onto one of its shelves, so its deck
    # edges have to stay where warehouse_layout says they are.
    WarehousePart("07._lemari_Plane.001", "storage_racks"),
    # Stored goods, each a heap of separate items.
    WarehousePart("09._kardus_Cube.021", "cardboard_boxes", 0.15),
    WarehousePart("10._bungkus_Cube.026", "wrapped_bundles", 0.25),
    WarehousePart("11._bungkus_plastik_Cube.002", "plastic_wrapped_goods", 0.15),
    WarehousePart("12._botol_Cube.001", "bottles", 0.25),
    WarehousePart("13._gentong_Cylinder.002", "barrels", 0.25),
    WarehousePart("14._gas_Cylinder.006", "gas_cylinders", 0.20),
    WarehousePart("15._box_besi_Cube.006", "steel_boxes"),
    WarehousePart("16._kyu_2_Cube.009", "timber_stacks"),
    WarehousePart("17._box_kayu_Cube.010", "wooden_crates"),
    # Equipment. The two names the source model does not spell out are named after what
    # they measurably are: wheeled frames 1.15 m tall, and fittings 3.3 m to 4.7 m up.
    WarehousePart("18._ondo_Cube.013", "wheeled_trolleys", 0.25),
    WarehousePart("23._KLLNG_Plane.002", "ceiling_fixtures"),
    WarehousePart("19._trafo_Cube.016", "transformer"),
    WarehousePart("20._saklar_Cube.003", "switchboards"),
    WarehousePart("21._lampu_Cylinder", "ceiling_lamps", 0.25),
    WarehousePart("22._tangga_Cube.023", "stairs"),
    WarehousePart("24._AC_Cube.031", "air_conditioners", 0.20),
    WarehousePart("25._kabel_Plane.010", "cable_runs"),
    WarehousePart("26._meja_Plane.012", "tables"),
    WarehousePart("27._KERETA_Cube.039", "hand_carts", 0.25),
    WarehousePart("28._PEMADAM_Cube.041", "fire_extinguishers", 0.30),
)
"""
Every object of the source model, in the order it is written to the URDF.

Parts above :data:`ROOF_CUT_HEIGHT` are still listed: the cut removes their triangles,
not their entry, and a part left with nothing simply writes no link.
"""

MINIMUM_COMPONENT_TRIANGLES = 12
"""
Triangles every connected piece keeps, which is a closed box, so nothing can collapse
into a sliver or vanish.
"""

# %% reading the source model


@dataclass
class MaterialGroup:
    """
    The triangles of one object that share a single material, exported as one link.
    """

    part: WarehousePart
    """
    The source object the triangles belong to.
    """

    material_name: str
    """
    Name of the Wavefront material the triangles are drawn in.
    """

    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    """
    Triangles as indices into the model's shared vertex array.
    """

    @property
    def link_name(self) -> str:
        """
        :return: Name of the URDF link this group becomes.
        """
        return "%s__%s" % (self.part.link_name, sanitize(self.material_name))


def sanitize(name: str) -> str:
    """
    :param name: A name taken from the source model.
    :return: The name reduced to characters that are safe in a file name and an
        identifier.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass
class SourceModel:
    """
    A Wavefront model read into a shared vertex array and per-material triangle groups.
    """

    vertices: np.ndarray
    """
    Vertex positions, rotated to a z-up frame with the floor surface at ``z = 0``.
    """

    groups: list[MaterialGroup]
    """
    The model's triangles, grouped by object and material.
    """

    @classmethod
    def read(cls, path: str) -> SourceModel:
        """
        :param path: Path of the Wavefront ``.obj`` file to read.
        :return: The model, with objects not listed in :data:`WAREHOUSE_PARTS` dropped.
        """
        parts_by_source_name = {part.source_name: part for part in WAREHOUSE_PARTS}
        vertices: list[tuple[float, float, float]] = []
        groups: dict[tuple[str, str], MaterialGroup] = {}
        part: WarehousePart | None = None
        material_name = "default"

        with open(path) as source:
            for line in source:
                if line.startswith("v "):
                    _, x, y, z = line.split()
                    # Blender writes y-up: (x, y, z) becomes (x, -z, y) in a z-up frame.
                    vertices.append(
                        (float(x), -float(z), float(y) - FLOOR_SURFACE_HEIGHT)
                    )
                elif line.startswith("o "):
                    part = parts_by_source_name.get(line[2:].strip())
                elif line.startswith("usemtl "):
                    material_name = line[7:].strip()
                elif line.startswith("f ") and part is not None:
                    corners = [
                        int(token.split("/")[0]) - 1 for token in line[2:].split()
                    ]
                    group = groups.setdefault(
                        (part.link_name, material_name),
                        MaterialGroup(part=part, material_name=material_name),
                    )
                    # Wavefront faces may be polygons; a triangle fan is exact for the
                    # convex quads this model is built from.
                    for first, second in zip(corners[1:-1], corners[2:]):
                        triangle = (corners[0], first, second)
                        if (
                            max(vertices[corner][2] for corner in triangle)
                            > ROOF_CUT_HEIGHT
                        ):
                            continue
                        group.triangles.append(triangle)

        missing = set(parts_by_source_name) - {
            group.part.source_name for group in groups.values()
        }
        if missing:
            raise SourceObjectMissing(sorted(missing))
        return cls(
            vertices=np.array(vertices, dtype=float),
            groups=[group for group in groups.values() if group.triangles],
        )


class SourceObjectMissing(Exception):
    """
    Raised when the source model does not contain every object the URDF expects.
    """

    def __init__(self, source_names: list[str]) -> None:
        super().__init__(
            "source model is missing the objects %s" % ", ".join(source_names)
        )
        self.source_names = source_names
        """
        Names of the objects that were expected but not found.
        """


def read_material_colors(path: str) -> dict[str, tuple[float, float, float, float]]:
    """
    :param path: Path of the Wavefront ``.mtl`` file to read.
    :return: The rgba display colour of every material in the library.
    """
    colors: dict[str, tuple[float, float, float, float]] = {}
    material_name = ""
    diffuse = (0.8, 0.8, 0.8)
    alpha = 1.0
    for line in open(path):
        if line.startswith("newmtl "):
            if material_name:
                colors[material_name] = (*to_display_color(diffuse), alpha)
            material_name, diffuse, alpha = line[7:].strip(), (0.8, 0.8, 0.8), 1.0
        elif line.startswith("Kd "):
            diffuse = tuple(float(value) for value in line.split()[1:4])
        elif line.startswith("d "):
            alpha = float(line.split()[1])
    if material_name:
        colors[material_name] = (*to_display_color(diffuse), alpha)
    return colors


def to_display_color(diffuse: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    :param diffuse: A material's ``Kd`` value.
    :return: The same colour in the space URDF ``rgba`` is read in.
    """
    if not SOURCE_MATERIALS_ARE_LINEAR:
        return diffuse
    return tuple(
        round(
            (
                1.055 * channel ** (1 / 2.4) - 0.055
                if channel > 0.0031308
                else 12.92 * channel
            ),
            4,
        )
        for channel in diffuse
    )


# %% writing the meshes


def decimate(mesh: trimesh.Trimesh, keep_fraction: float) -> trimesh.Trimesh:
    """
    :param mesh: The mesh to reduce.
    :param keep_fraction: Share of the mesh's triangles to keep.
    :return: The reduced mesh, unchanged if it is already at or below the target.
    """
    target = max(MINIMUM_COMPONENT_TRIANGLES, round(len(mesh.faces) * keep_fraction))
    if len(mesh.faces) <= target:
        return mesh
    reduced = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(mesh.vertices),
        open3d.utility.Vector3iVector(mesh.faces),
    ).simplify_quadric_decimation(target_number_of_triangles=target)
    return trimesh.Trimesh(
        vertices=np.asarray(reduced.vertices),
        faces=np.asarray(reduced.triangles),
        process=False,
    )


def decimate_per_component(
    mesh: trimesh.Trimesh, keep_fraction: float
) -> trimesh.Trimesh:
    """
    :param mesh: The mesh to reduce, a collection of separate pieces.
    :param keep_fraction: Share of each piece's triangles to keep.
    :return: The pieces reduced one by one and concatenated again.
    """
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    return trimesh.util.concatenate(
        [
            decimate(mesh.submesh([component], append=True), keep_fraction)
            for component in components
        ]
    )


def write_meshes(model: SourceModel, mesh_directory: str) -> dict[str, int]:
    """
    :param model: The source model to export.
    :param mesh_directory: Directory the binary STL files are written to.
    :return: The triangle count of every link that was written.
    """
    os.makedirs(mesh_directory, exist_ok=True)
    reduce = {
        DecimationStrategy.WHOLE_GROUP: decimate,
        DecimationStrategy.PER_COMPONENT: decimate_per_component,
    }
    written: dict[str, int] = {}
    for group in model.groups:
        triangles = np.array(group.triangles, dtype=np.int64)
        used = np.unique(triangles)
        # Welding duplicate vertices is what makes the pieces of a part connected in the
        # first place; the source model repeats a position per face that touches it.
        mesh = trimesh.Trimesh(
            vertices=model.vertices[used],
            faces=np.searchsorted(used, triangles),
            process=True,
        )
        mesh = reduce[group.part.decimation](mesh, group.part.detail_fraction)
        mesh.export(os.path.join(mesh_directory, group.link_name + ".stl"))
        written[group.link_name] = len(mesh.faces)
    return written


# %% writing the URDF


def write_urdf(
    model: SourceModel,
    colors: dict[str, tuple[float, float, float, float]],
    urdf_path: str,
    mesh_directory_name: str,
) -> None:
    """
    :param model: The exported source model, one link per material group.
    :param colors: The rgba display colour of every material in the library.
    :param urdf_path: Path the URDF is written to.
    :param mesh_directory_name: Name of the mesh directory, relative to the URDF.
    """
    used_materials = sorted(
        {group.material_name for group in model.groups}
        | {fitting.material_name for fitting in FITTINGS if fitting.material_name}
    )
    lines = [
        '<?xml version="1.0"?>',
        "<!--",
        "  A storage warehouse, converted from a Wavefront model by",
        "  generate_warehouse_model.py. The building and its stored goods are visual",
        "  only; collision geometry is limited to the two surfaces the robot works on,",
        "  the target shelf and the loaded pallet, so motion planning interacts with",
        "  exactly those and nothing else. The floor surface is at z = 0.",
        "-->",
        '<robot name="storage_warehouse">',
        "",
    ]
    for material_name in used_materials:
        red, green, blue, alpha = colors[material_name]
        lines += [
            '  <material name="%s">' % sanitize(material_name),
            '    <color rgba="%s %s %s %s"/>' % (red, green, blue, alpha),
            "  </material>",
        ]
    lines += ["", '  <link name="warehouse_root"/>', ""]

    for group in model.groups:
        lines += [
            '  <link name="%s">' % group.link_name,
            "    <visual>",
            "      <geometry>",
            '        <mesh filename="%s/%s.stl"/>'
            % (mesh_directory_name, group.link_name),
            "      </geometry>",
            '      <material name="%s"/>' % sanitize(group.material_name),
            "    </visual>",
            "  </link>",
        ] + fixed_joint(group.link_name)

    for fitting in FITTINGS:
        lines += ['  <link name="%s">' % fitting.link_name]
        for tag in ("visual",) if fitting.material_name else ():
            lines += box_geometry(tag, fitting)
        lines += box_geometry("collision", fitting)
        lines += ["  </link>"] + fixed_joint(fitting.link_name)

    lines += ["</robot>", ""]
    with open(urdf_path, "w") as urdf:
        urdf.write("\n".join(lines))


def box_geometry(tag: str, fitting: Fitting) -> list[str]:
    """
    :param tag: Either ``visual`` or ``collision``.
    :param fitting: The fitting to render as a box.
    :return: The lines of the fitting's geometry element.
    """
    lines = [
        "    <%s>" % tag,
        '      <origin xyz="%s %s %s" rpy="0 0 0"/>' % fitting.center,
        "      <geometry>",
        '        <box size="%s %s %s"/>' % fitting.size,
        "      </geometry>",
    ]
    if tag == "visual":
        lines.append('      <material name="%s"/>' % sanitize(fitting.material_name))
    return lines + ["    </%s>" % tag]


def fixed_joint(link_name: str) -> list[str]:
    """
    :param link_name: The link to attach to the warehouse root.
    :return: The lines of the fixed joint attaching it.
    """
    return [
        '  <joint name="%s_joint" type="fixed">' % link_name,
        '    <parent link="warehouse_root"/>',
        '    <child link="%s"/>' % link_name,
        "  </joint>",
        "",
    ]


# %% command line entry point


def main() -> None:
    """
    Regenerate the warehouse URDF and meshes from a local copy of the source model.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="path of the Wavefront .obj source model")
    parser.add_argument(
        "--out",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="directory the URDF and the mesh directory are written to",
    )
    arguments = parser.parse_args()

    model = SourceModel.read(arguments.source)
    colors = read_material_colors(os.path.splitext(arguments.source)[0] + ".mtl")
    written = write_meshes(model, os.path.join(arguments.out, "meshes"))
    write_urdf(
        model,
        colors,
        os.path.join(arguments.out, "storage_warehouse.urdf"),
        "meshes",
    )
    print("wrote %d links, %d triangles" % (len(written), sum(written.values())))


if __name__ == "__main__":
    main()
