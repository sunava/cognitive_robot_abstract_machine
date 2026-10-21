"""
``cramera-import`` end to end: a fixture stage in, a complete scene bundle out.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy
import pytest
from PIL import Image

pxr = pytest.importorskip("pxr")

from pxr import Usd, UsdGeom

from cramera._version import __version__
from cramera.onboard.usd_scene.bundle import BundleKey, SceneBundle, SceneCamera
from cramera.onboard.usd_scene.bundle_layout import BundleFile
from cramera.onboard.usd_scene.command_line import ImportOption, main
from cramera.onboard.usd_scene.exceptions import SelectedPrimLeftOut
from cramera.onboard.usd_scene.gltf import (
    AttributeName,
    GltfKey,
    MaterialKey,
)
from cramera.onboard.usd_scene.materials import (
    AlphaMode,
    MaterialMapper,
    OmniPhysicallyBasedInput,
    TransmissionBlending,
)
from cramera.onboard.usd_scene.robot_description import UrdfAttribute, UrdfTag
from cramera.onboard.usd_scene.scene_description import (
    ExclusionReason,
    SceneDescription,
    SkipReason,
)
from cramera.onboard.usd_scene.stage_reader import PrimExclusion, StageReader
from cramera.scene_presentation import PresentationField

from .test_usd_scene_gltf import GlbContent

DATASET = Path(__file__).parent / "dataset" / "usd_scene"
"""
Small stages, one per behaviour.
"""


def convert(stage: str, scenes: Path, name: str, *options: str) -> Path:
    """
    Run ``cramera-import`` on a fixture stage.

    :param stage: The fixture file name.
    :param scenes: The scenes directory.
    :param name: The bundle name.
    :param options: Further command line options.
    """
    main(
        [
            str(DATASET / stage),
            ImportOption.NAME,
            name,
            ImportOption.OUT,
            str(scenes),
            *options,
        ]
    )
    return scenes / name


def load(bundle: Path, file: BundleFile) -> dict:
    """
    One JSON file of a bundle.

    :param bundle: The bundle directory.
    :param file: The file.
    """
    return json.loads((bundle / file).read_text(encoding="utf-8"))


# %% scene files


class TestCabinetBundle:
    """
    The bundle of the jointed cabinet.
    """

    @pytest.fixture()
    def bundle(self, tmp_path) -> Path:
        return convert("jointed_cabinet.usda", tmp_path, "cabinet")

    def test_scene_declares_one_environment_model_keeping_its_materials(self, bundle):
        scene = load(bundle, BundleFile.SCENE)
        description = StageReader(DATASET / "jointed_cabinet.usda").read()
        assert scene[PresentationField.MODELS] == [
            {
                BundleKey.NAME: "environment",
                BundleKey.URDF: BundleFile.ENVIRONMENT,
                BundleKey.PREFIX: "",
                PresentationField.ROBOT: False,
                BundleKey.LINKS: len(description.links),
                BundleKey.MOVABLE_JOINTS: [
                    joint.name for joint in description.movable_joints
                ],
                PresentationField.PRESERVE_MATERIALS: True,
            }
        ]
        assert scene[BundleKey.SOURCE][BundleKey.CONVERTER] == __version__

    def test_rendering_turns_off_the_viewer_floor_and_points_at_the_lighting(
        self, bundle
    ):
        rendering = load(bundle, BundleFile.SCENE)[PresentationField.RENDERING]
        assert rendering[BundleKey.GROUND_PLANE] is False
        assert rendering[BundleKey.AMBIENT_OCCLUSION] is False
        assert rendering[PresentationField.ENVIRONMENT] == BundleFile.LIGHTING
        assert (bundle / rendering[PresentationField.ENVIRONMENT]).is_file()

    def test_rendering_keeps_the_default_exposure_and_shadows_the_dense_core(
        self, bundle
    ):
        rendering = load(bundle, BundleFile.SCENE)[PresentationField.RENDERING]
        description = StageReader(DATASET / "jointed_cabinet.usda").read()
        assert rendering[BundleKey.EXPOSURE] == SceneBundle.exposure
        assert (
            rendering[BundleKey.SHADOW_BOUNDS] == description.central_bounds.to_json()
        )

    def test_static_trajectory_holds_the_initial_joint_values(self, bundle):
        trajectory = load(bundle, BundleFile.TRAJECTORY)
        description = StageReader(DATASET / "jointed_cabinet.usda").read()
        assert trajectory[BundleKey.FRAMES] == [
            {joint.name: joint.initial_value for joint in description.movable_joints}
        ]
        assert (
            len(trajectory[BundleKey.BASE])
            == len(trajectory[PresentationField.OBJECTS])
            == 1
        )

    def test_urdf_links_reference_existing_meshes(self, bundle):
        robot = ElementTree.parse(bundle / BundleFile.ENVIRONMENT).getroot()
        files = [mesh.get(UrdfAttribute.FILENAME) for mesh in robot.iter(UrdfTag.MESH)]
        assert files and all((bundle / file).is_file() for file in files)
        assert sorted(bundle.joinpath(BundleFile.MESHES).glob("*.glb")) == sorted(
            bundle / file for file in files
        )

    def test_index_lists_the_scene(self, bundle):
        index = load(bundle.parent, BundleFile.INDEX)
        assert [entry["name"] for entry in index["scenes"]] == ["cabinet"]
        assert index["default"] == "cabinet"

    def test_report_lists_skipped_joints(self, bundle):
        report = load(bundle, BundleFile.REPORT)
        assert report[BundleKey.SKIPPED_EXAMPLES][SkipReason.LOOP_CLOSING_JOINT] == [
            "/World/Cabinet/Joints/DrawerLock"
        ]


def test_reconversion_removes_meshes_of_vanished_links(tmp_path):
    stale = tmp_path / "quad" / BundleFile.MESHES / "Gone.glb"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"glTF")
    convert("textured_quad.usda", tmp_path, "quad")
    assert not stale.exists()


def test_camera_options_set_the_initial_view(tmp_path):
    bundle = convert(
        "glass_pane.usda",
        tmp_path,
        "pane",
        ImportOption.CAMERA_POSITION,
        "1",
        "-2",
        "1.5",
        ImportOption.CAMERA_TARGET,
        "0",
        "0",
        "0.5",
        ImportOption.CAMERA_FIELD_OF_VIEW,
        "40",
    )
    camera = load(bundle, BundleFile.SCENE)[PresentationField.CAMERA]
    assert camera[BundleKey.POSITION] == [1.0, -2.0, 1.5]
    assert camera[BundleKey.TARGET] == [0.0, 0.0, 0.5]
    assert camera[BundleKey.FIELD_OF_VIEW] == 40.0
    description = StageReader(DATASET / "glass_pane.usda").read()
    framing = SceneCamera.framing(description.central_bounds, description.bounds)
    assert camera[BundleKey.FAR] == round(framing.far, 2)


@pytest.mark.parametrize(
    "options",
    [
        [ImportOption.CAMERA_POSITION, "1", "-2", "1.5"],
        [ImportOption.CAMERA_TARGET, "0", "0", "0.5"],
        [ImportOption.CAMERA_FIELD_OF_VIEW, "40"],
    ],
)
def test_camera_options_without_position_and_target_are_refused(tmp_path, options):
    with pytest.raises(SystemExit):
        convert("glass_pane.usda", tmp_path, "pane", *options)
    assert not (tmp_path / "pane").exists()


def test_exposure_option_sets_the_tone_mapping_exposure(tmp_path):
    bundle = convert("glass_pane.usda", tmp_path, "pane", ImportOption.EXPOSURE, "1.3")
    rendering = load(bundle, BundleFile.SCENE)[PresentationField.RENDERING]
    assert rendering[BundleKey.EXPOSURE] == 1.3


def test_first_authored_camera_gives_the_initial_view(tmp_path):
    bundle = convert("lit_room.usda", tmp_path, "room")
    camera = load(bundle, BundleFile.SCENE)[PresentationField.CAMERA]
    stage = Usd.Stage.Open(str(DATASET / "lit_room.usda"))
    authored = UsdGeom.Camera(stage.GetPrimAtPath("/World/Overview"))
    position = authored.GetLocalTransformation().ExtractTranslation()
    vertical = numpy.degrees(
        2.0
        * numpy.arctan(
            authored.GetVerticalApertureAttr().Get()
            / 2.0
            / authored.GetFocalLengthAttr().Get()
        )
    )
    assert camera[BundleKey.POSITION] == pytest.approx(list(position))
    assert camera[BundleKey.FIELD_OF_VIEW] == pytest.approx(vertical, abs=1e-3)
    looking = numpy.array(camera[BundleKey.TARGET]) - numpy.array(position)
    assert looking / numpy.linalg.norm(looking) == pytest.approx(
        [0.0, 1.0, 0.0], abs=1e-6
    )


# %% robots and excluded prims


def test_report_lists_the_robots_left_out_with_the_mark_found(tmp_path):
    bundle = convert("robot_markers.usda", tmp_path, "cell")
    assert load(bundle, BundleFile.REPORT)[BundleKey.EXCLUDED] == [
        {
            BundleKey.PATH: "/World/Robots/Rover",
            BundleKey.REASON: ExclusionReason.ISAAC_ROBOT_SCHEMA,
        },
        {
            BundleKey.PATH: "/World/Robots/Walker",
            BundleKey.REASON: ExclusionReason.ROBOT_ENTITY_TYPE,
        },
    ]


def test_urdf_joints_connect_the_links_it_declares(tmp_path):
    bundle = convert("robot_markers.usda", tmp_path, "cell")
    robot = ElementTree.parse(bundle / BundleFile.ENVIRONMENT).getroot()
    links = {link.get(UrdfAttribute.NAME) for link in robot.iter(UrdfTag.LINK)}
    connected = {
        element.get(UrdfAttribute.LINK)
        for joint in robot.iter(UrdfTag.JOINT)
        for element in [joint.find(UrdfTag.PARENT), joint.find(UrdfTag.CHILD)]
    }
    assert connected <= links
    assert len(links) == len(StageReader(DATASET / "robot_markers.usda").read().links)


def test_exclude_option_leaves_out_a_subtree(tmp_path):
    bundle = convert(
        "robot_markers.usda", tmp_path, "cell", ImportOption.EXCLUDE, "/World/Robots"
    )
    assert load(bundle, BundleFile.REPORT)[BundleKey.EXCLUDED] == [
        {BundleKey.PATH: "/World/Robots", BundleKey.REASON: ExclusionReason.REQUESTED}
    ]
    assert list(load(bundle, BundleFile.SEMANTICS)[BundleKey.PRIMS]) == ["/World/Cell"]


def test_keep_robots_option_keeps_the_marked_robots_in_the_environment(tmp_path):
    bundle = convert("robot_markers.usda", tmp_path, "cell", ImportOption.KEEP_ROBOTS)
    kept = StageReader(
        DATASET / "robot_markers.usda", exclusion=PrimExclusion(keep_robots=True)
    ).read()
    assert load(bundle, BundleFile.REPORT)[BundleKey.EXCLUDED] == []
    [model] = load(bundle, BundleFile.SCENE)[PresentationField.MODELS]
    assert model[BundleKey.LINKS] == len(kept.links)


def test_robot_option_keeps_its_robot_as_a_model_of_its_own(tmp_path):
    bundle = convert(
        "robot_markers.usda",
        tmp_path,
        "cell",
        ImportOption.ROBOT,
        "/World/Robots/Rover",
    )
    models = load(bundle, BundleFile.SCENE)[PresentationField.MODELS]
    assert [model[PresentationField.ROBOT] for model in models] == [False, True]
    assert load(bundle, BundleFile.REPORT)[BundleKey.EXCLUDED] == [
        {
            BundleKey.PATH: "/World/Robots/Walker",
            BundleKey.REASON: ExclusionReason.ROBOT_ENTITY_TYPE,
        }
    ]


def test_prim_option_inside_a_robot_writes_no_bundle(tmp_path):
    with pytest.raises(SelectedPrimLeftOut):
        convert(
            "robot_markers.usda",
            tmp_path,
            "rover",
            ImportOption.PRIM,
            "/World/Robots/Rover/Chassis",
        )
    assert list(tmp_path.iterdir()) == []


def test_report_lists_the_bodies_detached_from_left_out_robots(tmp_path):
    bundle = convert("detached_gripper.usda", tmp_path, "gripper")
    description = StageReader(DATASET / "detached_gripper.usda").read()
    assert load(bundle, BundleFile.REPORT)[BundleKey.DETACHED] == [
        {
            BundleKey.BODY: detached.body,
            BundleKey.PARENT_BODY: detached.parent_body,
            BundleKey.JOINT: detached.joint,
            BundleKey.MOVABLE_JOINTS: detached.movable_joints,
        }
        for detached in description.report.detached
    ]


def test_report_lists_the_articulations_kept_in_the_environment(tmp_path):
    bundle = convert("unmarked_articulation.usda", tmp_path, "articulations")
    description = StageReader(DATASET / "unmarked_articulation.usda").read()
    assert load(bundle, BundleFile.REPORT)[BundleKey.ARTICULATIONS] == [
        {
            BundleKey.PATH: articulation.path,
            BundleKey.MOVABLE_JOINTS: articulation.movable_joints,
        }
        for articulation in description.report.articulations
    ]


def printed_summary(
    stage: str, scenes: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[SceneDescription, str]:
    """
    The description of a fixture stage and what ``cramera-import`` printed for it.

    :param stage: The fixture file name.
    :param scenes: The scenes directory.
    :param capsys: The captured output.
    """
    convert(stage, scenes, Path(stage).stem)
    return StageReader(DATASET / stage).read(), capsys.readouterr().out


def test_summary_names_every_detached_body_and_its_joint(tmp_path, capsys):
    description, printed = printed_summary("detached_gripper.usda", tmp_path, capsys)
    assert all(
        detached.body in printed and detached.joint in printed
        for detached in description.report.detached
    )


def test_summary_names_every_articulation_kept_in_the_environment(tmp_path, capsys):
    description, printed = printed_summary(
        "unmarked_articulation.usda", tmp_path, capsys
    )
    assert all(
        articulation.path in printed
        for articulation in description.report.articulations
    )


def test_summary_prints_the_warnings(tmp_path, capsys):
    description, printed = printed_summary("spherical_joint.usda", tmp_path, capsys)
    assert len(description.report.warnings) == 1
    assert all(warning in printed for warning in description.report.warnings)


# %% meshes


def test_textured_quad_glb_flips_v_and_packs_roughness_with_metalness(tmp_path):
    bundle = convert("textured_quad.usda", tmp_path, "quad")
    stage = Usd.Stage.Open(str(DATASET / "textured_quad.usda"))
    authored = numpy.asarray(
        UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Decal")).GetPrimvar("st").Get()
    )
    (glb,) = bundle.joinpath(BundleFile.MESHES).glob("*.glb")
    content = GlbContent.of_bytes(glb.read_bytes())
    primitive = content.primitive(0)
    coordinates = content.accessor(
        primitive[GltfKey.ATTRIBUTES][AttributeName.TEXTURE_COORDINATES]
    )
    assert sorted(map(tuple, coordinates)) == sorted(
        (first, 1.0 - second) for first, second in authored
    )
    material = content.document[GltfKey.MATERIALS][primitive[GltfKey.MATERIAL]]
    texture = content.document[GltfKey.TEXTURES][
        material[MaterialKey.METALLIC_ROUGHNESS][
            MaterialKey.METALLIC_ROUGHNESS_TEXTURE
        ][GltfKey.INDEX]
    ]
    packed = numpy.asarray(
        Image.open(io.BytesIO(content.embedded(texture[GltfKey.SOURCE])))
    )
    assert numpy.array_equal(
        packed[..., 1],
        numpy.asarray(Image.open(DATASET / "textures" / "roughness.png")),
    )
    assert numpy.array_equal(
        packed[..., 2], numpy.asarray(Image.open(DATASET / "textures" / "metallic.png"))
    )


def test_glass_glb_is_blended_without_transmission_extension(tmp_path):
    bundle = convert("glass_pane.usda", tmp_path, "pane")
    description = StageReader(DATASET / "glass_pane.usda").read()
    pane = next(link for link in description.links if link.path == "/World/Pane")
    glass = MaterialMapper().map(description.materials["/World/Looks/Glass"])
    drawn = TransmissionBlending().approximate(glass)
    glb = bundle / BundleFile.MESHES / f"{pane.name}.glb"
    materials = GlbContent.of_bytes(glb.read_bytes()).document[GltfKey.MATERIALS]
    written = next(
        material for material in materials if material[GltfKey.NAME] == glass.name
    )
    assert written[MaterialKey.METALLIC_ROUGHNESS][
        MaterialKey.BASE_COLOR_FACTOR
    ] == list(drawn.base_color_factor)
    assert written[MaterialKey.ALPHA_MODE] == AlphaMode.BLEND
    assert written[MaterialKey.DOUBLE_SIDED] is True
    assert written[GltfKey.EXTRAS] == {
        MaterialKey.TRANSMISSION_FACTOR: glass.transmission,
        MaterialKey.INDEX_OF_REFRACTION: glass.index_of_refraction,
        MaterialKey.BASE_COLOR_FACTOR: list(glass.base_color_factor),
        MaterialKey.METALLIC_FACTOR: glass.metallic_factor,
    }
    assert GltfKey.EXTENSIONS not in written


def test_unmapped_material_inputs_are_reported(tmp_path):
    bundle = convert("textured_quad.usda", tmp_path, "quad")
    report = load(bundle, BundleFile.REPORT)
    assert report[BundleKey.UNMAPPED_INPUTS] == {
        "/World/Looks/DecalPaint": [OmniPhysicallyBasedInput.ALBEDO_BRIGHTNESS]
    }


# %% sidecars


def test_semantics_name_each_prims_link(tmp_path):
    bundle = convert("textured_quad.usda", tmp_path, "quad")
    semantics = load(bundle, BundleFile.SEMANTICS)
    description = StageReader(DATASET / "textured_quad.usda").read()
    (decal,) = [link for link in description.links if link.meshes]
    assert semantics[BundleKey.PRIMS]["/World/Decal"] == {
        BundleKey.LINK: decal.name,
        "sdt": {"class": "AprilTag", "tagId": 7},
    }
    assert (semantics[BundleKey.UNITS], semantics[BundleKey.UP_AXIS]) == ("metres", "Z")


def test_lighting_describes_sun_dome_and_panels(tmp_path):
    bundle = convert("lit_room.usda", tmp_path, "room")
    lighting = load(bundle, BundleFile.LIGHTING)
    description = StageReader(DATASET / "lit_room.usda").read()
    (panel,) = lighting[BundleKey.PANELS]
    assert panel[BundleKey.NORMAL] == [0.0, 0.0, -1.0]
    assert panel[BundleKey.CENTER] == [0.5, 0.0, 3.0]
    sun = next(light for light in description.lights if light.path == "/World/Sun")
    assert lighting[BundleKey.SUN][BundleKey.DIRECTION] == pytest.approx(
        list(sun.emission_direction), abs=1e-4
    )
    assert lighting[BundleKey.DOME][BundleKey.INTENSITY] == 2200.0
