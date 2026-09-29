"""
Verify usable interaction geometry in independent analysis instruments.
"""

import importlib
from math import cos, hypot, isclose, sin
from pathlib import Path
from types import ModuleType

import pytest

# %% definition loading


@pytest.fixture
def analysis_devices(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """
    Load independent instruments through the standalone authoring import path.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "cramera/tools/laboratory/instruments"
    )
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("analysis_devices")


@pytest.fixture
def specification(analysis_devices: ModuleType) -> ModuleType:
    """
    Load the same specification enums used by the instrument definitions.
    """
    return importlib.import_module("spec")


# %% optical alignment


def test_microscope_sample_travel_reaches_the_selected_objective(
    analysis_devices: ModuleType,
) -> None:
    """
    The default objective axis passes through the center of the sample stage.
    """
    microscope = analysis_devices.build_digital_microscope()
    parts = {part.name: part for part in microscope.parts}
    frames = {frame.name: frame for frame in microscope.frames}
    stage_frame = frames["sample_stage"]
    objective_frame = frames["objective_axis"]
    stage_position = tuple(
        stage_frame.position[coordinate]
        + parts["focus_carriage"].origin[coordinate]
        + parts["slide_x"].origin[coordinate]
        + parts["slide_y"].origin[coordinate]
        for coordinate in range(3)
    )
    objective_position = tuple(
        objective_frame.position[coordinate] + parts["optical_head"].origin[coordinate]
        for coordinate in range(3)
    )
    assert isclose(stage_position[0], objective_position[0], abs_tol=1e-8)
    assert isclose(stage_position[1], objective_position[1], abs_tol=1e-8)
    assert stage_position[2] + parts["focus_carriage"].upper < objective_position[2]


# %% movable mechanisms and independent accessories


def test_analysis_devices_have_valid_separate_hierarchies(
    analysis_devices: ModuleType,
) -> None:
    """
    Every instrument or accessory can be exported as a self-contained tree.
    """
    devices = analysis_devices.build_devices()
    assert len({device.identifier for device in devices}) == len(devices)
    for device in devices:
        device.validate()


def test_probe_fits_the_meter_holder(
    analysis_devices: ModuleType, specification: ModuleType
) -> None:
    """
    The removable electrode grip fits through its support clamp.
    """
    meter = analysis_devices.build_ph_meter()
    probe = analysis_devices.build_ph_probe()
    holder = next(
        shape
        for part in meter.parts
        for shape in part.shapes
        if shape.name == "probe_clamp"
    )
    handle = next(
        shape
        for part in probe.parts
        for shape in part.shapes
        if shape.name == "probe_handle"
    )
    assert holder.kind == specification.ShapeKind.RING
    assert handle.dimensions[0] < holder.dimensions[1]
    assert probe.identifier in meter.accessories


def test_probe_support_arm_leaves_the_clamp_bore_clear(
    analysis_devices: ModuleType,
) -> None:
    """
    The support beam terminates at the clamp wall without covering its opening.
    """
    meter = analysis_devices.build_ph_meter()
    parts = {part.name: part for part in meter.parts}
    shapes = {shape.name: shape for shape in parts["probe_arm"].shapes}
    arm = shapes["arm"]
    clamp = shapes["probe_clamp"]
    assert arm.center[1] - arm.dimensions[1] / 2 > clamp.center[1] + clamp.dimensions[1]


def test_scanner_dock_leaves_the_handle_grasp_above_the_cradle(
    analysis_devices: ModuleType, specification: ModuleType
) -> None:
    """
    The robot can grasp a docked scanner without reaching through its cradle.
    """
    scanner = analysis_devices.build_barcode_scanner()
    dock = analysis_devices.build_barcode_dock()
    grasp = next(frame for frame in scanner.frames if frame.name == "body_grasp")
    mounting = next(frame for frame in dock.frames if frame.name == "scanner_mount")
    cradle_top = max(
        shape.center[2] + shape.dimensions[2] / 2
        for part in dock.parts
        for shape in part.shapes
        if shape.kind == specification.ShapeKind.BOX
    )
    assert grasp.position[2] + mounting.position[2] > cradle_top
    assert dock.identifier in scanner.accessories


def test_printer_media_roll_is_accessible_through_the_opening_cover(
    analysis_devices: ModuleType, specification: ModuleType
) -> None:
    """
    The articulated cover exposes a separately rotating media roll.
    """
    printer = analysis_devices.build_label_printer()
    parts = {part.name: part for part in printer.parts}
    cover = parts["media_cover"]
    spindle = parts["media_spindle"]
    assert cover.joint == specification.JointKind.REVOLUTE
    assert cover.axis == (-1, 0, 0)
    assert cover.lower < cover.open_value <= cover.upper
    assert spindle.joint == specification.JointKind.CONTINUOUS
    assert spindle.parent == cover.parent
    assert any(shape.kind == specification.ShapeKind.RING for shape in spindle.shapes)


def test_printer_output_label_is_connected_to_the_feed_web(
    analysis_devices: ModuleType,
) -> None:
    """
    The printable strip exits the slot without a floating or duplicated sheet.
    """
    printer = analysis_devices.build_label_printer()
    parts = {part.name: part for part in printer.parts}
    shapes = {shape.name: shape for shape in parts["base"].shapes}
    for upstream, downstream in (
        ("label_web", "exit_web"),
        ("exit_web", "printed_label"),
    ):
        first = shapes[upstream]
        second = shapes[downstream]
        first_end = (
            first.center[1] - first.dimensions[1] / 2 * cos(first.rotation[0]),
            first.center[2] - first.dimensions[1] / 2 * sin(first.rotation[0]),
        )
        second_end = (
            second.center[1] + second.dimensions[1] / 2 * cos(second.rotation[0]),
            second.center[2] + second.dimensions[1] / 2 * sin(second.rotation[0]),
        )
        assert first_end == pytest.approx(second_end, abs=1e-8)


def test_printer_paper_path_clears_the_front_upper_panel(
    analysis_devices: ModuleType,
) -> None:
    """
    The feed ramp finishes before the front wall and exits through its slot.
    """
    printer = analysis_devices.build_label_printer()
    parts = {part.name: part for part in printer.parts}
    shapes = {shape.name: shape for shape in parts["base"].shapes}
    front = shapes["front_upper"]
    label = shapes["printed_label"]
    ramp = shapes["exit_web"]
    assert (
        label.center[2] + label.dimensions[2] / 2
        < front.center[2] - front.dimensions[2] / 2
    )
    assert (
        ramp.center[1] - ramp.dimensions[1] / 2 * cos(ramp.rotation[0])
        > front.center[1] + front.dimensions[1] / 2
    )


def test_printer_cover_side_clears_the_rear_casing_through_its_travel(
    analysis_devices: ModuleType,
) -> None:
    """
    The rear cover edge does not sweep through the printer's fixed back wall.
    """
    printer = analysis_devices.build_label_printer()
    parts = {part.name: part for part in printer.parts}
    cover = parts["media_cover"]
    side = next(shape for shape in cover.shapes if shape.name == "cover_side")
    rear = next(shape for shape in parts["base"].shapes if shape.name == "printer_rear")
    wall_corners = [
        (
            rear.center[1] + horizontal * rear.dimensions[1] / 2,
            rear.center[2] + vertical * rear.dimensions[2] / 2,
        )
        for horizontal in (-1, 1)
        for vertical in (-1, 1)
    ]
    for sample in range(21):
        angle = cover.lower + (cover.upper - cover.lower) * sample / 20
        cover_corners = [
            (
                cover.origin[1]
                + (side.center[1] + horizontal * side.dimensions[1] / 2) * cos(angle)
                + (side.center[2] + vertical * side.dimensions[2] / 2) * sin(angle),
                cover.origin[2]
                - (side.center[1] + horizontal * side.dimensions[1] / 2) * sin(angle)
                + (side.center[2] + vertical * side.dimensions[2] / 2) * cos(angle),
            )
            for horizontal in (-1, 1)
            for vertical in (-1, 1)
        ]
        separated = False
        for axis in (
            (1, 0),
            (0, 1),
            (cos(angle), -sin(angle)),
            (sin(angle), cos(angle)),
        ):
            wall_projection = [
                point[0] * axis[0] + point[1] * axis[1] for point in wall_corners
            ]
            cover_projection = [
                point[0] * axis[0] + point[1] * axis[1] for point in cover_corners
            ]
            separated |= max(wall_projection) <= min(cover_projection) or max(
                cover_projection
            ) <= min(wall_projection)
        assert separated, f"Rear casing intersects cover at {angle} radians"


def test_printer_hinge_sleeves_rotate_clear_of_their_inner_shafts(
    analysis_devices: ModuleType,
) -> None:
    """
    The visible moving bearing has clearance around its fixed spindle.
    """
    printer = analysis_devices.build_label_printer()
    parts = {part.name: part for part in printer.parts}
    cover = parts["media_cover"]
    sleeves = [shape for shape in cover.shapes if shape.name == "hinge_sleeve"]
    shafts = [shape for shape in parts["base"].shapes if shape.name == "cover_hinge"]
    assert len(sleeves) == len(shafts) == 2
    for sleeve, shaft in zip(sleeves, shafts):
        assert sleeve.center[1:] == (0, 0)
        assert shaft.center[1:] == cover.origin[1:]
        assert shaft.dimensions[0] < sleeve.dimensions[1]
        assert (
            abs(sleeve.center[0] - shaft.center[0]) + sleeve.dimensions[2] / 2
            < shaft.dimensions[1] / 2
        )


def test_printer_hinge_brackets_stay_clear_of_the_rear_casing(
    analysis_devices: ModuleType,
) -> None:
    """
    External bearing supports retain lateral and vertical casing clearance.
    """
    printer = analysis_devices.build_label_printer()
    parts = {part.name: part for part in printer.parts}
    cover = parts["media_cover"]
    rear = next(shape for shape in parts["base"].shapes if shape.name == "printer_rear")
    floor = next(
        shape for shape in parts["base"].shapes if shape.name == "printer_floor"
    )
    for shape in cover.shapes:
        if shape.name == "hinge_strap":
            assert (
                abs(shape.center[0]) - shape.dimensions[0] / 2 > floor.dimensions[0] / 2
            )
        if shape.name == "hinge_bridge":
            for sample in range(21):
                angle = cover.lower + (cover.upper - cover.lower) * sample / 20
                bottom = min(
                    cover.origin[2]
                    - (shape.center[1] + longitudinal * shape.dimensions[1] / 2)
                    * sin(angle)
                    + (shape.center[2] + vertical * shape.dimensions[2] / 2)
                    * cos(angle)
                    for longitudinal in (-1, 1)
                    for vertical in (-1, 1)
                )
                assert bottom > rear.center[2] + rear.dimensions[2] / 2


def test_printer_hinge_straps_leave_the_inner_shaft_clear(
    analysis_devices: ModuleType,
) -> None:
    """
    A rotating support must meet the sleeve without crossing the fixed shaft.
    """
    printer = analysis_devices.build_label_printer()
    parts = {part.name: part for part in printer.parts}
    shaft = next(shape for shape in parts["base"].shapes if shape.name == "cover_hinge")
    straps = [
        shape for shape in parts["media_cover"].shapes if shape.name == "hinge_strap"
    ]
    assert len(straps) == 2
    for strap in straps:
        clearance = hypot(
            max(abs(strap.center[1]) - strap.dimensions[1] / 2, 0),
            max(abs(strap.center[2]) - strap.dimensions[2] / 2, 0),
        )
        assert clearance > shaft.dimensions[0]
