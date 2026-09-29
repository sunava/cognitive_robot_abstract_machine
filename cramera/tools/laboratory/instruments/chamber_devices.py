"""
Original articulated chamber instruments and their reusable sample plate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from spec import DeviceSpec, Finish, JointKind, PartSpec


# %% Shared instrument details
@dataclass(frozen=True)
class ChamberDetails:
    """
    Add small exterior features shared by the chamber instruments.
    """

    def add_feet(self, part: PartSpec, width: float, depth: float) -> None:
        """
        Place four resilient feet beneath a freestanding instrument.
        """
        for column, horizontal in enumerate((-width / 2, width / 2)):
            for row, longitudinal in enumerate((-depth / 2, depth / 2)):
                part.cylinder(
                    f"foot_{column}_{row}",
                    (horizontal, longitudinal, 0.012),
                    0.014,
                    0.024,
                    Finish.DARK,
                )

    def add_rear_vents(
        self, part: PartSpec, width: float, back: float, bottom: float, count: int
    ) -> None:
        """
        Represent a shallow rear ventilation grille without internal protrusions.
        """
        for index in range(count):
            part.box(
                f"rear_vent_{index}",
                (0, back, bottom + index * 0.009),
                (width, 0.001, 0.003),
                Finish.DARK,
                bevel=0.001,
                collision=False,
            )

    def add_screen(
        self,
        part: PartSpec,
        center: tuple[float, float, float],
        width: float,
        height: float,
        first_line: str,
        second_line: str,
    ) -> None:
        """
        Place an inset display with a bezel and two readable front-facing lines.
        """
        horizontal, longitudinal, vertical = center
        part.box(
            "display_bezel", center, (width + 0.008, 0.004, height + 0.008), Finish.TEAL
        )
        part.box(
            "display_glass",
            (horizontal, longitudinal - 0.0025, vertical),
            (width, 0.0015, height),
            Finish.SCREEN,
            bevel=0.001,
            collision=False,
        )
        part.label(
            first_line,
            (horizontal, longitudinal - 0.0035, vertical + height * 0.18),
            height * 0.19,
            Finish.PALE_TEAL,
            width=width * 0.9,
        )
        part.label(
            second_line,
            (horizontal, longitudinal - 0.0035, vertical - height * 0.20),
            height * 0.19,
            Finish.PAPER,
            width=width * 0.9,
        )


# %% Tube capper
@dataclass(frozen=True)
class TubeCapper:
    """
    Define an open tube-handling station with a movable rotary chuck.
    """

    def build(self) -> DeviceSpec:
        """
        Construct accessible tube supports, opposing jaws and the rotary head.
        """
        device = DeviceSpec(
            "tube_capper",
            "Tube Capper / Decapper",
            "Articulated tube holder, opposing clamp jaws and a translating rotary chuck.",
        )
        base = device.part("base", mass=7.2)
        details = ChamberDetails()
        details.add_feet(base, 0.245, 0.205)
        base.box(
            "cast_base", (0, 0, 0.047), (0.30, 0.28, 0.05), Finish.WHITE, bevel=0.012
        )
        base.box(
            "lower_trim", (0, 0, 0.025), (0.294, 0.274, 0.010), Finish.TEAL, bevel=0.005
        )
        base.box(
            "rear_column",
            (0, 0.090, 0.226),
            (0.130, 0.072, 0.31),
            Finish.WHITE,
            bevel=0.012,
        )
        base.box(
            "column_face", (0, 0.0525, 0.237), (0.081, 0.004, 0.244), Finish.PALE_TEAL
        )
        for index, horizontal in enumerate((-0.028, 0.028)):
            base.cylinder(
                f"guide_{index}",
                (horizontal, 0.044, 0.239),
                0.0055,
                0.233,
                Finish.STEEL,
            )
        base.logo((0, -0.141, 0.049), 0.065)
        base.label("TUBE CAPPER", (0, 0.050, 0.373), 0.010, width=0.098)
        base.cylinder("sample_support", (0, -0.060, 0.079), 0.027, 0.012, Finish.STEEL)
        base.ring(
            "sample_centering", (0, -0.060, 0.088), 0.024, 0.015, 0.006, Finish.DARK
        )
        base.box(
            "clamp_support",
            (0, -0.010, 0.112),
            (0.101, 0.020, 0.043),
            Finish.TEAL,
            bevel=0.004,
        )
        details.add_screen(base, (0.088, -0.141, 0.048), 0.057, 0.024, "READY", "OPEN")
        base.cylinder(
            "status_indicator",
            (-0.114, -0.142, 0.05),
            0.004,
            0.003,
            Finish.BLUE,
            rotation=(pi / 2, 0, 0),
            collision=False,
        )
        for suffix, horizontal, direction in (
            ("left", -0.027, 1),
            ("right", 0.027, -1),
        ):
            jaw = device.part(
                f"clamp_{suffix}",
                parent="base",
                origin=(horizontal, -0.060, 0.115),
                joint=JointKind.PRISMATIC,
                axis=(direction, 0, 0),
                lower=0,
                upper=0.014,
                open_value=0,
                mass=0.06,
            )
            jaw.box(
                "jaw_body", (0, 0, 0), (0.016, 0.032, 0.034), Finish.STEEL, bevel=0.003
            )
            jaw.box(
                "slide_stem",
                (0, 0.025, 0),
                (0.012, 0.038, 0.014),
                Finish.STEEL,
                bevel=0.002,
            )
            jaw.box(
                "compliant_pad",
                (direction * 0.008, 0, 0),
                (0.003, 0.026, 0.025),
                Finish.DARK,
                bevel=0.001,
            )
            device.frame(
                f"{suffix}_clamp_contact",
                f"clamp_{suffix}",
                (direction * 0.010, 0, 0),
                "contact",
            )
        head = device.part(
            "head_carriage",
            parent="base",
            origin=(0, 0, 0.330),
            joint=JointKind.PRISMATIC,
            axis=(0, 0, -1),
            lower=0,
            upper=0.13,
            open_value=0,
            mass=0.85,
        )
        head.box(
            "head_bridge",
            (0, -0.013, 0.012),
            (0.117, 0.147, 0.043),
            Finish.WHITE,
            bevel=0.009,
        )
        head.box("head_trim", (0, -0.013, -0.010), (0.114, 0.144, 0.009), Finish.TEAL)
        head.cylinder("motor_casing", (0, -0.060, -0.018), 0.029, 0.024, Finish.TEAL)
        head.label("ROTARY CHUCK", (0, -0.087, 0.014), 0.007, width=0.095)
        chuck = device.part(
            "rotary_chuck",
            parent="head_carriage",
            origin=(0, -0.060, -0.040),
            joint=JointKind.CONTINUOUS,
            axis=(0, 0, 1),
            open_value=pi / 3,
            mass=0.19,
        )
        chuck.ring("chuck_shell", (0, 0, 0), 0.024, 0.011, 0.025, Finish.STEEL)
        chuck.ring("grip_liner", (0, 0, -0.006), 0.013, 0.0095, 0.016, Finish.DARK)
        for index, angle in enumerate((0, 2 * pi / 3, 4 * pi / 3)):
            chuck.box(
                f"chuck_index_{index}",
                (0, 0, 0.014),
                (0.004, 0.043, 0.003),
                Finish.DARK,
                rotation=(0, 0, angle),
                collision=False,
            )
        device.frame("sample_seat", "base", (0, -0.060, 0.085), "sample")
        device.frame("chuck_contact", "rotary_chuck", (0, 0, -0.012), "tool")
        device.frame("front_access", "base", (0, -0.23, 0.20), "approach")
        device.frame("display", "base", (0.088, -0.145, 0.048), "control")
        return device


# %% Microplate reader
@dataclass(frozen=True)
class MicroplateReader:
    """
    Define an optical plate instrument with an unobstructed sliding drawer.
    """

    def build(self) -> DeviceSpec:
        """
        Construct an enclosing shell and a removable-plate drawer well.
        """
        device = DeviceSpec(
            "microplate_reader",
            "Microplate Reader",
            "Enclosed optical-reader model with an articulated drawer and a 96-well plate seat.",
        )
        device.accessories.append("microplate_96_well")
        body = device.part("base", mass=8.5)
        details = ChamberDetails()
        details.add_feet(body, 0.345, 0.295)
        body.box(
            "base_plinth",
            (0, 0, 0.047),
            (0.40, 0.35, 0.046),
            Finish.PALE_TEAL,
            bevel=0.009,
        )
        body.box("chamber_floor", (0, 0.008, 0.073), (0.38, 0.332, 0.012), Finish.DARK)
        for suffix, horizontal in (("left", -0.188), ("right", 0.188)):
            body.box(
                f"{suffix}_wall",
                (horizontal, 0, 0.157),
                (0.024, 0.35, 0.17),
                Finish.WHITE,
                bevel=0.009,
            )
        body.box(
            "rear_wall",
            (0, 0.163, 0.155),
            (0.355, 0.024, 0.168),
            Finish.WHITE,
            bevel=0.009,
        )
        body.box(
            "top_cover", (0, 0, 0.226), (0.389, 0.342, 0.036), Finish.WHITE, bevel=0.010
        )
        body.box(
            "front_upper",
            (0, -0.162, 0.181),
            (0.358, 0.026, 0.065),
            Finish.WHITE,
            bevel=0.006,
        )
        body.box("front_lower", (0, -0.164, 0.084), (0.355, 0.024, 0.020), Finish.WHITE)
        for suffix, horizontal in (("left", -0.132), ("right", 0.132)):
            body.box(
                f"front_cheek_{suffix}",
                (horizontal, -0.164, 0.121),
                (0.085, 0.024, 0.057),
                Finish.WHITE,
            )
            body.box(
                f"drawer_rail_{suffix}",
                (horizontal * 0.595, -0.005, 0.099),
                (0.008, 0.285, 0.014),
                Finish.STEEL,
            )
        body.box(
            "optical_bridge", (0, 0.003, 0.186), (0.150, 0.125, 0.022), Finish.DARK
        )
        body.logo((-0.109, -0.176, 0.191), 0.079)
        body.label("MICROPLATE READER", (-0.096, -0.176, 0.164), 0.008, width=0.138)
        details.add_screen(
            body, (0.077, -0.177, 0.182), 0.102, 0.032, "READY", "PLATE OUT"
        )
        details.add_rear_vents(body, 0.25, 0.1755, 0.118, 8)
        for index, horizontal in enumerate((0.143, 0.163)):
            body.cylinder(
                f"control_button_{index}",
                (horizontal, -0.178, 0.181),
                0.006,
                0.004,
                Finish.TEAL,
                rotation=(pi / 2, 0, 0),
            )
        drawer = device.part(
            "plate_drawer",
            parent="base",
            origin=(0, -0.035, 0.110),
            joint=JointKind.PRISMATIC,
            axis=(0, -1, 0),
            lower=0,
            upper=0.195,
            open_value=0.195,
            mass=0.35,
        )
        drawer.box(
            "drawer_floor",
            (0, 0, -0.008),
            (0.146, 0.25, 0.007),
            Finish.STEEL,
            bevel=0.002,
        )
        drawer.box(
            "plate_seat_floor",
            (0, 0, -0.003),
            (0.138, 0.097, 0.003),
            Finish.DARK,
            bevel=0.001,
        )
        for suffix, horizontal in (("left", -0.068), ("right", 0.068)):
            drawer.box(
                f"plate_side_{suffix}",
                (horizontal, 0, 0.004),
                (0.006, 0.099, 0.012),
                Finish.TEAL,
                bevel=0.001,
            )
        for suffix, longitudinal in (("front", -0.047), ("back", 0.047)):
            drawer.box(
                f"plate_end_{suffix}",
                (0, longitudinal, 0.004),
                (0.13, 0.006, 0.012),
                Finish.TEAL,
                bevel=0.001,
            )
        drawer.box(
            "drawer_front",
            (0, -0.137, 0.011),
            (0.171, 0.019, 0.052),
            Finish.TEAL,
            bevel=0.005,
        )
        drawer.box(
            "handle",
            (0, -0.150, 0.010),
            (0.091, 0.012, 0.015),
            Finish.STEEL,
            bevel=0.005,
        )
        drawer.label("PLATE", (0, -0.160, 0.010), 0.007, Finish.INK, width=0.059)
        drawer.label(
            "A1", (-0.057, -0.051, 0.011), 0.005, Finish.PAPER, rotation=(0, 0, 0)
        )
        device.frame("drawer_handle", "plate_drawer", (0, -0.158, 0.010), "handle")
        device.frame("plate_seat", "plate_drawer", (0, 0, -0.0015), "sample")
        device.frame("load_approach", "plate_drawer", (0, 0, 0.13), "approach")
        device.frame("eject_button", "base", (0.143, -0.183, 0.181), "control")
        device.frame("start_button", "base", (0.163, -0.183, 0.181), "control")
        return device


# %% Incubator
@dataclass(frozen=True)
class Incubator:
    """
    Define a benchtop incubation chamber with shelves and a glazed door.
    """

    def build(self) -> DeviceSpec:
        """
        Construct a hollow chamber, independently framed shelves and a hinged door.
        """
        device = DeviceSpec(
            "incubator",
            "Benchtop Incubator",
            "Three-shelf incubation chamber with a transparent articulated door and temperature display.",
        )
        body = device.part("base", mass=18)
        details = ChamberDetails()
        details.add_feet(body, 0.345, 0.325)
        body.box(
            "base_plinth",
            (0, 0, 0.043),
            (0.425, 0.410, 0.040),
            Finish.TEAL,
            bevel=0.008,
        )
        body.box(
            "chamber_floor",
            (0, 0, 0.068),
            (0.394, 0.386, 0.020),
            Finish.STEEL,
            bevel=0.002,
        )
        for suffix, horizontal in (("left", -0.201), ("right", 0.201)):
            body.box(
                f"{suffix}_wall",
                (horizontal, 0, 0.292),
                (0.024, 0.399, 0.460),
                Finish.WHITE,
                bevel=0.006,
            )
            body.box(
                f"{suffix}_liner",
                (horizontal * 0.932, 0.009, 0.292),
                (0.003, 0.366, 0.420),
                Finish.STEEL,
                bevel=0.001,
            )
        body.box(
            "rear_wall",
            (0, 0.190, 0.292),
            (0.389, 0.020, 0.459),
            Finish.WHITE,
            bevel=0.006,
        )
        body.box(
            "rear_liner",
            (0, 0.178, 0.292),
            (0.372, 0.003, 0.420),
            Finish.STEEL,
            bevel=0.001,
        )
        body.box(
            "control_header",
            (0, 0, 0.550),
            (0.425, 0.410, 0.069),
            Finish.WHITE,
            bevel=0.011,
        )
        body.logo((-0.121, -0.206, 0.550), 0.083)
        body.label("INCUBATOR", (-0.121, -0.206, 0.530), 0.0075, width=0.085)
        details.add_screen(
            body, (0.060, -0.207, 0.550), 0.139, 0.036, "37.0 C", "READY"
        )
        body.cylinder(
            "program_dial",
            (0.173, -0.214, 0.550),
            0.013,
            0.017,
            Finish.TEAL,
            rotation=(pi / 2, 0, 0),
        )
        for suffix, horizontal in (("left", -0.190), ("right", 0.190)):
            body.box(
                f"gasket_{suffix}",
                (horizontal, -0.204, 0.293),
                (0.008, 0.008, 0.451),
                Finish.DARK,
                bevel=0.001,
            )
        for suffix, height in (("bottom", 0.069), ("top", 0.517)):
            body.box(
                f"gasket_{suffix}",
                (0, -0.204, height),
                (0.38, 0.008, 0.008),
                Finish.DARK,
                bevel=0.001,
            )
        body.ring(
            "circulation_guard",
            (0, 0.173, 0.425),
            0.047,
            0.039,
            0.006,
            Finish.DARK,
            rotation=(pi / 2, 0, 0),
            collision=False,
        )
        for index in range(5):
            body.box(
                f"fan_guard_bar_{index}",
                (0, 0.168, 0.394 + index * 0.015),
                (0.066, 0.004, 0.003),
                Finish.STEEL,
                bevel=0.001,
                collision=False,
            )
        details.add_rear_vents(body, 0.23, 0.201, 0.092, 9)
        for level, height in enumerate((0.177, 0.305, 0.433), start=1):
            shelf = device.part(
                f"shelf_{level}", parent="base", origin=(0, -0.009, height), mass=0.4
            )
            shelf.box(
                "shelf_surface",
                (0, 0, 0),
                (0.350, 0.337, 0.006),
                Finish.STEEL,
                bevel=0.001,
            )
            shelf.box(
                "front_lip",
                (0, -0.168, 0.004),
                (0.35, 0.006, 0.009),
                Finish.STEEL,
                bevel=0.001,
            )
            for suffix, horizontal in (("left", -0.179), ("right", 0.179)):
                body.box(
                    f"shelf_{level}_{suffix}_rail",
                    (horizontal, -0.005, height - 0.006),
                    (0.014, 0.34, 0.01),
                    Finish.STEEL,
                    bevel=0.001,
                )
            for column, horizontal in enumerate((-0.105, 0, 0.105), start=1):
                for row, longitudinal in enumerate((-0.08, 0.08), start=1):
                    device.frame(
                        f"shelf_{level}_sample_{column}_{row}",
                        f"shelf_{level}",
                        (horizontal, longitudinal, 0.003),
                        "sample",
                    )
        door = device.part(
            "door",
            parent="base",
            origin=(-0.207, -0.226, 0.291),
            joint=JointKind.REVOLUTE,
            axis=(0, 0, 1),
            lower=-1.80,
            upper=0,
            open_value=-1.63,
            mass=2.6,
        )
        for suffix, horizontal in (("hinge", 0.014), ("latch", 0.399)):
            door.box(
                f"{suffix}_stile",
                (horizontal, 0, 0),
                (0.028, 0.034, 0.461),
                Finish.WHITE,
                bevel=0.006,
            )
        for suffix, height in (("bottom", -0.217), ("top", 0.217)):
            door.box(
                f"{suffix}_rail",
                (0.207, 0, height),
                (0.389, 0.034, 0.027),
                Finish.WHITE,
                bevel=0.005,
            )
        door.box(
            "glazing", (0.207, 0, 0), (0.362, 0.008, 0.417), Finish.GLASS, bevel=0.001
        )
        for suffix, height in (("bottom", -0.150), ("top", 0.150)):
            door.cylinder(f"hinge_{suffix}", (0, 0, height), 0.009, 0.05, Finish.STEEL)
        for suffix, height in (("bottom", -0.052), ("top", 0.052)):
            door.box(
                f"handle_mount_{suffix}",
                (0.386, -0.031, height),
                (0.016, 0.040, 0.018),
                Finish.TEAL,
                bevel=0.005,
            )
        door.box(
            "handle",
            (0.386, -0.050, 0),
            (0.019, 0.019, 0.125),
            Finish.TEAL,
            bevel=0.007,
        )
        device.frame("door_handle", "door", (0.386, -0.0595, 0), "handle")
        device.frame("door_hinge", "door", (0, 0, 0), "joint")
        device.frame("temperature_control", "base", (0.173, -0.225, 0.550), "control")
        device.frame("chamber_approach", "base", (0, -0.39, 0.31), "approach")
        return device


# %% Thermal cycler
@dataclass(frozen=True)
class ThermalCycler:
    """
    Define a reaction block beneath an articulated heated cover.
    """

    def build(self) -> DeviceSpec:
        """
        Construct an accessible 96-position block and a hinged heated lid.
        """
        device = DeviceSpec(
            "pcr_thermocycler",
            "PCR Thermocycler",
            "Ninety-six-position thermal block with an articulated heated cover and cycle display.",
        )
        device.accessories.append("microplate_96_well")
        body = device.part("base", mass=6.8)
        details = ChamberDetails()
        details.add_feet(body, 0.215, 0.252)
        body.box(
            "lower_housing",
            (0, 0, 0.067),
            (0.277, 0.318, 0.083),
            Finish.WHITE,
            bevel=0.013,
        )
        body.box(
            "lower_trim", (0, 0, 0.030), (0.268, 0.311, 0.012), Finish.TEAL, bevel=0.004
        )
        body.box(
            "reaction_floor",
            (0, 0.004, 0.112),
            (0.176, 0.155, 0.012),
            Finish.STEEL,
            bevel=0.002,
        )
        for suffix, horizontal in (("left", -0.112), ("right", 0.112)):
            body.box(
                f"shoulder_{suffix}",
                (horizontal, 0.018, 0.128),
                (0.053, 0.253, 0.042),
                Finish.WHITE,
                bevel=0.006,
            )
        body.box(
            "rear_shoulder",
            (0, 0.121, 0.129),
            (0.187, 0.047, 0.043),
            Finish.WHITE,
            bevel=0.006,
        )
        body.box(
            "front_console",
            (0, -0.120, 0.122),
            (0.241, 0.078, 0.032),
            Finish.WHITE,
            bevel=0.006,
        )
        body.logo((-0.080, -0.160, 0.086), 0.075)
        body.label("THERMOCYCLER", (-0.080, -0.160, 0.061), 0.007, width=0.092)
        details.add_screen(
            body, (0.055, -0.160, 0.079), 0.112, 0.033, "READY  22 C", "0 / 30 CYCLES"
        )
        details.add_rear_vents(body, 0.18, 0.1595, 0.057, 5)
        body.cylinder(
            "start_button",
            (0.113, -0.161, 0.118),
            0.007,
            0.004,
            Finish.TEAL,
            rotation=(pi / 2, 0, 0),
        )
        block = device.part(
            "reaction_block", parent="base", origin=(0, 0.004, 0.117), mass=0.55
        )
        block.box(
            "block_floor",
            (0, 0, 0.001),
            (0.126, 0.084, 0.002),
            Finish.STEEL,
            bevel=0.001,
        )
        for row in range(8):
            for column in range(12):
                horizontal = (column - 5.5) * 0.009
                longitudinal = (row - 3.5) * 0.009
                block.ring(
                    f"block_well_{row}_{column}",
                    (horizontal, longitudinal, 0.006),
                    0.0044,
                    0.0040,
                    0.008,
                    Finish.STEEL,
                )
                device.frame(
                    f"block_{chr(65 + row)}{column + 1}",
                    "reaction_block",
                    (horizontal, longitudinal, 0.010),
                    "sample",
                )
        body.label(
            "A1", (-0.070, -0.039, 0.124), 0.005, Finish.TEAL, rotation=(0, 0, 0)
        )
        lid = device.part(
            "heated_lid",
            parent="base",
            origin=(0, 0.114, 0.160),
            joint=JointKind.REVOLUTE,
            axis=(1, 0, 0),
            lower=-1.85,
            upper=0,
            open_value=-1.70,
            mass=1.25,
        )
        lid.box(
            "lid_shell",
            (0, -0.091, 0.017),
            (0.249, 0.236, 0.047),
            Finish.WHITE,
            bevel=0.013,
        )
        lid.box(
            "lid_trim",
            (0, -0.091, -0.007),
            (0.242, 0.229, 0.009),
            Finish.TEAL,
            bevel=0.005,
        )
        lid.box(
            "heated_pressure_pad",
            (0, -0.109, -0.014),
            (0.137, 0.094, 0.009),
            Finish.STEEL,
            bevel=0.003,
        )
        for suffix, horizontal in (("left", -0.091), ("right", 0.091)):
            lid.cylinder(
                f"hinge_{suffix}",
                (horizontal, 0, 0),
                0.012,
                0.045,
                Finish.STEEL,
                rotation=(0, pi / 2, 0),
            )
        lid.box(
            "lid_handle",
            (0, -0.211, 0.022),
            (0.092, 0.027, 0.024),
            Finish.TEAL,
            bevel=0.008,
        )
        lid.label("HEATED LID", (0, -0.228, 0.022), 0.008, Finish.PAPER, width=0.079)
        lid.label("PCR", (0, -0.104, 0.041), 0.022, Finish.TEAL, rotation=(0, 0, 0))
        device.frame("lid_handle", "heated_lid", (0, -0.225, 0.023), "handle")
        device.frame("lid_hinge", "heated_lid", (0, 0, 0), "joint")
        device.frame("plate_seat", "reaction_block", (0, 0, 0.010), "sample")
        device.frame("load_approach", "reaction_block", (0, 0, 0.16), "approach")
        device.frame("start_button", "base", (0.113, -0.165, 0.118), "control")
        return device


# %% Microplate accessory
@dataclass(frozen=True)
class Microplate:
    """
    Define a separate 96-well plate with open wells and a standard outer footprint.
    """

    def build(self) -> DeviceSpec:
        """
        Construct a free object with 96 individually addressable open sample wells.
        """
        device = DeviceSpec(
            "microplate_96_well",
            "96-Well Microplate",
            "Separate 127.76 by 85.48 millimetre plate with 9 millimetre well pitch and 96 open wells.",
        )
        plate = device.part("base", mass=0.045)
        plate.box(
            "bottom",
            (0, 0, 0.0006),
            (0.12776, 0.08548, 0.0012),
            Finish.WHITE,
            bevel=0.0005,
        )
        for suffix, horizontal in (("left", -0.06188), ("right", 0.06188)):
            plate.box(
                f"rim_{suffix}",
                (horizontal, 0, 0.007775),
                (0.004, 0.08548, 0.01315),
                Finish.WHITE,
                bevel=0.0006,
            )
        for suffix, longitudinal in (("front", -0.04074), ("back", 0.04074)):
            plate.box(
                f"rim_{suffix}",
                (0, longitudinal, 0.007775),
                (0.11976, 0.004, 0.01315),
                Finish.WHITE,
                bevel=0.0006,
            )
        for row in range(8):
            for column in range(12):
                horizontal = (column - 5.5) * 0.009
                longitudinal = (row - 3.5) * 0.009
                well_name = f"{chr(65 + row)}{column + 1}"
                plate.ring(
                    f"well_{well_name}",
                    (horizontal, longitudinal, 0.00725),
                    0.00395,
                    0.00335,
                    0.0123,
                    Finish.WHITE,
                )
                device.frame(
                    f"well_{well_name}",
                    "base",
                    (horizontal, longitudinal, 0.0134),
                    "sample",
                )
            plate.label(
                chr(65 + row),
                (-0.06188, (row - 3.5) * 0.009, 0.0145),
                0.0033,
                Finish.TEAL,
                rotation=(0, 0, 0),
            )
        for column in range(12):
            plate.label(
                str(column + 1),
                ((column - 5.5) * 0.009, -0.04074, 0.0145),
                0.0032,
                Finish.TEAL,
                rotation=(0, 0, 0),
            )
        plate.logo((0, -0.0428, 0.0077), 0.023)
        device.frame("left_grip", "base", (-0.064, 0, 0.008), "grasp")
        device.frame("right_grip", "base", (0.064, 0, 0.008), "grasp")
        device.frame("plate_center", "base", (0, 0, 0), "placement")
        return device


# %% Public catalogue


def build_devices() -> list[DeviceSpec]:
    """
    Return independent chamber instruments and the reusable microplate accessory.
    """
    return [
        TubeCapper().build(),
        MicroplateReader().build(),
        Incubator().build(),
        ThermalCycler().build(),
        Microplate().build(),
    ]
