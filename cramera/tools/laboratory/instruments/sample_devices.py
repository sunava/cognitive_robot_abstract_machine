"""
Original articulated instruments for sample preparation and liquid handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from spec import (
    DeviceSpec,
    Finish,
    JointKind,
    PartSpec,
)


# %% shared physical details
@dataclass(frozen=True)
class BenchSupports:
    """
    Place four elastomer feet beneath a rectangular instrument.
    """

    half_width: float
    """
    Lateral distance between the origin and each foot.
    """

    half_depth: float
    """
    Longitudinal distance between the origin and each foot.
    """

    radius: float = 0.010
    """
    Outer radius of each circular foot.
    """

    def add_to(self, part: PartSpec) -> None:
        """
        Add pads whose lower surface defines the bench contact plane.
        """
        for column, horizontal in enumerate((-self.half_width, self.half_width)):
            for row, longitudinal in enumerate((-self.half_depth, self.half_depth)):
                part.cylinder(
                    f"foot_{column}_{row}",
                    (horizontal, longitudinal, 0.004),
                    self.radius,
                    0.008,
                    Finish.DARK,
                )


@dataclass(frozen=True)
class WellGrid:
    """
    A rectangular array of genuine open cylindrical receptacles.
    """

    columns: int
    """
    Number of receptacles across the instrument.
    """

    rows: int
    """
    Number of receptacles along the instrument.
    """

    pitch: float
    """
    Distance between adjacent receptacle centers.
    """

    outer_radius: float
    """
    External radius of a receptacle sleeve.
    """

    inner_radius: float
    """
    Clear radius available for a sample or pipette tip.
    """

    depth: float
    """
    Vertical extent of each sleeve.
    """

    bottom: float
    """
    Height of the sleeve's bottom face in its parent frame.
    """

    def add_to(
        self, device: DeviceSpec, parent: PartSpec, material: Finish, role: str
    ) -> None:
        """
        Create hollow sleeves, connecting ribs, labels, and insertion frames.
        """
        half_width = self.columns * self.pitch / 2
        half_depth = self.rows * self.pitch / 2
        center_height = self.bottom + self.depth / 2
        top = self.bottom + self.depth
        for column in range(self.columns + 1):
            parent.box(
                f"column_rib_{column}",
                (-half_width + column * self.pitch, 0, center_height),
                (0.002, self.rows * self.pitch + 0.002, self.depth),
                material,
                bevel=0.0004,
            )
        for row in range(self.rows + 1):
            parent.box(
                f"row_rib_{row}",
                (0, -half_depth + row * self.pitch, center_height),
                (self.columns * self.pitch + 0.002, 0.002, self.depth),
                material,
                bevel=0.0004,
            )
        for row in range(self.rows):
            longitudinal = (row - (self.rows - 1) / 2) * self.pitch
            row_letter = chr(ord("A") + row)
            parent.label(
                row_letter,
                (-half_width - 0.004, longitudinal, top + 0.0004),
                0.0035,
                rotation=(0, 0, 0),
            )
            for column in range(self.columns):
                horizontal = (column - (self.columns - 1) / 2) * self.pitch
                socket = f"{row_letter}{column + 1}"
                parent.ring(
                    f"well_{socket}",
                    (horizontal, longitudinal, center_height),
                    self.outer_radius,
                    self.inner_radius,
                    self.depth,
                    material,
                )
                device.frame(
                    f"slot_{socket}",
                    parent.name,
                    (horizontal, longitudinal, top),
                    role,
                )
        for column in range(self.columns):
            horizontal = (column - (self.columns - 1) / 2) * self.pitch
            parent.label(
                str(column + 1),
                (horizontal, -half_depth - 0.004, top + 0.0004),
                0.0035,
                rotation=(0, 0, 0),
            )


# %% vortex mixer
@dataclass(frozen=True)
class VortexMixer:
    """
    A compact contact-operated mixer with an articulated sprung cup.
    """

    def build(self) -> DeviceSpec:
        """
        Define the enclosure, compliant contact head, and physical controls.
        """
        device = DeviceSpec(
            "vortex_mixer",
            "Vortex Mixer",
            "Bench vortex mixer with a spring-loaded contact cup and speed control.",
        )
        body = device.part("body", mass=1.4)
        BenchSupports(0.057, 0.061).add_to(body)
        body.box("base", (0, 0, 0.014), (0.150, 0.160, 0.018), Finish.TEAL, 0.007)
        body.box("lower_seam", (0, 0, 0.026), (0.147, 0.156, 0.003), Finish.DARK, 0.007)
        body.box("housing", (0, 0, 0.060), (0.148, 0.157, 0.067), Finish.WHITE, 0.020)
        body.cylinder("neck", (0, 0.006, 0.097), 0.025, 0.020, Finish.STEEL)
        for index in range(3):
            body.ring(
                f"cup_suspension_{index}",
                (0, 0.006, 0.093 + index * 0.005),
                0.029,
                0.023,
                0.0025,
                Finish.DARK,
            )
        body.logo((-0.038, -0.0786, 0.066), 0.044)
        body.label("VORTEX", (-0.040, -0.079, 0.049), 0.004, width=0.045)
        body.box(
            "contact_switch",
            (-0.036, -0.080, 0.029),
            (0.018, 0.006, 0.009),
            Finish.TEAL,
            0.002,
        )
        body.label("TOUCH", (-0.036, -0.0832, 0.030), 0.0025, Finish.PAPER)
        body.cylinder(
            "status_light",
            (-0.007, -0.080, 0.030),
            0.0022,
            0.003,
            Finish.BLUE,
            rotation=(pi / 2, 0, 0),
            collision=False,
        )
        for index in range(5):
            body.box(
                f"vent_{index}",
                (0.074, 0.020 + index * 0.008, 0.051),
                (0.001, 0.003, 0.022),
                Finish.DARK,
                0.0005,
                collision=False,
            )
        head = device.part(
            "contact_head",
            "body",
            origin=(0, 0.006, 0.110),
            joint=JointKind.PRISMATIC,
            axis=(0, 0, 1),
            lower=-0.006,
            upper=0,
            open_value=-0.005,
            mass=0.06,
        )
        head.cylinder("cup_support", (0, 0, -0.004), 0.032, 0.008, Finish.DARK)
        head.ring("rubber_cup", (0, 0, 0.002), 0.034, 0.020, 0.012, Finish.DARK)
        head.cylinder("cup_contact", (0, 0, -0.003), 0.020, 0.004, Finish.DARK)
        knob = device.part(
            "speed_control",
            "body",
            origin=(0.036, -0.083, 0.050),
            joint=JointKind.REVOLUTE,
            axis=(0, -1, 0),
            lower=0,
            upper=1.5 * pi,
            open_value=0.9 * pi,
            mass=0.012,
        )
        knob.cylinder(
            "knob", (0, -0.004, 0), 0.015, 0.012, Finish.TEAL, rotation=(pi / 2, 0, 0)
        )
        knob.box(
            "pointer",
            (0, -0.0105, 0.009),
            (0.002, 0.001, 0.005),
            Finish.PAPER,
            0.0004,
            collision=False,
        )
        device.frame("sample_contact", "contact_head", (0, 0, -0.001), "sample_contact")
        device.frame("speed_control_grasp", "speed_control", (0, -0.011, 0), "control")
        device.frame("body_grasp", "body", (0, 0, 0.057), "grasp")
        return device


# %% electronic pipette
@dataclass(frozen=True)
class ElectronicPipette:
    """
    An upright electronic pipette with independently moving hand controls.
    """

    def build(self) -> DeviceSpec:
        """
        Define a handheld pipette with a hollow nozzle and detachable tip frame.
        """
        device = DeviceSpec(
            "electronic_pipette",
            "Electronic Pipette",
            "Single-channel pipette with separate plunger and tip-ejector travel.",
        )
        device.accessories.append("tip_rack")
        body = device.part("body", mass=0.15)
        body.box("grip", (0, 0, 0.204), (0.037, 0.034, 0.154), Finish.WHITE, 0.014)
        body.box(
            "back_grip",
            (0, 0.016, 0.189),
            (0.021, 0.005, 0.090),
            Finish.PALE_TEAL,
            0.007,
        )
        body.cone("lower_body", (0, 0, 0.117), 0.008, 0.017, 0.042, Finish.WHITE)
        body.cylinder("shaft", (0, 0, 0.070), 0.0055, 0.055, Finish.STEEL)
        body.ring("tip_seat", (0, 0, 0.039), 0.0048, 0.0022, 0.015, Finish.STEEL)
        body.ring("tip_seal", (0, 0, 0.045), 0.0051, 0.0042, 0.002, Finish.DARK)
        body.box(
            "screen_bezel",
            (0, -0.017, 0.248),
            (0.029, 0.004, 0.037),
            Finish.TEAL,
            0.003,
        )
        body.box(
            "display",
            (0, -0.0193, 0.249),
            (0.024, 0.001, 0.027),
            Finish.SCREEN,
            0.001,
            collision=False,
        )
        body.label("1000", (0, -0.020, 0.254), 0.005, Finish.PAPER, width=0.022)
        body.label("uL", (0, -0.020, 0.243), 0.003, Finish.PALE_TEAL)
        for index, horizontal in enumerate((-0.008, 0.008)):
            body.cylinder(
                f"volume_button_{index}",
                (horizontal, -0.018, 0.220),
                0.004,
                0.003,
                Finish.TEAL,
                rotation=(pi / 2, 0, 0),
            )
        body.label("-", (-0.008, -0.020, 0.220), 0.004, Finish.PAPER)
        body.label("+", (0.008, -0.020, 0.220), 0.004, Finish.PAPER)
        body.logo((0, -0.0172, 0.191), 0.027)
        body.label("P1000", (0, -0.0174, 0.178), 0.0036)
        body.box(
            "finger_rest", (0, 0.028, 0.268), (0.034, 0.036, 0.009), Finish.TEAL, 0.006
        )
        body.box(
            "finger_hook", (0, 0.044, 0.261), (0.030, 0.008, 0.019), Finish.TEAL, 0.003
        )
        for index in range(5):
            body.box(
                f"grip_ridge_{index}",
                (0, 0.018, 0.150 + index * 0.009),
                (0.019, 0.002, 0.003),
                Finish.TEAL,
                0.001,
                collision=False,
            )
        plunger = device.part(
            "plunger",
            "body",
            origin=(0, 0, 0.281),
            joint=JointKind.PRISMATIC,
            axis=(0, 0, 1),
            lower=-0.007,
            upper=0,
            open_value=-0.006,
            mass=0.010,
        )
        plunger.cylinder("stem", (0, 0, 0.006), 0.005, 0.014, Finish.STEEL)
        plunger.cylinder("thumb_pad", (0, 0, 0.016), 0.014, 0.007, Finish.TEAL)
        plunger.label("DOSE", (0, 0, 0.0196), 0.003, Finish.PAPER, rotation=(0, 0, 0))
        ejector = device.part(
            "tip_ejector",
            "body",
            origin=(0, 0, 0),
            joint=JointKind.PRISMATIC,
            axis=(0, 0, 1),
            lower=-0.009,
            upper=0,
            open_value=-0.008,
            mass=0.015,
        )
        ejector.ring("ejector_collar", (0, 0, 0.056), 0.009, 0.006, 0.012, Finish.TEAL)
        ejector.box(
            "ejector_rod",
            (0.010, 0.002, 0.109),
            (0.004, 0.005, 0.102),
            Finish.STEEL,
            0.001,
        )
        ejector.box(
            "ejector_button",
            (0.024, 0, 0.271),
            (0.013, 0.022, 0.016),
            Finish.TEAL,
            0.004,
        )
        ejector.box(
            "ejector_upper_rod",
            (0.020, 0.007, 0.214),
            (0.004, 0.005, 0.109),
            Finish.STEEL,
            0.001,
        )
        ejector.box(
            "ejector_bridge",
            (0.015, 0.003, 0.162),
            (0.014, 0.007, 0.005),
            Finish.STEEL,
            0.001,
        )
        device.frame("tip_mount", "body", (0, 0, 0.0315), "tool_attachment")
        device.frame("grip_center", "body", (0, 0, 0.181), "grasp")
        device.frame("plunger_contact", "plunger", (0, 0, 0.0195), "control")
        device.frame("ejector_contact", "tip_ejector", (0.024, 0, 0.279), "control")
        return device


# %% tip rack
@dataclass(frozen=True)
class TipRack:
    """
    A 24-position refillable pipette-tip carrier with a hinged clear lid.
    """

    def build(self) -> DeviceSpec:
        """
        Define a hollow carrier and independent removable socket insert.
        """
        device = DeviceSpec(
            "tip_rack",
            "Pipette Tip Rack",
            "Twenty-four open tip sockets in a covered refillable carrier.",
        )
        device.accessories.append("electronic_pipette")
        body = device.part("body", mass=0.20)
        body.box("base", (0, 0, 0.003), (0.130, 0.092, 0.006), Finish.TEAL, 0.003)
        for side, horizontal in (("left", -0.063), ("right", 0.063)):
            body.box(
                f"wall_{side}",
                (horizontal, 0, 0.028),
                (0.004, 0.090, 0.051),
                Finish.WHITE,
                0.0015,
            )
        for side, longitudinal in (("front", -0.044), ("rear", 0.044)):
            body.box(
                f"wall_{side}",
                (0, longitudinal, 0.028),
                (0.124, 0.004, 0.051),
                Finish.WHITE,
                0.0015,
            )
        body.logo((-0.025, -0.0463, 0.031), 0.041)
        body.label("TIP RACK", (0.034, -0.0463, 0.031), 0.004, width=0.046)
        body.box(
            "latch_seat", (0, -0.047, 0.048), (0.022, 0.005, 0.011), Finish.TEAL, 0.002
        )
        for horizontal in (-0.045, 0.045):
            body.cylinder(
                f"hinge_{horizontal}",
                (horizontal, 0.045, 0.057),
                0.004,
                0.018,
                Finish.TEAL,
                rotation=(0, pi / 2, 0),
            )
        insert = device.part("socket_insert", "body", origin=(0, 0, 0.019), mass=0.065)
        grid = WellGrid(6, 4, 0.016, 0.0075, 0.0045, 0.033, 0)
        grid.add_to(device, insert, Finish.PALE_TEAL, "tip_socket")
        for side, horizontal in (("left", -0.055), ("right", 0.055)):
            insert.box(
                f"insert_rail_{side}",
                (horizontal, 0, 0.030),
                (0.012, 0.080, 0.006),
                Finish.PALE_TEAL,
                0.001,
            )
        for side, longitudinal in (("front", -0.039), ("rear", 0.039)):
            insert.box(
                f"insert_rail_{side}",
                (0, longitudinal, 0.030),
                (0.104, 0.012, 0.006),
                Finish.PALE_TEAL,
                0.001,
            )
        lid = device.part(
            "lid",
            "body",
            origin=(0, 0.045, 0.057),
            joint=JointKind.REVOLUTE,
            axis=(-1, 0, 0),
            lower=0,
            upper=1.90,
            open_value=1.70,
            mass=0.055,
        )
        lid.box(
            "clear_cover",
            (0, -0.045, 0.005),
            (0.130, 0.092, 0.006),
            Finish.GLASS,
            0.003,
        )
        for side, horizontal in (("left", -0.064), ("right", 0.064)):
            lid.box(
                f"cover_edge_{side}",
                (horizontal, -0.045, 0),
                (0.003, 0.090, 0.012),
                Finish.PALE_TEAL,
                0.001,
            )
        lid.box(
            "cover_front_edge",
            (0, -0.090, 0),
            (0.126, 0.003, 0.012),
            Finish.PALE_TEAL,
            0.001,
        )
        lid.box(
            "lid_latch", (0, -0.094, -0.004), (0.022, 0.008, 0.013), Finish.TEAL, 0.002
        )
        device.frame("lid_handle", "lid", (0, -0.098, -0.003), "handle")
        device.frame("insert_grasp", "socket_insert", (0.055, 0, 0.034), "grasp")
        device.frame("body_grasp", "body", (0, 0, 0.026), "grasp")
        return device


# %% thermomixer
@dataclass(frozen=True)
class Thermomixer:
    """
    A covered heating and mixing instrument with a separate 24-well block.
    """

    def build(self) -> DeviceSpec:
        """
        Define open sample wells, a hinged cover, and a moving block carrier.
        """
        device = DeviceSpec(
            "thermomixer",
            "Thermomixer",
            "Covered temperature-controlled mixer with a removable 24-well aluminum insert.",
        )
        body = device.part("body", mass=3.8)
        BenchSupports(0.090, 0.099, 0.012).add_to(body)
        body.box("base", (0, 0, 0.017), (0.225, 0.260, 0.020), Finish.TEAL, 0.012)
        body.box("housing", (0, 0, 0.060), (0.224, 0.256, 0.074), Finish.WHITE, 0.012)
        body.box(
            "upper_seam", (0, 0.016, 0.100), (0.208, 0.210, 0.004), Finish.DARK, 0.007
        )
        body.box(
            "chamber_floor",
            (0, 0.016, 0.103),
            (0.184, 0.180, 0.005),
            Finish.STEEL,
            0.003,
        )
        for side, horizontal in (("left", -0.098), ("right", 0.098)):
            body.box(
                f"chamber_rim_{side}",
                (horizontal, 0.016, 0.123),
                (0.014, 0.210, 0.043),
                Finish.WHITE,
                0.004,
            )
        for side, longitudinal in (("front", -0.082), ("rear", 0.114)):
            body.box(
                f"chamber_rim_{side}",
                (0, longitudinal, 0.123),
                (0.182, 0.014, 0.043),
                Finish.WHITE,
                0.004,
            )
        body.logo((-0.065, -0.1283, 0.082), 0.047)
        body.label("THERMOMIXER", (0.034, -0.1283, 0.083), 0.005, width=0.102)
        body.box(
            "display_bezel",
            (-0.027, -0.128, 0.050),
            (0.123, 0.005, 0.036),
            Finish.TEAL,
            0.004,
        )
        body.box(
            "display",
            (-0.027, -0.131, 0.050),
            (0.111, 0.001, 0.027),
            Finish.SCREEN,
            0.001,
            collision=False,
        )
        body.label(
            "37.0 C   00:00",
            (-0.027, -0.1317, 0.054),
            0.0062,
            Finish.PAPER,
            width=0.102,
        )
        body.label(
            "READY     0 RPM",
            (-0.027, -0.1317, 0.043),
            0.0036,
            Finish.PALE_TEAL,
            width=0.098,
        )
        for index in range(8):
            body.box(
                f"side_vent_{index}",
                (0.1123, -0.032 + index * 0.015, 0.055),
                (0.001, 0.004, 0.029),
                Finish.DARK,
                0.0005,
                collision=False,
            )
        for horizontal in (-0.076, 0.076):
            body.cylinder(
                f"lid_hinge_{horizontal}",
                (horizontal, 0.119, 0.155),
                0.010,
                0.035,
                Finish.TEAL,
                rotation=(0, pi / 2, 0),
            )
        carrier = device.part(
            "mixing_carrier",
            "body",
            origin=(0, 0.016, 0.105),
            joint=JointKind.PRISMATIC,
            axis=(1, 0, 0),
            lower=-0.0015,
            upper=0.0015,
            mass=0.16,
        )
        carrier.cylinder("bearing", (0, 0, 0.002), 0.062, 0.004, Finish.DARK)
        orbital = device.part(
            "orbital_carrier",
            "mixing_carrier",
            origin=(0, 0, 0.002),
            joint=JointKind.PRISMATIC,
            axis=(0, 1, 0),
            lower=-0.0015,
            upper=0.0015,
            mass=0.08,
        )
        orbital.cylinder("carrier_plate", (0, 0, 0.001), 0.060, 0.002, Finish.STEEL)
        block = device.part(
            "sample_block", "orbital_carrier", origin=(0, 0, 0.002), mass=0.58
        )
        block.box(
            "block_bottom", (0, 0, 0.002), (0.169, 0.126, 0.004), Finish.STEEL, 0.002
        )
        grid = WellGrid(6, 4, 0.024, 0.0112, 0.0067, 0.040, 0.004)
        grid.add_to(device, block, Finish.STEEL, "sample_well")
        for side, horizontal in (("left", -0.079), ("right", 0.079)):
            block.box(
                f"lifting_rail_{side}",
                (horizontal, 0, 0.037),
                (0.012, 0.122, 0.014),
                Finish.STEEL,
                0.002,
            )
        for side, longitudinal in (("front", -0.057), ("rear", 0.057)):
            block.box(
                f"block_rail_{side}",
                (0, longitudinal, 0.037),
                (0.146, 0.016, 0.014),
                Finish.STEEL,
                0.002,
            )
        lid = device.part(
            "lid",
            "body",
            origin=(0, 0.119, 0.155),
            joint=JointKind.REVOLUTE,
            axis=(-1, 0, 0),
            lower=0,
            upper=1.80,
            open_value=1.65,
            mass=0.28,
        )
        for side, horizontal in (("left", -0.091), ("right", 0.091)):
            lid.box(
                f"lid_rail_{side}",
                (horizontal, -0.101, 0.013),
                (0.025, 0.212, 0.024),
                Finish.WHITE,
                0.007,
            )
        for side, longitudinal in (("front", -0.190), ("rear", -0.012)):
            lid.box(
                f"lid_rail_{side}",
                (0, longitudinal, 0.013),
                (0.157, 0.034, 0.024),
                Finish.WHITE,
                0.007,
            )
        lid.box(
            "viewing_window",
            (0, -0.101, 0.013),
            (0.157, 0.150, 0.006),
            Finish.GLASS,
            0.002,
        )
        lid.box(
            "front_handle",
            (0, -0.220, 0.013),
            (0.092, 0.026, 0.021),
            Finish.TEAL,
            0.006,
        )
        lid.box(
            "handle_mount_left",
            (-0.035, -0.204, 0.013),
            (0.013, 0.020, 0.016),
            Finish.TEAL,
            0.003,
        )
        lid.box(
            "handle_mount_right",
            (0.035, -0.204, 0.013),
            (0.013, 0.020, 0.016),
            Finish.TEAL,
            0.003,
        )
        lid.label(
            "24 x 1.5 / 2 mL",
            (0, -0.190, 0.026),
            0.005,
            rotation=(0, 0, 0),
            width=0.092,
        )
        control = device.part(
            "cycle_control",
            "body",
            origin=(0.073, -0.133, 0.050),
            joint=JointKind.REVOLUTE,
            axis=(0, -1, 0),
            lower=0,
            upper=2 * pi,
            open_value=pi / 2,
            mass=0.015,
        )
        control.cylinder(
            "knob", (0, 0, 0), 0.016, 0.012, Finish.TEAL, rotation=(pi / 2, 0, 0)
        )
        control.box(
            "pointer",
            (0, -0.0065, 0.010),
            (0.002, 0.001, 0.005),
            Finish.PAPER,
            0.0004,
            collision=False,
        )
        device.frame("lid_handle", "lid", (0, -0.232, 0.013), "handle")
        device.frame("block_grasp", "sample_block", (0.079, 0, 0.044), "grasp")
        device.frame("cycle_control_grasp", "cycle_control", (0, -0.006, 0), "control")
        device.frame("body_grasp", "body", (0, 0, 0.060), "grasp")
        return device


# %% catalog
def build_devices() -> list[DeviceSpec]:
    """
    Return independently exportable sample-preparation instruments.
    """
    return [
        VortexMixer().build(),
        ElectronicPipette().build(),
        TipRack().build(),
        Thermomixer().build(),
    ]
