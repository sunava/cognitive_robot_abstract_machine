"""
Original articulated AICOR instruments for imaging, analysis and identification.
"""

from __future__ import annotations

from math import atan2, cos, hypot, pi, sin

from spec import DeviceSpec, Finish, JointKind

# %% digital microscope


def build_digital_microscope() -> DeviceSpec:
    """
    Build a digital microscope with focus, slide travel and objective selection.
    """
    device = DeviceSpec(
        "digital_microscope",
        "Digital microscope",
        "Digital microscope with a focus carriage, a two-axis slide stage and a three-position objective turret.",
    )
    base = device.part("base", mass=3.0)
    base.box("foot", (0, 0, 0.031), (0.268, 0.330, 0.052), Finish.WHITE, 0.019)
    base.box("foot_trim", (0, 0, 0.009), (0.251, 0.313, 0.014), Finish.TEAL, 0.008)
    for side in (-1, 1):
        for end in (-1, 1):
            base.cylinder(
                "rubber_foot",
                (side * 0.105, end * 0.127, 0.004),
                0.014,
                0.008,
                Finish.DARK,
            )
    base.box("arm_lower", (0, 0.100, 0.133), (0.109, 0.101, 0.163), Finish.WHITE, 0.034)
    base.box("arm_rear", (0, 0.132, 0.259), (0.092, 0.098, 0.157), Finish.WHITE, 0.032)
    base.box(
        "arm_shoulder", (0, 0.067, 0.351), (0.101, 0.168, 0.080), Finish.WHITE, 0.031
    )
    base.box("arm_inlay", (0, 0.183, 0.250), (0.062, 0.006, 0.151), Finish.TEAL, 0.009)
    base.box(
        "focus_guide", (0, 0.042, 0.204), (0.055, 0.014, 0.170), Finish.STEEL, 0.002
    )
    base.cylinder("illuminator", (0, -0.101, 0.065), 0.039, 0.018, Finish.DARK)
    base.cylinder(
        "light_diffuser",
        (0, -0.101, 0.075),
        0.031,
        0.003,
        Finish.PALE_TEAL,
        collision=False,
    )
    base.logo((-0.055, -0.167, 0.032), 0.061)
    base.label("DIGITAL MICROSCOPE", (0.039, -0.167, 0.031), 0.006, width=0.100)
    base.cylinder("light_dial", (0.099, -0.120, 0.064), 0.014, 0.012, Finish.TEAL)
    base.box(
        "power_key",
        (-0.096, -0.120, 0.061),
        (0.020, 0.016, 0.008),
        Finish.PALE_TEAL,
        0.003,
    )

    optical = device.part(
        "optical_head", parent="base", origin=(0, -0.073, 0.347), mass=0.65
    )
    optical.cylinder("camera_column", (0, 0, 0.012), 0.038, 0.131, Finish.WHITE)
    optical.cylinder("camera_collared_base", (0, 0, -0.047), 0.0395, 0.018, Finish.TEAL)
    optical.box(
        "monitor_pivot", (0, 0.020, 0.094), (0.047, 0.032, 0.025), Finish.DARK, 0.008
    )
    monitor = device.part(
        "monitor",
        parent="optical_head",
        origin=(0, 0.020, 0.095),
        joint=JointKind.REVOLUTE,
        axis=(1, 0, 0),
        lower=-0.12,
        upper=0.50,
        open_value=0.20,
        mass=0.22,
    )
    monitor.box(
        "monitor_back", (0, 0, 0.040), (0.179, 0.021, 0.118), Finish.WHITE, 0.010
    )
    monitor.box(
        "monitor_bezel", (0, -0.012, 0.042), (0.161, 0.005, 0.099), Finish.DARK, 0.006
    )
    monitor.box(
        "monitor_screen",
        (0, -0.015, 0.045),
        (0.147, 0.001, 0.082),
        Finish.SCREEN,
        0.002,
        collision=False,
    )
    monitor.label("LIVE VIEW", (-0.034, -0.016, 0.074), 0.005, Finish.PALE_TEAL)
    monitor.label("10x    READY", (0.012, -0.016, 0.014), 0.005, Finish.PAPER)
    monitor.cylinder(
        "screen_status",
        (0.070, -0.012, -0.008),
        0.0025,
        0.003,
        Finish.TEAL,
        (pi / 2, 0, 0),
        collision=False,
    )

    turret = device.part(
        "objective_turret",
        parent="optical_head",
        origin=(0, 0, -0.073),
        joint=JointKind.REVOLUTE,
        lower=-2 * pi / 3,
        upper=2 * pi / 3,
        open_value=2 * pi / 3,
        mass=0.18,
    )
    turret.cylinder("turret_disk", (0, 0, 0), 0.050, 0.018, Finish.DARK)
    turret.cylinder("turret_hub", (0, 0, 0.012), 0.018, 0.011, Finish.STEEL)
    for number, (height, accent) in enumerate(
        ((0.031, Finish.RED), (0.038, Finish.AMBER), (0.045, Finish.BLUE))
    ):
        angle = -pi / 2 + number * 2 * pi / 3
        center = (0.028 * cos(angle), 0.028 * sin(angle))
        turret.cylinder(
            "objective_barrel",
            (*center, -0.009 - height / 2),
            0.010,
            height,
            Finish.STEEL,
        )
        turret.cylinder(
            "objective_band", (*center, -height + 0.001), 0.0105, 0.005, accent
        )
        turret.cylinder(
            "objective_tip", (*center, -height - 0.007), 0.0075, 0.006, Finish.DARK
        )
        turret.cylinder(
            "objective_lens",
            (*center, -height - 0.0101),
            0.0054,
            0.0005,
            Finish.GLASS,
            collision=False,
        )

    focus = device.part(
        "focus_carriage",
        parent="base",
        origin=(0, 0.025, 0.151),
        joint=JointKind.PRISMATIC,
        axis=(0, 0, 1),
        lower=-0.020,
        upper=0.020,
        open_value=0.010,
        mass=0.44,
    )
    focus.box("carriage", (0, 0, 0.001), (0.099, 0.033, 0.064), Finish.DARK, 0.006)
    for side in (-1, 1):
        focus.box(
            "stage_support_side",
            (side * 0.052, -0.086, 0),
            (0.024, 0.183, 0.014),
            Finish.DARK,
            0.005,
        )
        focus.box(
            "stage_support_end",
            (0, -0.126 + side * 0.050, 0),
            (0.092, 0.019, 0.014),
            Finish.DARK,
            0.005,
        )
    for side in (-1, 1):
        focus.cylinder(
            "coarse_focus_dial",
            (side * 0.078, 0, 0.005),
            0.028,
            0.030,
            Finish.TEAL,
            (0, pi / 2, 0),
        )
        focus.cylinder(
            "fine_focus_dial",
            (side * 0.099, 0, 0.005),
            0.017,
            0.015,
            Finish.DARK,
            (0, pi / 2, 0),
        )
    slide_x = device.part(
        "slide_x",
        parent="focus_carriage",
        origin=(0, -0.126, 0.016),
        joint=JointKind.PRISMATIC,
        axis=(1, 0, 0),
        lower=-0.028,
        upper=0.028,
        open_value=0.020,
        mass=0.15,
    )
    for side in (-1, 1):
        slide_x.box(
            "horizontal_slide_side",
            (side * 0.067, 0, 0),
            (0.032, 0.130, 0.014),
            Finish.STEEL,
            0.004,
        )
        slide_x.box(
            "horizontal_slide_end",
            (0, side * 0.054, 0),
            (0.116, 0.023, 0.014),
            Finish.STEEL,
            0.004,
        )
    slide_y = device.part(
        "slide_y",
        parent="slide_x",
        origin=(0, 0, 0.013),
        joint=JointKind.PRISMATIC,
        axis=(0, 1, 0),
        lower=-0.020,
        upper=0.020,
        open_value=-0.015,
        mass=0.10,
    )
    for side in (-1, 1):
        slide_y.box(
            "slide_side_rail",
            (side * 0.057, 0, 0),
            (0.024, 0.115, 0.012),
            Finish.DARK,
            0.003,
        )
        slide_y.box(
            "slide_end_rail",
            (0, side * 0.044, 0),
            (0.091, 0.027, 0.012),
            Finish.DARK,
            0.003,
        )
        slide_y.box(
            "slide_clip",
            (side * 0.043, 0.008, 0.011),
            (0.006, 0.044, 0.007),
            Finish.STEEL,
            0.001,
        )
    slide_y.box(
        "specimen_slide",
        (0, 0, 0.008),
        (0.075, 0.025, 0.0014),
        Finish.GLASS,
        0.0003,
        collision=False,
    )
    slide_y.box(
        "slide_label",
        (0.027, 0, 0.009),
        (0.017, 0.024, 0.0005),
        Finish.PAPER,
        0.0002,
        collision=False,
    )
    slide_y.label("S01", (0.027, 0, 0.0094), 0.004, rotation=(0, 0, 0))
    device.frame("sample_stage", "slide_y", (0, 0, 0.010), "sample")
    device.frame(
        "focus_control", "focus_carriage", (0.108, 0, 0.005), "control", (0, pi / 2, 0)
    )
    device.frame("body_grasp", "base", (0, 0.133, 0.247), "grasp")
    device.frame("objective_axis", "optical_head", (0, -0.028, -0.118), "measurement")
    return device


# %% pH meter and removable probe


def build_ph_meter() -> DeviceSpec:
    """
    Build a benchtop meter with a rotating probe support and open rinse holder.
    """
    device = DeviceSpec(
        "ph_meter",
        "pH meter",
        "Benchtop pH meter with an articulated support arm and a removable probe resource.",
    )
    base = device.part("base", mass=1.8)
    base.box("meter_lower", (0, 0, 0.022), (0.231, 0.177, 0.035), Finish.TEAL, 0.015)
    base.box("meter_housing", (0, 0, 0.048), (0.228, 0.173, 0.051), Finish.WHITE, 0.016)
    base.box(
        "screen_bezel",
        (-0.014, -0.007, 0.075),
        (0.144, 0.098, 0.008),
        Finish.DARK,
        0.006,
    )
    base.box(
        "screen",
        (-0.014, -0.007, 0.0794),
        (0.132, 0.085, 0.001),
        Finish.SCREEN,
        0.003,
        collision=False,
    )
    base.label(
        "7.00 pH", (-0.017, -0.004, 0.0801), 0.018, Finish.PAPER, (0, 0, 0), width=0.112
    )
    base.label(
        "25.0 C    READY", (-0.015, -0.031, 0.0801), 0.005, Finish.PALE_TEAL, (0, 0, 0)
    )
    base.logo((-0.057, -0.088, 0.045), 0.064)
    base.label("pH METER", (0.054, -0.088, 0.044), 0.008)
    for number, title in enumerate(("READ", "CAL", "MODE")):
        base.box(
            "meter_key",
            (0.090, 0.040 - 0.034 * number, 0.077),
            (0.026, 0.022, 0.008),
            Finish.PALE_TEAL,
            0.004,
        )
        base.label(
            title, (0.090, 0.040 - 0.034 * number, 0.0813), 0.004, rotation=(0, 0, 0)
        )
    base.box(
        "stand_foot", (0.174, 0.010, 0.012), (0.098, 0.205, 0.024), Finish.WHITE, 0.012
    )
    base.cylinder("stand_socket", (0.174, 0.077, 0.033), 0.025, 0.030, Finish.TEAL)
    base.cylinder("support_column", (0.174, 0.077, 0.217), 0.009, 0.350, Finish.STEEL)
    base.sphere("column_cap", (0.174, 0.077, 0.391), 0.010, Finish.DARK)
    base.ring(
        "rinse_holder", (0.174, -0.043, 0.044), 0.041, 0.0355, 0.062, Finish.PALE_TEAL
    )
    base.cylinder(
        "rinse_holder_floor", (0.174, -0.043, 0.016), 0.0355, 0.005, Finish.PALE_TEAL
    )
    base.label("RINSE", (0.174, -0.085, 0.044), 0.005)
    collar = device.part(
        "support_height",
        parent="base",
        origin=(0.174, 0.077, 0.260),
        joint=JointKind.PRISMATIC,
        axis=(0, 0, 1),
        lower=-0.070,
        upper=0.085,
        open_value=0.045,
        mass=0.12,
    )
    collar.cylinder("height_collar", (0, 0, 0), 0.021, 0.035, Finish.TEAL)
    collar.cylinder(
        "height_lock", (0.025, 0, 0), 0.013, 0.023, Finish.DARK, (0, pi / 2, 0)
    )
    arm = device.part(
        "probe_arm",
        parent="support_height",
        origin=(0, 0, 0),
        joint=JointKind.REVOLUTE,
        axis=(0, 0, 1),
        lower=-1.9,
        upper=1.9,
        open_value=0.7,
        mass=0.11,
    )
    arm.box("arm", (0, -0.052, 0), (0.022, 0.104, 0.019), Finish.WHITE, 0.007)
    arm.ring("probe_clamp", (0, -0.120, 0), 0.017, 0.0095, 0.021, Finish.TEAL)
    arm.cylinder(
        "clamp_screw", (0.021, -0.120, 0), 0.007, 0.016, Finish.DARK, (0, pi / 2, 0)
    )
    device.frame("probe_mount", "probe_arm", (0, -0.120, -0.155), "mount")
    device.frame("rinse_station", "base", (0.174, -0.043, 0.020), "sample")
    device.frame("read_control", "base", (0.090, 0.040, 0.082), "control")
    device.frame("body_grasp", "base", (0, 0, 0.046), "grasp")
    device.accessories.append("ph_probe")
    return device


def build_ph_probe() -> DeviceSpec:
    """
    Build a separate glass electrode with a sensing tip and cable connector.
    """
    device = DeviceSpec(
        "ph_probe",
        "pH probe",
        "Removable glass electrode with a sensing bulb, protective cage and detachable cable connector.",
    )
    probe = device.part("probe", mass=0.08)
    probe.sphere("sensing_bulb", (0, 0, 0.015), 0.006, Finish.GLASS)
    probe.cylinder("glass_stem", (0, 0, 0.059), 0.0055, 0.084, Finish.GLASS)
    probe.cylinder(
        "electrode_core", (0, 0, 0.067), 0.0014, 0.076, Finish.STEEL, collision=False
    )
    for number in range(4):
        angle = number * pi / 2
        probe.box(
            "tip_guard",
            (0.008 * cos(angle), 0.008 * sin(angle), 0.033),
            (0.003, 0.003, 0.055),
            Finish.WHITE,
            0.0008,
        )
    probe.ring("guard_collar", (0, 0, 0.059), 0.0105, 0.006, 0.009, Finish.WHITE)
    probe.cylinder("probe_handle", (0, 0, 0.130), 0.009, 0.137, Finish.WHITE)
    probe.cylinder("handle_band", (0, 0, 0.183), 0.0095, 0.012, Finish.TEAL)
    probe.cylinder("connector", (0, 0, 0.210), 0.006, 0.026, Finish.STEEL)
    probe.cylinder("connector_tip", (0, 0, 0.225), 0.004, 0.009, Finish.DARK)
    probe.label("pH", (0, -0.00915, 0.139), 0.006, width=0.010)
    device.frame("body_grasp", "probe", (0, 0, 0.142), "grasp")
    device.frame("sensing_tip", "probe", (0, 0, 0.009), "measurement")
    device.frame("cable_connector", "probe", (0, 0, 0.229), "connector")
    return device


# %% handheld identification scanner


def build_barcode_scanner() -> DeviceSpec:
    """
    Build an independent handheld scanner with a movable trigger.
    """
    device = DeviceSpec(
        "barcode_scanner",
        "Barcode scanner",
        "Handheld identification scanner with a trigger and a separate docking cradle.",
    )
    body = device.part("body", mass=0.23)
    body.box(
        "grip",
        (0, 0.014, 0.067),
        (0.038, 0.047, 0.115),
        Finish.WHITE,
        0.013,
        rotation=(0.15, 0, 0),
    )
    body.box(
        "grip_inlay",
        (0, 0.040, 0.064),
        (0.027, 0.005, 0.082),
        Finish.DARK,
        0.003,
        rotation=(0.15, 0, 0),
    )
    body.box(
        "handle_foot", (0, 0.023, 0.011), (0.043, 0.050, 0.019), Finish.TEAL, 0.006
    )
    body.box("head", (0, -0.015, 0.146), (0.079, 0.128, 0.058), Finish.WHITE, 0.020)
    body.box("head_trim", (0, -0.016, 0.127), (0.077, 0.125, 0.010), Finish.TEAL, 0.008)
    body.box(
        "scan_bezel", (0, -0.079, 0.147), (0.066, 0.011, 0.040), Finish.DARK, 0.006
    )
    body.box(
        "scan_window",
        (0, -0.085, 0.148),
        (0.051, 0.001, 0.026),
        Finish.SCREEN,
        0.003,
        collision=False,
    )
    body.cylinder(
        "optical_sensor",
        (-0.011, -0.086, 0.150),
        0.007,
        0.0015,
        Finish.GLASS,
        (pi / 2, 0, 0),
        collision=False,
    )
    body.box(
        "scan_light",
        (0.012, -0.086, 0.144),
        (0.018, 0.001, 0.004),
        Finish.RED,
        0.001,
        collision=False,
    )
    body.box(
        "status_led",
        (0, -0.025, 0.1755),
        (0.020, 0.009, 0.002),
        Finish.PALE_TEAL,
        0.002,
        collision=False,
    )
    body.logo((0, 0.012, 0.176), 0.041, rotation=(0, 0, 0))
    body.label("SCAN", (0, -0.058, 0.177), 0.005, rotation=(0, 0, 0))
    trigger = device.part(
        "trigger",
        parent="body",
        origin=(0, -0.021, 0.113),
        joint=JointKind.REVOLUTE,
        axis=(1, 0, 0),
        lower=0,
        upper=0.22,
        open_value=0.22,
        mass=0.008,
    )
    trigger.box(
        "trigger_button",
        (0, -0.005, -0.012),
        (0.028, 0.012, 0.030),
        Finish.TEAL,
        0.004,
        rotation=(-0.15, 0, 0),
    )
    device.frame("body_grasp", "body", (0, 0.012, 0.068), "grasp")
    device.frame("scan_axis", "body", (0, -0.086, 0.148), "measurement", (pi / 2, 0, 0))
    device.frame("trigger_control", "trigger", (0, -0.012, -0.015), "control")
    device.frame("dock_contact", "body", (0, 0.023, 0.003), "mount")
    device.accessories.append("barcode_dock")
    return device


def build_barcode_dock() -> DeviceSpec:
    """
    Build an open cradle that leaves the scanner grip accessible to a robot.
    """
    device = DeviceSpec(
        "barcode_dock",
        "Barcode scanner dock",
        "Weighted charging cradle for the independent handheld barcode scanner.",
    )
    base = device.part("base", mass=0.38)
    base.box("dock_base", (0, 0, 0.018), (0.116, 0.124, 0.031), Finish.WHITE, 0.014)
    base.box("dock_trim", (0, 0, 0.005), (0.109, 0.117, 0.009), Finish.TEAL, 0.006)
    base.box(
        "handle_support_floor",
        (0, 0.015, 0.038),
        (0.060, 0.071, 0.012),
        Finish.DARK,
        0.004,
    )
    for side in (-1, 1):
        base.box(
            "cradle_side",
            (side * 0.030, 0.014, 0.056),
            (0.010, 0.074, 0.044),
            Finish.PALE_TEAL,
            0.004,
        )
    base.box(
        "cradle_back", (0, 0.047, 0.060), (0.061, 0.012, 0.052), Finish.PALE_TEAL, 0.004
    )
    base.box(
        "cradle_front_stop",
        (0, -0.020, 0.046),
        (0.058, 0.011, 0.020),
        Finish.PALE_TEAL,
        0.003,
    )
    for side in (-1, 1):
        base.cylinder(
            "charging_contact",
            (side * 0.011, 0.015, 0.045),
            0.0025,
            0.004,
            Finish.STEEL,
        )
    base.label("SCANNER DOCK", (0, -0.063, 0.019), 0.005, width=0.082)
    device.frame("scanner_mount", "base", (0, -0.008, 0.044), "mount")
    device.frame("body_grasp", "base", (0, 0, 0.020), "grasp")
    device.accessories.append("barcode_scanner")
    return device


# %% label printer


def build_label_printer() -> DeviceSpec:
    """
    Build an opening thermal-label printer with accessible roll and feed path.
    """
    device = DeviceSpec(
        "label_printer",
        "Label printer",
        "Desktop specimen-label printer with an opening media cover, a roll spindle and a feed roller.",
    )
    base = device.part("base", mass=1.7)
    base.box("printer_floor", (0, 0, 0.018), (0.212, 0.260, 0.036), Finish.TEAL, 0.013)
    for side in (-1, 1):
        base.box(
            "printer_lower_side",
            (side * 0.099, 0.002, 0.060),
            (0.014, 0.253, 0.071),
            Finish.WHITE,
            0.007,
        )
        base.cylinder(
            "cover_hinge",
            (side * 0.100, 0.108, 0.094),
            0.006,
            0.066,
            Finish.DARK,
            (0, pi / 2, 0),
        )
        base.box(
            "roll_support",
            (side * 0.079, 0.032, 0.065),
            (0.013, 0.029, 0.072),
            Finish.PALE_TEAL,
            0.006,
        )
    base.box(
        "printer_rear", (0, 0.121, 0.061), (0.191, 0.015, 0.074), Finish.WHITE, 0.005
    )
    base.box(
        "front_upper", (0, -0.118, 0.077), (0.201, 0.024, 0.031), Finish.WHITE, 0.008
    )
    base.box(
        "front_lower", (0, -0.118, 0.037), (0.201, 0.024, 0.033), Finish.WHITE, 0.008
    )
    base.box(
        "output_slot_lining",
        (0, -0.120, 0.054),
        (0.136, 0.028, 0.001),
        Finish.DARK,
        0.0002,
    )
    base.box(
        "label_lip", (0, -0.139, 0.047), (0.146, 0.020, 0.006), Finish.STEEL, 0.001
    )
    base.logo((-0.035, -0.131, 0.078), 0.066)
    base.label("LABEL PRINTER", (0.045, -0.131, 0.078), 0.004, width=0.070)
    for side in (-1, 1):
        for end in (-1, 1):
            base.cylinder(
                "rubber_foot",
                (side * 0.079, end * 0.100, 0.002),
                0.009,
                0.004,
                Finish.DARK,
            )

    cover = device.part(
        "media_cover",
        parent="base",
        origin=(0, 0.108, 0.094),
        joint=JointKind.REVOLUTE,
        axis=(-1, 0, 0),
        lower=0,
        upper=1.70,
        open_value=1.35,
        mass=0.36,
    )
    cover.box(
        "cover_top", (0, -0.100, 0.079), (0.213, 0.233, 0.032), Finish.WHITE, 0.014
    )
    for side in (-1, 1):
        cover.box(
            "cover_side",
            (side * 0.099, -0.11475, 0.038),
            (0.016, 0.2035, 0.064),
            Finish.WHITE,
            0.008,
        )
        cover.ring(
            "hinge_sleeve",
            (side * 0.119, 0, 0),
            0.016,
            0.007,
            0.020,
            Finish.TEAL,
            (0, pi / 2, 0),
        )
        cover.box(
            "hinge_strap",
            (side * 0.119, -0.021, 0.016),
            (0.020, 0.026, 0.032),
            Finish.WHITE,
            0.003,
        )
        cover.box(
            "hinge_bridge",
            (side * 0.106, -0.030, 0.027),
            (0.027, 0.021, 0.018),
            Finish.WHITE,
            0.003,
        )
    cover.box(
        "cover_front", (0, -0.207, 0.039), (0.192, 0.015, 0.065), Finish.WHITE, 0.006
    )
    cover.box(
        "inspection_window",
        (-0.034, -0.058, 0.096),
        (0.077, 0.103, 0.0015),
        Finish.SCREEN,
        0.012,
        collision=False,
    )
    cover.label("MEDIA", (-0.034, -0.058, 0.097), 0.008, Finish.PALE_TEAL, (0, 0, 0))
    cover.box(
        "status_panel",
        (0.056, -0.091, 0.096),
        (0.034, 0.130, 0.003),
        Finish.TEAL,
        0.010,
    )
    cover.cylinder("feed_key", (0.056, -0.119, 0.100), 0.011, 0.005, Finish.PALE_TEAL)
    cover.label("FEED", (0.056, -0.144, 0.099), 0.004, Finish.PAPER, (0, 0, 0))
    cover.box(
        "ready_light",
        (0.056, -0.067, 0.099),
        (0.011, 0.005, 0.001),
        Finish.PALE_TEAL,
        0.001,
        collision=False,
    )
    cover.label("READY", (0.056, -0.052, 0.099), 0.0036, Finish.PAPER, (0, 0, 0))
    cover.box(
        "cover_handle", (0, -0.218, 0.050), (0.066, 0.019, 0.014), Finish.TEAL, 0.005
    )

    spindle = device.part(
        "media_spindle",
        parent="base",
        origin=(0, 0.030, 0.097),
        joint=JointKind.CONTINUOUS,
        axis=(1, 0, 0),
        open_value=pi / 3,
        mass=0.22,
    )
    spindle.cylinder("spindle", (0, 0, 0), 0.012, 0.159, Finish.STEEL, (0, pi / 2, 0))
    spindle.ring(
        "paper_roll", (0, 0, 0), 0.055, 0.020, 0.114, Finish.PAPER, (0, pi / 2, 0)
    )
    spindle.ring(
        "paper_core", (0, 0, 0), 0.020, 0.0125, 0.117, Finish.AMBER, (0, pi / 2, 0)
    )
    for side in (-1, 1):
        spindle.cylinder(
            "roll_guide",
            (side * 0.062, 0, 0),
            0.039,
            0.008,
            Finish.PALE_TEAL,
            (0, pi / 2, 0),
        )
    feed = device.part(
        "feed_roller",
        parent="base",
        origin=(0, -0.079, 0.058),
        joint=JointKind.CONTINUOUS,
        axis=(1, 0, 0),
        open_value=pi / 2,
        mass=0.045,
    )
    feed.cylinder("rubber_feed", (0, 0, 0), 0.012, 0.125, Finish.DARK, (0, pi / 2, 0))
    base.box(
        "label_web",
        (0, -0.0555, 0.071),
        (0.104, 0.049, 0.0008),
        Finish.PAPER,
        0.0001,
        collision=False,
    )
    base.box(
        "exit_web",
        (0, -0.0925, 0.064),
        (0.104, hypot(0.025, 0.014), 0.0008),
        Finish.PAPER,
        0.0001,
        collision=False,
        rotation=(atan2(0.014, 0.025), 0, 0),
    )
    base.box(
        "printed_label",
        (0, -0.1545, 0.057),
        (0.104, 0.099, 0.0008),
        Finish.PAPER,
        0.0001,
        collision=False,
    )
    base.label("SAMPLE 0001", (0, -0.185, 0.0575), 0.006, Finish.INK, (0, 0, 0))
    for number in range(24):
        width = 0.0015 if number % 3 else 0.0025
        base.box(
            "barcode_bar",
            (-0.040 + number * 0.0034, -0.161, 0.0576),
            (width, 0.021, 0.0001),
            Finish.INK,
            0.00003,
            collision=False,
        )
    device.frame("cover_handle", "media_cover", (0, -0.229, 0.050), "grasp")
    device.frame("media_loading", "base", (0, 0.030, 0.097), "sample")
    device.frame("label_output", "base", (0, -0.204, 0.057), "output")
    device.frame("feed_control", "media_cover", (0.056, -0.119, 0.103), "control")
    device.frame("body_grasp", "base", (0, 0, 0.041), "grasp")
    return device


# %% catalog


def build_devices() -> list[DeviceSpec]:
    """
    Return the independently placeable analysis and identification instruments.
    """
    return [
        build_digital_microscope(),
        build_ph_meter(),
        build_ph_probe(),
        build_barcode_scanner(),
        build_barcode_dock(),
        build_label_printer(),
    ]
