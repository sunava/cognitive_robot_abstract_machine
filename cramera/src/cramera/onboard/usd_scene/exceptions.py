"""
Errors raised while converting a USD stage into a scene bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

from cramera.onboard.usd_scene.scene_description import ExcludedPrim, ExclusionReason

# %% stage access


@dataclass
class StageNotFound(DataclassException):
    """
    The stage file to convert does not exist.
    """

    path: Path
    """
    The stage file that was asked for.
    """

    def error_message(self) -> str:
        return f"No USD stage at {self.path}"

    def suggest_correction(self) -> str:
        return "Pass the path of an existing .usd, .usda or .usdc file."


@dataclass
class StageUnreadable(DataclassException):
    """
    The stage file exists but USD cannot open it.
    """

    path: Path
    """
    The stage file USD refused to open.
    """

    def error_message(self) -> str:
        return f"USD could not open the stage {self.path}"

    def suggest_correction(self) -> str:
        return (
            "Check that the file is a valid USD layer and that its sublayers resolve."
        )


@dataclass
class PrimNotFound(DataclassException):
    """
    A prim selected for conversion, or excluded from it, does not exist on the composed
    stage.
    """

    path: str
    """
    The prim path that was asked for.
    """

    def error_message(self) -> str:
        return f"The stage has no active prim at {self.path}"

    def suggest_correction(self) -> str:
        return "Name an existing prim path, for example /World."


@dataclass
class RobotExcluded(DataclassException):
    """
    The robot to convert as a model of its own lies in a subtree excluded from the
    conversion.
    """

    robot_path: str
    """
    The prim chosen as the robot.
    """

    excluded_path: str
    """
    The excluded prim holding it.
    """

    def error_message(self) -> str:
        return f"The robot {self.robot_path} lies in the excluded subtree {self.excluded_path}"

    def suggest_correction(self) -> str:
        return "Exclude the prims beside the robot instead of a prim holding it."


@dataclass
class SelectedPrimLeftOut(DataclassException):
    """
    A prim selected for conversion is, or lies below, a prim the conversion leaves out:

    a robot the stage marks, or an excluded prim.
    """

    prim_path: str
    """
    The selected prim.
    """

    left_out: ExcludedPrim
    """
    The outermost prim holding it that the conversion leaves out, and why.
    """

    def error_message(self) -> str:
        if self.prim_path == self.left_out.path:
            return (
                f"The selected prim {self.prim_path} is left out of the conversion "
                f"({self.left_out.reason})"
            )
        return (
            f"The selected prim {self.prim_path} lies in {self.left_out.path}, which "
            f"the conversion leaves out ({self.left_out.reason})"
        )

    def suggest_correction(self) -> str:
        if self.left_out.reason == ExclusionReason.REQUESTED:
            return "Exclude the prims beside the selected prim instead of a prim holding it."
        return (
            "Select environment prims beside the robot. To convert the robot as well, "
            "keep the stage's robots or convert this one as the scene's robot model."
        )


# %% stage variants


@dataclass
class VariantNotFound(DataclassException):
    """
    A variant selection names a prim, variant set or variant the stage does not have.
    """

    prim_path: str
    """
    The prim whose variant set was to be selected.
    """

    variant_set: str
    """
    The name of the variant set.
    """

    variant: str
    """
    The variant that was to be selected.
    """

    def error_message(self) -> str:
        return (
            f"The stage has no variant {self.variant!r} in the variant set "
            f"{self.variant_set!r} of {self.prim_path}"
        )

    def suggest_correction(self) -> str:
        return "Select a variant the prim's variant set offers, for example with usdview's variant menu."


# %% animation sources


@dataclass
class TraceNotFound(DataclassException):
    """
    The recorded body trace to animate the scene with does not exist.
    """

    path: Path
    """
    The trace file that was asked for.
    """

    def error_message(self) -> str:
        return f"No body trace at {self.path}"

    def suggest_correction(self) -> str:
        return "Pass the .npz file the twin's recorder wrote, or leave out the trace to sample the stage's time samples."


@dataclass
class TraceIncomplete(DataclassException):
    """
    The body trace lacks arrays every trace carries.
    """

    path: Path
    """
    The trace file.
    """

    missing: list[str]
    """
    The names of the arrays that are not in the file.
    """

    def error_message(self) -> str:
        return f"The body trace {self.path} has no {', '.join(self.missing)}"

    def suggest_correction(self) -> str:
        return "Record the trace again; it needs the body paths, positions, scalar-first quaternions and the sample rate."


@dataclass
class RobotRootNotFound(DataclassException):
    """
    The prim chosen as the robot holds no link, or several links none of which holds all
    the others.
    """

    path: str
    """
    The prim chosen as the robot.
    """

    candidates: list[str]
    """
    The names of the topmost links below the prim.
    """

    def error_message(self) -> str:
        if not self.candidates:
            return f"No converted link stands for {self.path} or lies below it"
        return f"{self.path} holds several separate links: {', '.join(self.candidates)}"

    def suggest_correction(self) -> str:
        return "Choose the prim of the robot's root link, such as the robot asset or its articulation root body."


# %% loading a converted bundle


@dataclass
class SceneBundleNotFound(DataclassException):
    """
    A directory given as a scene bundle holds no scene file.
    """

    directory: Path
    """
    The directory that was given.
    """

    def error_message(self) -> str:
        return f"No scene bundle at {self.directory}"

    def suggest_correction(self) -> str:
        return "Pass a directory written by cramera-import, which holds a scene.json."


@dataclass
class SceneNotFound(DataclassException):
    """
    No scene root holds a bundle of the requested name.
    """

    name: str
    """
    The scene name that was asked for.
    """

    def error_message(self) -> str:
        return f"No scene root holds a scene named {self.name}"

    def suggest_correction(self) -> str:
        return (
            "Convert the stage with cramera-import into the directory CRAMERA_SCENES "
            "points at, or pass the bundle directory."
        )


@dataclass
class UnknownPrimLink(DataclassException):
    """
    The semantics of a bundle place a prim on a link its descriptions do not define.
    """

    prim_path: str
    """
    The USD path of the prim.
    """

    link: str
    """
    The link the semantics name for the prim.
    """

    def error_message(self) -> str:
        return f"The prim {self.prim_path} belongs to the unknown link {self.link}"

    def suggest_correction(self) -> str:
        return "Convert the stage again so semantics.json and the URDF match."
