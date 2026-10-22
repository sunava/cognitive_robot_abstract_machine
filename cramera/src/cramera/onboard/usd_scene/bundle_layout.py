"""
The layout of a CRAMERA scene bundle: the files it holds and the fixed model names its
``scene.json`` uses.

Nothing here imports more than the standard library, so readers of bundles can share
the names with the converter that writes them.
"""

from __future__ import annotations

from enum import StrEnum

# %% files


class BundleFile(StrEnum):
    """
    The files and directories of a scene bundle, relative to the bundle directory.
    """

    SCENE = "scene.json"
    """
    What the viewer loads first.
    """

    TRAJECTORY = "trajectory.json"
    """
    Joint values and poses per frame.
    """

    ENVIRONMENT = "environment.urdf"
    """
    The links and joints of the converted stage.
    """

    MESHES = "meshes"
    """
    The directory holding one GLB per environment link, and a directory per robot.
    """

    OBJECT_MESHES = "meshes/objects"
    """
    The directory holding one GLB per object.
    """

    RESIDUALS = "residuals.csv"
    """
    How far the playback strays from the recording, per body and joint.
    """

    SEMANTICS = "semantics.json"
    """
    The semantic attributes of the converted prims.
    """

    LIGHTING = "lighting.json"
    """
    The stage's sun, dome and ceiling panels.
    """

    REPORT = "conversion_report.json"
    """
    What the conversion read, wrote and left out.
    """

    INDEX = "index.json"
    """
    The scenes index in the scenes directory, next to the bundles.
    """


# %% models


class ModelName(StrEnum):
    """
    The names of the models a converted scene always has.
    """

    ENVIRONMENT = "environment"
    """
    The model holding every link that is neither the robot nor a free object.
    """
