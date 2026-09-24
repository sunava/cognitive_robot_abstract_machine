"""
Display labels shared by laboratory scenes and their public catalog.
"""

from enum import StrEnum


# %% laboratory display identity
class LaboratoryBranding(StrEnum):
    """
    Public laboratory name independent of stable scene identifiers.
    """

    NAME = "AICOR Laboratory"
    """
    Visible environment name in authored scenes, replays and scene pickers.
    """
