"""
Stable grasp frames shared by authored laboratory resources and semantic worlds.
"""

from enum import StrEnum


# %% authored handle frames
class LaboratoryHandle(StrEnum):
    """
    Named frames centred on the laboratory's visible handle bars.
    """

    DRAWER = "laboratory_drawer_handle"
    """
    Handle fixed to the articulated bench drawer.
    """

    CABINET_LOWER = "laboratory_cabinet_lower_handle"
    """
    Handle on the stationary lower cabinet front.
    """

    CABINET_MIDDLE = "laboratory_cabinet_middle_handle"
    """
    Handle on the stationary middle cabinet front.
    """

    CABINET_RIGHT = "laboratory_cabinet_right_handle"
    """
    Handle on the stationary right cabinet front.
    """

    CENTRIFUGE_LID = "laboratory_centrifuge_lid_handle"
    """
    Handle fixed to the centrifuge's articulated protective lid.
    """

    REFRIGERATOR_DOOR = "laboratory_refrigerator_door_handle"
    """
    Handle fixed to the sample cooler's articulated glass door.
    """
