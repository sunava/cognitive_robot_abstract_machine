from datetime import timedelta


class ActionConfig:
    pick_up_prepose_distance = 0.03

    grasping_prepose_distance = 0.03

    navigate_keep_joint_states = True

    face_at_keep_joint_states = True

    top_grasp_flatness_ratio = 0.5
    """
    How flat an object has to be to be grasped from above rather than from the side: its
    height is at most this fraction of its smaller horizontal side.
    """

    top_grasp_max_height = 0.05
    """
    How short an object has to be, in metres, to be grasped from above rather than from
    the side, whatever its footprint.
    """

    execution_delay: timedelta = timedelta(seconds=0.0)
    """
    The delay between the execution of actions/motions to imitate real world execution
    time.
    """
