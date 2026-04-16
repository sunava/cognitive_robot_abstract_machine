import logging
import warnings

logging.getLogger("solvers").setLevel(logging.ERROR)
logging.getLogger("polytope").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*cvxopt\.glpk.*")
warnings.filterwarnings("ignore", message=r".*scipy\.optimize\.linprog.*")

from demos.thesis_single_object.single_object_cut_demo import run_single_object_cut_demo

ROBOT = "g1"
ENVIRONMENT = "kitchen"
OBJECT_KIND = "apple"
SPAWN_POSITION = None
SPAWN_YAW = None
SPAWN_SCALE = 1.0


if __name__ == "__main__":
    run_single_object_cut_demo(
        robot_name=ROBOT,
        environment_name=ENVIRONMENT,
        object_kind=OBJECT_KIND,
        spawn_position=SPAWN_POSITION,
        spawn_yaw=SPAWN_YAW,
        spawn_scale=SPAWN_SCALE,
    )
