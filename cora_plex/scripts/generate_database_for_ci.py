"""
This script generates the required database for the CI pipeline to execute notebooks.
This database is required for the execution of cora_plex/examples/improving_actions.ipynb.

For this script to work, you have to set a local variable in your environment called PYCRORM_CI_URI.
This variable contains the URI to the database is used for the CI pipeline with the credentials for an account that
has writing permissions to the database.

ONLY EXECUTE THIS IF YOU ARE SURE THAT YOU WANT TO DELETE THE DATABASE AND CREATE A NEW ONE.
"""

import os
import random
from datetime import timedelta

import numpy as np
import sqlalchemy.orm

from pycrap.ontologies import Robot, Milk

import cora_plex.orm.base
from cora_plex.designators.object_designator import ObjectDesignatorDescription
from cora_plex.worlds.bullet_world import BulletWorld
from cora_plex.world_concepts.world_object import Object
from cora_plex.datastructures.enums import (
    WorldMode,
    ApproachDirection,
    VerticalAlignment,
)
from cora_plex.datastructures.pose import PoseStamped
from cora_plex.ros_utils.viz_marker_publisher import VizMarkerPublisher
from cora_plex.process_module import ProcessModule, simulated_robot
from cora_plex.designators.specialized_designators.probabilistic.probabilistic_action import (
    MoveAndPickUp,
    Arms,
    Grasp,
)
from cora_plex.tasktree import task_tree
import cora_plex.orm.base


def main():
    np.random.seed(69)
    random.seed(69)

    pycrorm_uri = os.environ["PYCRORM_URI"]  # os.environ['PYCRORM_CI_URI']
    pycrorm_uri = "mysql+pymysql://" + pycrorm_uri

    engine = sqlalchemy.create_engine(pycrorm_uri)
    session = sqlalchemy.orm.sessionmaker(bind=engine)()
    cora_plex.orm.base.Base.metadata.create_all(engine)

    world = BulletWorld(WorldMode.DIRECT)

    robot = Object("pr2", Robot, "pr2.urdf")
    milk = Object("milk", Milk, "milk.stl", pose=PoseStamped.from_list([1.3, 1, 0.9]))
    viz_marker_publisher = VizMarkerPublisher()
    milk_description = ObjectDesignatorDescription(types=[Milk]).ground()

    fpa = MoveAndPickUp(
        milk_description,
        arms=[Arms.LEFT, Arms.RIGHT],
        grasps=[
            ApproachDirection.FRONT.value,
            ApproachDirection.LEFT.value,
            ApproachDirection.RIGHT.value,
            VerticalAlignment.TOP.value,
        ],
    )

    cora_plex.orm.base.ProcessMetaData().description = (
        "Experimenting with Pick Up Actions"
    )
    fpa.sample_amount = 100
    with simulated_robot:
        fpa.batch_rollout()
        task_tree.root.insert(session)
    session.commit()
    task_tree.reset_tree()
    world.exit()


if __name__ == "__main__":
    main()
