from enum import Enum

import os
from contextlib import contextmanager
from enum import Enum

from demos.bachelor_thesis.events.event_handler import EventDispatcher
from pycram.datastructures.dataclasses import Context
from pycram.plans.factories import execute_single
from semantic_digital_twin.adapters.mesh import STLParser



class Environment(Enum):
    SuturoApartmentLab = 1
    Pr2ApartmentLab = 2


#-- METHODS ------------------------------------------------------------------------------------------------------------

def perf_print(message: str):
    pass


@contextmanager
def perf_step(label: str):
    yield


def timed_parse_stl(label: str, filename: str):
    with perf_step(f"parse STL: {label}"):
        return STLParser(
            os.path.join(
                os.path.dirname(__file__), "../..", "..", "resources", "objects", filename
            )
        ).parse()


def timed_plan(label: str, action, context: Context):
    with perf_step(f"build plan: {label}"):
        return execute_single(action, context).plan

def debug_task_list_for_demo(dispatcher: EventDispatcher):
    print("\n \n", "DEBUG")

    for task in dispatcher.activated_tasks:
        print("." * 110)
        print(task.name)
        print(task.required_objects)
        print(task.precondition())
        print(task.constraints())