from demos.thesis_new import run_thesis_demo_isolated
from krrood.ormatic.utils import drop_database
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
import matplotlib
import os

# use this in terminal: MPLBACKEND=Agg python headmaps.py
matplotlib.use("Agg")
robots = ("tiago", "justin", "stretch", "hsrb", "pr2", "armar7", "g1", "garmi")
robots = ("garmi",)
actions = ("mix", "wipe", "cut")
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "records")
ACTION_RESULT_CSVS = {
    "mix": os.path.join(RECORDS_DIR, "mix_all_bowls_results.csv"),
    "mixing": os.path.join(RECORDS_DIR, "mix_all_bowls_results.csv"),
    "wipe": os.path.join(RECORDS_DIR, "wipe_all_spaces_results.csv"),
    "wiping": os.path.join(RECORDS_DIR, "wipe_all_spaces_results.csv"),
    "cut": os.path.join(RECORDS_DIR, "cut_all_breads_results.csv"),
    "cutting": os.path.join(RECORDS_DIR, "cut_all_breads_results.csv"),
}
TARGET_CSV_TRIES = 3000
ACTIONS_TO_RUN = ["mix"]
ENVIRONMENT_SEQUENCE = (
    "apartment",
    "isr",
    "kitchen",
)


def _csv_try_count(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0
    with open(csv_path, encoding="utf-8") as csv_file:
        return max(0, sum(1 for _ in csv_file) - 1)


def _run_action_until_csv_target(action, target_tries=TARGET_CSV_TRIES):
    csv_path = ACTION_RESULT_CSVS[action]
    cycle = 0
    while _csv_try_count(csv_path) < target_tries:
        cycle += 1
        print(
            f"[main] {action}: cycle {cycle}, "
            f"{_csv_try_count(csv_path)}/{target_tries} tries in {csv_path}"
        )
        for env in ENVIRONMENT_SEQUENCE:
            for robot in robots:
                if _csv_try_count(csv_path) >= target_tries:
                    return
                run_thesis_demo_isolated(
                    task_name=action,
                    robot_name=robot,
                    environment_name=env,
                )


# pr2,hsrb,stretch,tiago,g1,justin,armar7
# apartment,kitchen,isr?, suturo, robocup, isr-testbed
if __name__ == "__main__":
    session = pycram_sessionmaker()()
    drop_database(session.bind)
    Base.metadata.create_all(session.bind)
    session.commit()

    for action in ACTIONS_TO_RUN:
        _run_action_until_csv_target(action)

    # run_thesis_demo(
    #     "cut",
    #     robot_name="justin",
    #     environment_name="apartment",
    #     container_kind="cucumber",z`
    # )
    # for env in ("isr", "kitchen", "apartment"):
    #     for action in actions:
    #         for robot in robots:
    #             run_thesis_demo(
    #                 task_name=action,
    #                 robot_name=robot,
    #                 environment_name=env,
    #             )

    #     )
    # for action in actions:
    #     for robot in robots:
    #         run_thesis_demo(
    #             action,
    #             robot_name=robot,
    #             environment_name="apartment",
    #         )
