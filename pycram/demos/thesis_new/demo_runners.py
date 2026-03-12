from demos.thesis_new.demo_cut_all_breads_retry import main_cutting
from demos.thesis_new.demo_mix_all_bowls_retry import main_mixing
from demos.thesis_new.demo_wipe_all_spaces_retry import main_wiping
from demos.thesis_new.world_setup import resolve_robot_name

THESIS_DEMO_RUNNERS = {
    "cut": main_cutting,
    "cutting": main_cutting,
    "mix": main_mixing,
    "mixing": main_mixing,
    "wipe": main_wiping,
    "wiping": main_wiping,
}


def get_thesis_demo_runner(task_name):
    normalized = str(task_name).strip().lower()
    if normalized not in THESIS_DEMO_RUNNERS:
        supported = ", ".join(sorted(THESIS_DEMO_RUNNERS))
        raise ValueError(f"Unsupported thesis_new demo '{task_name}'. Supported: {supported}")
    return THESIS_DEMO_RUNNERS[normalized]


def run_thesis_demo(task_name, *, seed=None, robot_name=None, environment_name=None):
    runner = get_thesis_demo_runner(task_name)
    return runner(
        seed=seed,
        robot_name=resolve_robot_name(robot_name),
        environment_name=environment_name,
    )
