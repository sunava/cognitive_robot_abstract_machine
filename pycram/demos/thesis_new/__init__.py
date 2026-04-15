try:
    from demos.thesis_new.world_setup import resolve_robot_name, setup_thesis_world
except ImportError:
    from .world_setup import resolve_robot_name, setup_thesis_world

__all__ = [
    "get_thesis_demo_runner",
    "run_thesis_demo",
    "resolve_robot_name",
    "setup_thesis_world",
]


def get_thesis_demo_runner(*args, **kwargs):
    try:
        from demos.thesis_new.demo_runners import get_thesis_demo_runner as _impl
    except ImportError:
        from .demo_runners import get_thesis_demo_runner as _impl

    return _impl(*args, **kwargs)


def run_thesis_demo(*args, **kwargs):
    try:
        from demos.thesis_new.demo_runners import run_thesis_demo as _impl
    except ImportError:
        from .demo_runners import run_thesis_demo as _impl

    return _impl(*args, **kwargs)
