"""
Parallel laboratory pages retain independent authenticated browser sessions.
"""

from dataclasses import replace
from http import HTTPStatus
from http.cookies import SimpleCookie
import threading

from cramera.laboratory_share import make_server
from cramera.laboratory_robot_session import RobotPhysicsRoute

from .test_laboratory_share import (
    SharedLaboratory,
    shared_laboratory,
    sharing_prefix,
    sharing_replay,
    sharing_scene,
)


# %% shared browser authentication
def test_two_laboratories_remain_accessible_after_both_logins(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    Logging into a compact world must not replace the larger world's session.
    """
    options = replace(shared_laboratory.options, path_prefix="/laboratory-small")
    server = make_server(options)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    compact = SharedLaboratory(
        server,
        shared_laboratory.backend,
        options,
        shared_laboratory.origin,
        shared_laboratory.password,
    )
    try:
        cookies = SimpleCookie()
        cookies.load(shared_laboratory.login().headers["Set-Cookie"])
        cookies.load(compact.login().headers["Set-Cookie"])
        browser_cookie = "; ".join(
            f"{name}={value.value}" for name, value in cookies.items()
        )
        for laboratory in (shared_laboratory, compact):
            response = laboratory.request(
                laboratory.options.path_prefix + RobotPhysicsRoute.STATE,
                headers={"Cookie": browser_cookie},
            )
            assert response.status == HTTPStatus.OK
        assert (
            compact.request(
                options.path_prefix + RobotPhysicsRoute.STATE,
                headers={"Cookie": shared_laboratory.cookie},
            ).status
            == HTTPStatus.UNAUTHORIZED
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
