"""
The live publisher's HTTP server: the viewer's server and origin policy, on the local
machine only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from cramera.http.origin import LOCAL_HOSTS, OriginPolicy
from cramera.http.server import HttpServer, RequestAnswerer
from cramera_bridge.exceptions import NonLocalAddress
from cramera_bridge.live.publisher import LivePublisher
from cramera_bridge.live.routes import LiveRoutes
from cramera_protocol.live import LIVE_HOST, LIVE_PORT

# %% server


@dataclass
class LiveServer:
    """
    Serves a publisher's routes between :meth:`start` and :meth:`stop`.
    """

    publisher: LivePublisher
    """
    The publisher.
    """

    host: str = LIVE_HOST
    """
    The address to listen on, one of this machine's own.
    """

    port: int = LIVE_PORT
    """
    The port to listen on; 0 picks a free one.
    """

    policy: OriginPolicy = field(default_factory=OriginPolicy)
    """
    Which pages and host names are answered.
    """

    server: Optional[HttpServer] = field(default=None, init=False)
    """
    The listening server while started.
    """

    def answerer(self) -> RequestAnswerer:
        """
        :return: The answerer of every live route under the origin policy.
        """
        return RequestAnswerer(
            router=LiveRoutes(self.publisher).router(), policy=self.policy
        )

    def start(self) -> LiveServer:
        """
        :return: This server, listening.
        :raises NonLocalAddress: When the address is not one of this machine's own.
        """
        if self.host not in LOCAL_HOSTS:
            raise NonLocalAddress(self.host)
        self.server = HttpServer(
            answerer=self.answerer(), host=self.host, port=self.port
        ).start()
        return self

    def url(self) -> str:
        """
        :return: The base URL of the started server.
        """
        return self.server.url()

    def stop(self) -> None:
        """
        Close the socket and wait for the serving thread.
        """
        if self.server is not None:
            self.server.stop()
            self.server = None
