"""
The live publisher's HTTP server: where it listens, whom it answers and its routes.
"""

from __future__ import annotations

import json
import urllib.request
from http import HTTPStatus

import pytest

from cramera.http.messages import Header, HttpMethod
from cramera.http.origin import LOCAL_HOSTS
from cramera_bridge.exceptions import NonLocalAddress
from cramera_bridge.live.server import LIVE_HOST, LIVE_PORT, LiveServer
from cramera_protocol.live import Capability, LiveInfo, LiveRoute
from cramera_protocol.live_messages import MarkerTopics

from .conftest import dataset_body, live_request

EVIL_ORIGIN = "https://evil.example"
"""
A page of another site.
"""

EVERY_ADDRESS = "0.0.0.0"
"""
The address that listens on every network interface.
"""

# %% where and whom


def test_the_publisher_listens_on_this_machine_only(publisher):
    server = LiveServer(publisher)
    assert server.host == LIVE_HOST
    assert server.host in LOCAL_HOSTS
    assert server.port == LIVE_PORT


def test_a_server_refuses_to_listen_beyond_this_machine(publisher):
    server = LiveServer(publisher, host=EVERY_ADDRESS, port=0)
    with pytest.raises(NonLocalAddress):
        server.start()
    assert server.server is None


def test_a_started_server_is_bound_to_the_loopback_address(publisher):
    server = LiveServer(publisher, port=0).start()
    with urllib.request.urlopen(server.url() + LiveRoute.INFO) as answer:
        info = LiveInfo.from_json(json.loads(answer.read()))
    address = server.server.server.server_address[0]
    server.stop()
    assert address == LIVE_HOST
    assert info.running is True


def test_a_query_from_another_site_is_refused(answerer):
    request = live_request(
        HttpMethod.POST,
        LiveRoute.QUERY,
        dataset_body("query_robot_free.json"),
        **{Header.ORIGIN.value: EVIL_ORIGIN},
    )
    assert answerer.answer(request).status is HTTPStatus.FORBIDDEN


def test_a_local_page_may_read_the_answer(answerer):
    origin = "http://localhost:8711"
    answer = answerer.answer(
        live_request(HttpMethod.GET, LiveRoute.INFO, **{Header.ORIGIN.value: origin})
    )
    assert answer.headers[Header.ALLOW_ORIGIN] == origin


# %% routes


def test_the_marker_topics_can_be_read(answerer):
    answer = answerer.answer(live_request(HttpMethod.GET, LiveRoute.MARKER_TOPICS))
    assert answer.status is HTTPStatus.OK
    assert MarkerTopics.from_json(json.loads(answer.body)).ros is False


def test_changing_marker_topics_without_ros_is_a_conflict(answerer):
    answer = answerer.answer(
        live_request(
            HttpMethod.POST, LiveRoute.MARKER_TOPICS, dataset_body("marker_topic.json")
        )
    )
    assert answer.status is HTTPStatus.CONFLICT


def test_every_live_route_is_answered(answerer):
    unanswered = [
        route
        for route in LiveRoute
        if answerer.answer(live_request(HttpMethod.GET, route)).status
        is HTTPStatus.NOT_FOUND
        and answerer.answer(live_request(HttpMethod.POST, route)).status
        is HTTPStatus.NOT_FOUND
    ]
    assert unanswered == []


def test_a_body_that_is_no_json_is_refused(answerer):
    answer = answerer.answer(
        live_request(HttpMethod.POST, LiveRoute.MOVE, dataset_body("not_json.txt"))
    )
    assert answer.status is HTTPStatus.BAD_REQUEST


def test_the_information_lists_what_is_offered(publisher):
    capabilities = publisher.info().capabilities
    assert Capability.QUESTIONS in capabilities
    assert Capability.MARKERS not in capabilities
    assert Capability.TELEOPERATION not in capabilities
