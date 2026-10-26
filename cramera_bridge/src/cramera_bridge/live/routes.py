"""
The live publisher's routes, built from the viewer's HTTP building blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Callable, Optional, Type, TypeVar

from cramera.http.messages import HttpMethod, MediaType, Request, Response
from cramera.http.routing import ExactRoute, Router
from cramera_bridge.exceptions import InvalidParameter, MalformedBody, MissingParameter
from cramera_bridge.live.publisher import LivePublisher
from cramera_protocol.document import JsonDocument, JsonValue
from cramera_protocol.live import LiveRoute
from cramera_protocol.live_messages import (
    ConstraintRequest,
    JointRequest,
    LiveParameter,
    MarkerTopicRequest,
    MoveRequest,
    RecordingSaveRequest,
    RobotSelection,
    TeleoperationRequest,
)
from cramera_protocol.questions import (
    QueryRequest,
    QueryScope,
    QuestionRequest,
    QuestionsParameter,
)

DocumentType = TypeVar("DocumentType", bound=JsonDocument)
"""
The class of a request body.
"""

EnumerationType = TypeVar("EnumerationType", bound=Enum)
"""
The enumeration a query parameter's value belongs to.
"""

# %% requests


@dataclass(frozen=True)
class LiveRequest:
    """
    Reads the body and the query parameters of one request.
    """

    request: Request
    """
    The request.
    """

    def document(self, document_class: Type[DocumentType]) -> DocumentType:
        """
        :param document_class: The class of the body.
        :return: The body.
        :raises MalformedBody: When the body is not JSON.
        :raises ProtocolError: When the JSON does not describe such a document.
        """
        # the one place where a body that is no JSON at all becomes a refused request
        try:
            data = self.request.json()
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MalformedBody(str(error)) from error
        return document_class.from_json(data)

    def parameter(self, name: str) -> Optional[str]:
        """
        :param name: A query parameter.
        :return: Its first value, or None when absent.
        """
        values = self.request.parameters().get(name)
        return values[0] if values else None

    def required_parameter(self, name: str) -> str:
        """
        :param name: A query parameter.
        :return: Its first value.
        :raises MissingParameter: When absent.
        """
        value = self.parameter(name)
        if value is None:
            raise MissingParameter(name)
        return value

    def scope(self) -> QueryScope:
        """
        :return: The body of knowledge asked about, the current state by default.
        :raises InvalidParameter: For an unknown scope.
        """
        value = self.parameter(QuestionsParameter.SCOPE)
        if value is None:
            return QueryScope.CURRENT_STATE
        if value not in {scope.value for scope in QueryScope}:
            raise InvalidParameter(QuestionsParameter.SCOPE, value)
        return QueryScope(value)


# %% routes


@dataclass(frozen=True)
class LiveRoutes:
    """
    Builds the router answering every :class:`LiveRoute` from a publisher.
    """

    publisher: LivePublisher
    """
    The publisher the routes read and change.
    """

    def router(self) -> Router:
        """
        :return: The router of every live route.
        """
        publisher = self.publisher
        reads = {
            LiveRoute.INFO: publisher.info,
            LiveRoute.STATE: publisher.current_state,
            LiveRoute.ROBOTS: publisher.robots,
            LiveRoute.PLAN: publisher.current_plan,
            LiveRoute.STATECHART: publisher.current_statechart,
            LiveRoute.TRANSFORMS: publisher.transform_graph,
            LiveRoute.OBJECTS: publisher.object_catalog,
            LiveRoute.CAPTURED_OBJECTS: publisher.captured_objects,
            LiveRoute.SURFACES: publisher.surfaces,
            LiveRoute.LIVE_SCENE: publisher.live_scene,
            LiveRoute.MARKERS: publisher.marker_overlay,
            LiveRoute.MARKER_TOPICS: publisher.marker_topics,
            LiveRoute.PRESETS: publisher.presets,
            LiveRoute.RECORDING: publisher.recording_status,
        }
        actions = {
            LiveRoute.RECORDING_STOP: publisher.finish_recording,
            LiveRoute.RECORDING_DISCARD: publisher.discard_recording,
            LiveRoute.TELEOPERATION_STOP: publisher.stop_teleoperation,
        }
        changes = {
            LiveRoute.ROBOT: (RobotSelection, publisher.select_robot),
            LiveRoute.MARKER_TOPICS: (
                MarkerTopicRequest,
                publisher.change_marker_topic,
            ),
            LiveRoute.QUERY: (QueryRequest, publisher.answer_query),
            LiveRoute.QUESTION: (QuestionRequest, publisher.answer_question),
            LiveRoute.RECORDING_SAVE: (RecordingSaveRequest, publisher.save_recording),
            LiveRoute.CONSTRAINT: (ConstraintRequest, publisher.attach_constraint),
            LiveRoute.JOINT: (JointRequest, publisher.set_joint),
            LiveRoute.TELEOPERATION: (TeleoperationRequest, publisher.teleoperate),
            LiveRoute.MOVE: (MoveRequest, publisher.move),
        }
        router = Router()
        for route, read in reads.items():
            router.add(ExactRoute(HttpMethod.GET, route, self.reading(read)))
        for route, action in actions.items():
            router.add(ExactRoute(HttpMethod.POST, route, self.reading(action)))
        for route, (document_class, change) in changes.items():
            router.add(
                ExactRoute(
                    HttpMethod.POST, route, self.changing(document_class, change)
                )
            )
        router.add(ExactRoute(HttpMethod.GET, LiveRoute.MESH, self.answer_mesh))
        router.add(
            ExactRoute(
                HttpMethod.GET,
                LiveRoute.PERCEPTION_IMAGE,
                self.answer_perception_image,
            )
        )
        router.add(
            ExactRoute(HttpMethod.GET, LiveRoute.VOCABULARY, self.answer_vocabulary)
        )
        router.add(ExactRoute(HttpMethod.GET, LiveRoute.MEMBERS, self.answer_members))
        return router

    @staticmethod
    def reading(read: Callable[[], JsonValue]) -> Callable[[Request], Response]:
        """
        :param read: Answers a document.
        :return: A handler answering the document.
        """
        return lambda request: Response.of_json(read())

    @staticmethod
    def changing(
        document_class: Type[DocumentType],
        change: Callable[[DocumentType], JsonValue],
    ) -> Callable[[Request], Response]:
        """
        :param document_class: The class of the request body.
        :param change: Carries the request out and answers a document.
        :return: A handler reading the body and answering the document.
        """
        return lambda request: Response.of_json(
            change(LiveRequest(request).document(document_class))
        )

    def answer_mesh(self, request: Request) -> Response:
        """
        :param request: A request naming a served mesh and optionally a side file.
        :return: The file.
        """
        reader = LiveRequest(request)
        return Response.of_file(
            self.publisher.mesh_file(
                reader.required_parameter(LiveParameter.KEY),
                reader.parameter(LiveParameter.SIDE),
            )
        )

    def answer_perception_image(self, request: Request) -> Response:
        """
        :param request: A request naming an image a perception step took.
        :return: The image as a PNG file.
        """
        image = self.publisher.perception_image(
            LiveRequest(request).required_parameter(LiveParameter.KEY)
        )
        return Response(body=image.image.png, media_type=MediaType.PNG)

    def answer_vocabulary(self, request: Request) -> Response:
        """
        :param request: A request optionally naming a scope.
        :return: The names a query of the scope may use.
        """
        return Response.of_json(self.publisher.vocabulary(LiveRequest(request).scope()))

    def answer_members(self, request: Request) -> Response:
        """
        :param request: A request naming a name and optionally a scope.
        :return: The members following the name's dot.
        """
        reader = LiveRequest(request)
        return Response.of_json(
            self.publisher.members(
                reader.scope(), reader.required_parameter(QuestionsParameter.NAME)
            )
        )
