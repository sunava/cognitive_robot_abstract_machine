"""Confined resource and document requests for scene authoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from urllib.parse import parse_qs
from xml.etree.ElementTree import ParseError

from typing_extensions import Any, ClassVar

from cramera.scene_builder import (
    InvalidSceneBuilderRequest,
    SceneBuilderService,
    MAXIMUM_DOCUMENT_BYTES,
)


# %% route and document vocabulary
class SceneBuilderRoute(StrEnum):
    """Operations offered by the standalone authoring workspace."""

    PREFIX = "/api/scene-builder"
    """Namespace owned by the authoring service."""
    CATALOG = "/api/scene-builder/catalog"
    """List models and source scenes."""
    TEMPLATE = "/api/scene-builder/template"
    """Convert a recorded scene into an editable document."""
    ASSETS = "/api/scene-builder/assets/"
    """Confined model and dependency files."""
    DRAFTS = "/api/scene-builder/drafts"
    """Saved scene documents."""
    PAGE = "/scene_builder.html"
    """Standalone editor page."""


class SceneBuilderField(StrEnum):
    """Fields of requests and responses at the authoring boundary."""

    IDENTIFIER = "id"
    """Existing draft identifier when updating a document."""
    DOCUMENT = "document"
    """Validated authoring document."""
    ASSET = "asset"
    """Catalog identifier of a source scene."""
    DRAFTS = "drafts"
    """Saved scene summaries."""
    ERROR = "error"
    """Human-readable request failure."""


@dataclass(frozen=True)
class SceneBuilderResponse:
    """One response independent of the local or protected HTTP listener."""

    status: HTTPStatus
    """Response status."""
    body: bytes
    """Serialized resource or JSON document."""
    content_type: str = "application/json"
    """Media type sent with the response."""

    @classmethod
    def document(
        cls, payload: Any, status: HTTPStatus = HTTPStatus.OK
    ) -> SceneBuilderResponse:
        """Serialize finite JSON data for a browser client."""
        return cls(status=status, body=json.dumps(payload, allow_nan=False).encode())


# %% common HTTP dispatch
@dataclass
class SceneBuilderHttp:
    """Dispatch authoring requests without exposing filesystem paths."""

    service: SceneBuilderService
    """Confined asset library and persistent draft store."""
    MAX_BODY_BYTES: ClassVar[int] = MAXIMUM_DOCUMENT_BYTES
    """Maximum serialized draft accepted by either listener."""

    @staticmethod
    def handles(route: str) -> bool:
        """Return whether a route belongs to the authoring namespace."""
        return route.startswith(SceneBuilderRoute.PREFIX + "/")

    @staticmethod
    def failure(error: Exception, status: HTTPStatus) -> SceneBuilderResponse:
        """Describe an invalid request without changing its stored document."""
        return SceneBuilderResponse.document(
            {SceneBuilderField.ERROR: str(error)}, status=status
        )

    def get(self, route: str, query: str) -> SceneBuilderResponse:
        """Read a declared model, source scene, or saved authoring document."""
        try:
            if route == SceneBuilderRoute.TEMPLATE:
                parameters = parse_qs(query, strict_parsing=True)
                if (
                    set(parameters) != {SceneBuilderField.ASSET}
                    or len(parameters[SceneBuilderField.ASSET]) != 1
                ):
                    raise InvalidSceneBuilderRequest("Select exactly one source scene")
                return SceneBuilderResponse.document(
                    self.service.load_scene(parameters[SceneBuilderField.ASSET][0])
                )
            if query:
                raise InvalidSceneBuilderRequest("This request accepts no query fields")
            if route == SceneBuilderRoute.CATALOG:
                return SceneBuilderResponse.document(self.service.catalog())
            if route == SceneBuilderRoute.DRAFTS:
                return SceneBuilderResponse.document(
                    {SceneBuilderField.DRAFTS: self.service.list_drafts()}
                )
            if route.startswith(SceneBuilderRoute.DRAFTS + "/"):
                return SceneBuilderResponse.document(
                    self.service.load_draft(
                        route.removeprefix(SceneBuilderRoute.DRAFTS + "/")
                    )
                )
            if route.startswith(SceneBuilderRoute.ASSETS):
                body, content_type = self.service.render_asset(
                    route.removeprefix(SceneBuilderRoute.ASSETS)
                )
                return SceneBuilderResponse(HTTPStatus.OK, body, content_type)
        except (ValueError, TypeError, ParseError) as error:
            return self.failure(error, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError:
            return self.failure(ValueError("Resource not found"), HTTPStatus.NOT_FOUND)
        return self.failure(ValueError("Unknown authoring route"), HTTPStatus.NOT_FOUND)

    def post(self, route: str, body: bytes) -> SceneBuilderResponse:
        """Save an explicit document inside the dedicated draft directory."""
        if route != SceneBuilderRoute.DRAFTS:
            return self.failure(
                ValueError("Unknown authoring route"), HTTPStatus.NOT_FOUND
            )
        if len(body) > self.MAX_BODY_BYTES:
            return self.failure(
                ValueError("Scene document is too large"),
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            payload = json.loads(body)
            if (
                not isinstance(payload, dict)
                or SceneBuilderField.DOCUMENT not in payload
                or set(payload)
                - {SceneBuilderField.IDENTIFIER, SceneBuilderField.DOCUMENT}
            ):
                raise InvalidSceneBuilderRequest(
                    "Expected a document and optional draft id"
                )
            return SceneBuilderResponse.document(
                self.service.save_draft(
                    payload.get(SceneBuilderField.IDENTIFIER),
                    payload[SceneBuilderField.DOCUMENT],
                )
            )
        except (ValueError, TypeError) as error:
            return self.failure(error, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError:
            return self.failure(ValueError("Draft not found"), HTTPStatus.NOT_FOUND)
