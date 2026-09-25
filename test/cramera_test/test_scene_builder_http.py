"""
The authoring API accepts bounded documents and confined resource requests.
"""

from http import HTTPStatus
import json

from cramera.scene_builder_http import SceneBuilderHttp, SceneBuilderRoute
from .test_scene_builder import builder_service


# %% request boundaries
def test_catalog_and_assets_use_the_same_confined_library(builder_service):
    handler = SceneBuilderHttp(service=builder_service)
    response = handler.get(SceneBuilderRoute.CATALOG, "")
    catalog = json.loads(response.body)
    assert response.status == HTTPStatus.OK
    assert catalog == builder_service.catalog()
    denied = handler.get(SceneBuilderRoute.ASSETS + "resources/../password", "")
    assert denied.status == HTTPStatus.BAD_REQUEST


def test_unknown_query_cannot_change_catalog_scope(builder_service):
    response = SceneBuilderHttp(service=builder_service).get(
        SceneBuilderRoute.CATALOG, "root=/etc"
    )
    assert response.status == HTTPStatus.BAD_REQUEST


def test_invalid_draft_payload_does_not_create_a_file(builder_service):
    response = SceneBuilderHttp(service=builder_service).post(
        SceneBuilderRoute.DRAFTS, b'{"document":{},"path":"/tmp/escape"}'
    )
    assert response.status == HTTPStatus.BAD_REQUEST
    assert builder_service.list_drafts() == []


def test_unrecognized_authoring_route_is_not_dispatched(builder_service):
    handler = SceneBuilderHttp(service=builder_service)
    assert not handler.handles("/api/plan/scaffold")
    assert handler.handles(SceneBuilderRoute.CATALOG)
    assert (
        handler.get(SceneBuilderRoute.PREFIX + "/unknown", "").status
        == HTTPStatus.NOT_FOUND
    )


def test_template_requires_exactly_one_scene_identifier(builder_service):
    response = SceneBuilderHttp(service=builder_service).get(
        SceneBuilderRoute.TEMPLATE, "asset=one&asset=two"
    )
    assert response.status == HTTPStatus.BAD_REQUEST


def test_malformed_urdf_returns_a_visible_error(builder_service):
    source = builder_service.roots["resources"] / "malformed.urdf"
    source.write_text("<robot>")
    response = SceneBuilderHttp(service=builder_service).get(
        SceneBuilderRoute.ASSETS + "resources/malformed.urdf", ""
    )
    assert response.status == HTTPStatus.BAD_REQUEST
    assert "error" in json.loads(response.body)
