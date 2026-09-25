"""
The authoring API accepts bounded documents and confined resource requests.
"""

from http import HTTPStatus
import json
from pathlib import Path

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


# %% persistent reference transport
def test_uploaded_scan_can_be_loaded_through_the_asset_library(builder_service):
    source = Path(__file__).parent / "dataset/scene_builder/transport_cloud.ply"
    handler = SceneBuilderHttp(service=builder_service)
    response = handler.post(SceneBuilderRoute.POINT_CLOUDS + "ply", source.read_bytes())
    assert response.status == HTTPStatus.OK
    asset = json.loads(response.body)
    stored = handler.get(SceneBuilderRoute.ASSETS + asset["id"], "")
    assert stored.status == HTTPStatus.OK
    assert stored.body == source.read_bytes()
    assert (
        asset in json.loads(handler.get(SceneBuilderRoute.CATALOG, "").body)["assets"]
    )


def test_invalid_scan_upload_is_reported_without_creating_an_asset(builder_service):
    handler = SceneBuilderHttp(service=builder_service)
    before = builder_service.catalog()
    response = handler.post(SceneBuilderRoute.POINT_CLOUDS + "ply", b"not a scan")
    assert response.status == HTTPStatus.BAD_REQUEST
    assert builder_service.catalog() == before


def test_scan_upload_has_an_independent_bound_from_layout_json(
    builder_service, monkeypatch
):
    source = Path(__file__).parent / "dataset/scene_builder/transport_cloud.ply"
    monkeypatch.setattr(SceneBuilderHttp, "MAX_BODY_BYTES", 1)
    handler = SceneBuilderHttp(service=builder_service)
    assert (
        handler.post(SceneBuilderRoute.POINT_CLOUDS + "ply", source.read_bytes()).status
        == HTTPStatus.OK
    )
    assert (
        handler.post(SceneBuilderRoute.DRAFTS, b"{}").status
        == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    )
