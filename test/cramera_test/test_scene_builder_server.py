"""
Local authoring writes remain separate from robot execution endpoints.
"""

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cramera.scene_builder_http import SceneBuilderRoute
from .test_server import server


# %% local document transport
def test_local_editor_saves_and_reopens_an_empty_layout(server):
    document = {"version": 1, "name": "Reusable workcell", "instances": []}
    request = Request(
        server + SceneBuilderRoute.DRAFTS,
        data=json.dumps({"document": document}).encode(),
        headers={"Origin": server, "Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        saved = json.load(response)
    with urlopen(server + SceneBuilderRoute.DRAFTS + "/" + saved["id"]) as response:
        assert json.load(response) == saved
    assert saved["document"]["name"] == document["name"]


def test_local_editor_rejects_cross_origin_saves(server):
    request = Request(
        server + SceneBuilderRoute.DRAFTS,
        data=b"{}",
        headers={
            "Origin": "https://untrusted.example",
            "Content-Type": "application/json",
        },
    )
    with pytest.raises(HTTPError) as failure:
        urlopen(request)
    assert failure.value.code == 403


def test_local_catalog_rejects_a_foreign_host_header(server):
    request = Request(
        server + SceneBuilderRoute.CATALOG,
        headers={"Host": "untrusted.example"},
    )
    with pytest.raises(HTTPError) as failure:
        urlopen(request)
    assert failure.value.code == 403


# %% local reference uploads
def test_local_scan_upload_can_be_read_back(server):
    source = Path(__file__).parent / "dataset/scene_builder/transport_cloud.ply"
    request = Request(
        server + SceneBuilderRoute.POINT_CLOUDS + "ply",
        data=source.read_bytes(),
        headers={"Origin": server, "Content-Type": "application/octet-stream"},
    )
    with urlopen(request) as response:
        asset = json.load(response)
    with urlopen(server + asset["url"]) as response:
        assert response.read() == source.read_bytes()


def test_local_scan_upload_rejects_cross_origin_requests(server):
    request = Request(
        server + SceneBuilderRoute.POINT_CLOUDS + "ply",
        data=b"not a scan",
        headers={
            "Origin": "https://untrusted.example",
            "Content-Type": "application/octet-stream",
        },
    )
    with pytest.raises(HTTPError) as failure:
        urlopen(request)
    assert failure.value.code == 403
