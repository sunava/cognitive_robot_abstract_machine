"""
Bounded point-cloud sources and independent persistent scene references.
"""

from __future__ import annotations

import hashlib
import struct

import pytest

from cramera.scene_builder import InvalidSceneBuilderRequest, SceneBuilderService
from cramera.scene_builder_point_cloud import InvalidPointCloud, PointCloudLimits

from .conftest import DATASET_DIR
from .test_scene_builder import builder_service


# %% representative source encodings
@pytest.fixture
def cloud_payloads():
    """
    Encode the same three colored samples in each supported source format.
    """
    directory = DATASET_DIR / "scene_builder_point_clouds"
    polygon = (directory / "colored.ply").read_bytes()
    cloud = (directory / "colored.pcd").read_bytes()
    polygon_header = polygon.split(b"end_header\n")[0] + b"end_header\n"
    cloud_header = cloud.split(b"DATA ascii\n")[0] + b"DATA binary\n"
    colors = [(0, 0, 0, 255, 0, 0), (1, 0, 0, 0, 255, 0), (0, 1, 0, 0, 0, 255)]
    return {
        "ply_ascii": ("ply", polygon),
        "ply_little": (
            "ply",
            polygon_header.replace(b"format ascii", b"format binary_little_endian")
            + b"".join(struct.pack("<fffBBB", *point) for point in colors),
        ),
        "ply_big": (
            "ply",
            polygon_header.replace(b"format ascii", b"format binary_big_endian")
            + b"".join(struct.pack(">fffBBB", *point) for point in colors),
        ),
        "pcd_ascii": ("pcd", cloud),
        "pcd_binary": (
            "pcd",
            cloud_header
            + b"".join(
                struct.pack("<fffI", *point)
                for point in [(0, 0, 0, 16711680), (1, 0, 0, 65280), (0, 1, 0, 255)]
            ),
        ),
    }


@pytest.mark.parametrize(
    "encoding", ["ply_ascii", "ply_little", "ply_big", "pcd_ascii", "pcd_binary"]
)
def test_supported_cloud_uploads_keep_original_bytes(
    builder_service, cloud_payloads, encoding
):
    """
    Accepted samples remain unchanged beneath deterministic resource IDs.
    """
    format_name, body = cloud_payloads[encoding]
    asset = builder_service.import_point_cloud(format_name, body)
    expected = f"point_clouds/{hashlib.sha256(body).hexdigest()}.{format_name}"
    assert asset["id"] == expected
    assert asset["kind"] == asset["category"] == "point-cloud"
    assert asset["metadata"]["pointCount"] == 3
    assert asset["metadata"]["finitePointCount"] == 3
    assert builder_service.render_asset(expected)[0] == body


@pytest.mark.parametrize(
    "filename",
    [
        "colored_ascii.ply",
        "colored_binary_le.ply",
        "colored_binary_be.ply",
        "colored_ascii.pcd",
        "colored_binary.pcd",
    ],
)
def test_browser_and_backend_share_valid_cloud_fixtures(builder_service, filename):
    """
    Frontend fixtures exercise the same accepted encodings at upload time.
    """
    path = DATASET_DIR / "point_clouds" / filename
    asset = builder_service.import_point_cloud(path.suffix[1:], path.read_bytes())
    assert asset["metadata"]["pointCount"] == 3
    assert asset["metadata"]["finitePointCount"] == 3


def test_upload_survives_restart_and_matches_catalog(builder_service, cloud_payloads):
    """
    Persistent scans retain their metadata across service reconstruction.
    """
    imported = builder_service.import_point_cloud(*cloud_payloads["ply_ascii"])
    restarted = SceneBuilderService(
        roots={
            name: path
            for name, path in builder_service.roots.items()
            if name != "point_clouds"
        },
        drafts_directory=builder_service.drafts_directory,
        url_prefix=builder_service.url_prefix,
    )
    catalog = {item["id"]: item for item in restarted.catalog()["assets"]}
    assert catalog[imported["id"]] == imported
    assert (
        restarted.resolve_asset(imported["id"]).read_bytes()
        == cloud_payloads["ply_ascii"][1]
    )


def test_duplicate_upload_reuses_the_same_asset(builder_service, cloud_payloads):
    """
    Repeated upload of source bytes does not create another visual resource.
    """
    first = builder_service.import_point_cloud(*cloud_payloads["pcd_ascii"])
    second = builder_service.import_point_cloud(*cloud_payloads["pcd_ascii"])
    assert first == second
    assert [
        item["id"]
        for item in builder_service.catalog()["assets"]
        if item["root"] == "point_clouds"
    ] == [first["id"]]


def test_existing_point_cloud_resources_preserve_notices(
    builder_service, cloud_payloads
):
    """
    Scans already in the local library retain their original attribution.
    """
    (builder_service.roots["resources"] / "scan.ply").write_bytes(
        cloud_payloads["ply_ascii"][1]
    )
    asset = next(
        item
        for item in builder_service.catalog()["assets"]
        if item["id"] == "resources/scan.ply"
    )
    assert asset["kind"] == asset["category"] == "point-cloud"
    assert asset["notices"] == [
        {
            "name": "NOTICE.md",
            "url": "/workshop/api/scene-builder/assets/resources/NOTICE.md",
        }
    ]


# %% malformed or excessive source bodies
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_xyz",
        "truncated",
        "extra_point",
        "infinite",
        "overflow",
        "duplicate_field",
        "point_limit",
        "not_a_cloud",
    ],
)
def test_invalid_polygon_upload_never_creates_a_file(
    builder_service, cloud_payloads, mutation
):
    """
    Malformed declarations and point records fail before storage starts.
    """
    _, body = cloud_payloads["ply_ascii"]
    if mutation == "missing_xyz":
        body = body.replace(b"property float z", b"property float depth")
    elif mutation == "truncated":
        body = body.rsplit(b"\n", 2)[0] + b"\n"
    elif mutation == "extra_point":
        body += b"1 2 3 0 0 0\n"
    elif mutation == "infinite":
        body = body.replace(b"0 0 0 255", b"inf 0 0 255")
    elif mutation == "overflow":
        body = body.replace(b"0 0 0 255", b"1e100 0 0 255")
    elif mutation == "duplicate_field":
        body = body.replace(b"property uchar red", b"property uchar x")
    elif mutation == "point_limit":
        body = body.replace(
            b"element vertex 3",
            f"element vertex {PointCloudLimits.MAXIMUM_POINTS + 1}".encode(),
        )
    else:
        body = b"This is not a point-cloud file."
    with pytest.raises(InvalidPointCloud):
        builder_service.import_point_cloud("ply", body)
    assert not builder_service.roots["point_clouds"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "count_mismatch",
        "compressed",
        "missing_xyz",
        "truncated",
        "duplicate_header",
        "unknown_encoding",
    ],
)
def test_malformed_cloud_data_upload_is_rejected(
    builder_service, cloud_payloads, mutation
):
    """
    PCD dimensions and storage declarations must match the complete source.
    """
    _, body = cloud_payloads["pcd_ascii"]
    if mutation == "count_mismatch":
        body = body.replace(b"WIDTH 3", b"WIDTH 4")
    elif mutation == "compressed":
        body = body.replace(b"DATA ascii", b"DATA binary_compressed")
    elif mutation == "missing_xyz":
        body = body.replace(b"FIELDS x y z rgb", b"FIELDS x y depth rgb")
    elif mutation == "truncated":
        body = body.rsplit(b"\n", 2)[0] + b"\n"
    elif mutation == "duplicate_header":
        body = body.replace(b"WIDTH 3", b"WIDTH 3\nWIDTH 3")
    else:
        body = body.replace(b"DATA ascii", b"DATA yaml")
    with pytest.raises(InvalidPointCloud):
        builder_service.import_point_cloud("pcd", body)


@pytest.mark.parametrize("encoding", ["ply_little", "ply_big", "pcd_binary"])
def test_binary_payload_length_matches_declared_points(
    builder_service, cloud_payloads, encoding
):
    """
    A partial binary record cannot become a supposedly complete source scan.
    """
    format_name, body = cloud_payloads[encoding]
    with pytest.raises(InvalidPointCloud):
        builder_service.import_point_cloud(format_name, body[:-1])


def test_missing_measurements_remain_when_usable_points_exist(
    builder_service, cloud_payloads
):
    """
    NaN coordinates remain in the original source while usable points are counted.
    """
    _, body = cloud_payloads["pcd_ascii"]
    body = body.replace(b"0 0 0 16711680", b"nan nan nan 16711680")
    asset = builder_service.import_point_cloud("pcd", body)
    assert asset["metadata"]["finitePointCount"] == 2
    assert builder_service.render_asset(asset["id"])[0] == body


def test_upload_byte_limit_precedes_storage(
    builder_service, cloud_payloads, monkeypatch
):
    """
    The service enforces its byte cap independently of transport validation.
    """
    _, body = cloud_payloads["ply_ascii"]
    monkeypatch.setattr(PointCloudLimits, "MAXIMUM_BYTES", len(body) - 1)
    with pytest.raises(InvalidPointCloud):
        builder_service.import_point_cloud("ply", body)
    assert not builder_service.roots["point_clouds"].exists()


def test_upload_directory_cannot_redirect_writes(
    builder_service, cloud_payloads, tmp_path
):
    """
    A late directory symlink cannot send uploads outside the draft directory.
    """
    builder_service.drafts_directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    builder_service.roots["point_clouds"].symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.import_point_cloud(*cloud_payloads["ply_ascii"])
    assert list(outside.iterdir()) == []


# %% optional reference appearance


def test_point_cloud_settings_round_trip(builder_service, cloud_payloads):
    """
    Visibility, editing lock and appearance persist with model transforms.
    """
    asset = builder_service.import_point_cloud(*cloud_payloads["ply_ascii"])
    document = builder_service.load_scene("scenes/fixture/scene.json")
    instance = document["instances"][0]
    instance["assetId"] = asset["id"]
    instance["pointCloud"] = {
        "visible": True,
        "locked": True,
        "pointSize": 0.006,
        "opacity": 0.65,
    }
    saved = builder_service.save_draft(None, document)
    assert (
        builder_service.load_draft(saved["id"])["document"]["instances"][0][
            "pointCloud"
        ]
        == instance["pointCloud"]
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("visible", 1),
        ("locked", "yes"),
        ("pointSize", float("nan")),
        ("pointSize", 0),
        ("pointSize", 0.2),
        ("opacity", 0),
        ("opacity", 1.1),
    ],
)
def test_invalid_point_cloud_settings_are_rejected(builder_service, setting, value):
    """
    Reference settings reject ambiguous types, nonfinite values and excess sizes.
    """
    document = builder_service.load_scene("scenes/fixture/scene.json")
    document["instances"][0]["pointCloud"] = {
        "visible": True,
        "locked": True,
        "pointSize": 0.006,
        "opacity": 0.65,
        setting: value,
    }
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.save_draft(None, document)


def test_ordinary_instances_gain_no_cloud_settings(builder_service):
    """
    Existing documents remain unchanged when scan references are unused.
    """
    document = builder_service.load_scene("scenes/fixture/scene.json")
    saved = builder_service.save_draft(None, document)
    assert all(
        "pointCloud" not in instance for instance in saved["document"]["instances"]
    )
