"""Bounded resource discovery and editable scene drafts."""

from __future__ import annotations

import json
import hashlib
import math
import mimetypes
import os
import posixpath
import re
import tempfile
import uuid
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from cramera import paths
from cramera.scene_builder_point_cloud import (
    PointCloudField,
    PointCloudFormat,
    PointCloudLimits,
    PointCloudSource,
)


# %% resource and document vocabulary
class BuilderField(StrEnum):
    """Fields shared with the browser's editable document format."""

    ID = "id"
    NAME = "name"
    VERSION = "version"
    INSTANCES = "instances"
    ASSET_ID = "assetId"
    POSITION = "position"
    ROTATION = "rotation"
    SCALE = "scale"
    LINK_OFFSETS = "linkOffsets"
    JOINT_POSITIONS = "jointPositions"
    POINT_CLOUD = "pointCloud"
    NOTICES = "notices"
    URL = "url"
    SOURCE_SCENE = "sourceScene"
    DOCUMENT = "document"
    UPDATED_AT = "updatedAt"
    ASSETS = "assets"
    SCENES = "scenes"
    DRAFTS = "drafts"
    FORMAT = "format"
    KIND = "kind"
    CATEGORY = "category"
    METADATA = "metadata"
    ROOT = "root"
    PATH = "path"


class ResourceKind(StrEnum):
    """Kinds accepted by the browser's model loaders."""

    URDF = "urdf"
    MESH = "mesh"
    ROBOT = "robot"
    ENVIRONMENT = "environment"
    OBJECT = "object"
    POINT_CLOUD = "point-cloud"


class ResourceFile(StrEnum):
    """Well-known description and attribution documents."""

    SCENE = "scene.json"
    NOTICE = "NOTICE.md"
    LICENSE = "LICENSE.txt"
    LICENSE_BARE = "LICENSE"
    LICENSE_MARKDOWN = "LICENSE.md"
    COPYING = "COPYING"
    RIGHTS = "rights.json"
    PACKAGE = "package.xml"


MODEL_SUFFIXES = frozenset({".urdf", ".glb", ".gltf", ".stl", ".obj", ".dae"})
"""Model formats supported by the existing browser loaders."""
POINT_CLOUD_SUFFIXES = frozenset(f".{format_name}" for format_name in PointCloudFormat)
"""Point-cloud source formats shown as spatial references."""
RESOURCE_SUFFIXES = MODEL_SUFFIXES | POINT_CLOUD_SUFFIXES
"""Visual resources accepted as editable scene instances."""
DEPENDENCY_SUFFIXES = frozenset(
    {
        ".mtl",
        ".bin",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tga",
        ".bmp",
        ".gif",
        ".svg",
        ".dds",
        ".ktx2",
    }
)
"""Material, buffer and texture formats permitted under configured roots."""
NOTICE_NAMES = frozenset(
    {
        ResourceFile.NOTICE,
        ResourceFile.LICENSE,
        ResourceFile.LICENSE_BARE,
        ResourceFile.LICENSE_MARKDOWN,
        ResourceFile.COPYING,
        ResourceFile.RIGHTS,
    }
)
"""Attribution documents retained in resource catalogs and saved layouts."""
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
"""Names allowed as configured roots, draft IDs and instance IDs."""
LAYOUT_VERSION = 1
"""Browser scene document format version."""
MAXIMUM_INSTANCES = 500
"""Maximum number of editable instances in one draft."""
MAXIMUM_DOCUMENT_BYTES = 2 * 1024 * 1024
"""Maximum serialized draft size."""
MAXIMUM_COORDINATE = 10000.0
"""Finite scene coordinates and joint values supported by the editor."""
MAXIMUM_SCALE = 1000.0
"""Largest allowed positive object scale."""
IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
"""Origin pose in x, y, z, quaternion x, y, z, w order."""


class InvalidSceneBuilderRequest(ValueError):
    """A resource identifier or editable document violates its allowed boundary."""


# %% finite external document validation
@dataclass
class SceneDocument:
    """Validate the external JSON document shared with the browser editor."""

    value: dict
    """Normalized document ready for JSON serialization."""

    @classmethod
    def from_json(cls, payload: object) -> SceneDocument:
        """Validate editable transforms while preserving source attribution."""
        if not isinstance(payload, dict):
            raise InvalidSceneBuilderRequest("A scene document must be an object.")
        allowed = {
            BuilderField.VERSION,
            BuilderField.NAME,
            BuilderField.INSTANCES,
            BuilderField.SOURCE_SCENE,
            BuilderField.NOTICES,
        }
        if (
            set(payload) - allowed
            or payload.get(BuilderField.VERSION) != LAYOUT_VERSION
        ):
            raise InvalidSceneBuilderRequest(
                "Unsupported scene document fields or version."
            )
        name = cls._text(payload.get(BuilderField.NAME), maximum=160)
        instances = payload.get(BuilderField.INSTANCES)
        if not isinstance(instances, list) or len(instances) > MAXIMUM_INSTANCES:
            raise InvalidSceneBuilderRequest(
                "A scene requires a bounded list of instances."
            )
        normalized = [cls._instance(item) for item in instances]
        identifiers = [item[BuilderField.ID] for item in normalized]
        if len(identifiers) != len(set(identifiers)):
            raise InvalidSceneBuilderRequest("Instance identifiers must be unique.")
        result = {
            BuilderField.VERSION: LAYOUT_VERSION,
            BuilderField.NAME: name,
            BuilderField.INSTANCES: normalized,
        }
        if BuilderField.SOURCE_SCENE in payload:
            result[BuilderField.SOURCE_SCENE] = cls._text(
                payload[BuilderField.SOURCE_SCENE], maximum=1024
            )
        notices = payload.get(BuilderField.NOTICES, [])
        if not isinstance(notices, list) or len(notices) > MAXIMUM_INSTANCES:
            raise InvalidSceneBuilderRequest("Invalid notice list.")
        result[BuilderField.NOTICES] = [cls._notice(item) for item in notices]
        if len(json.dumps(result, allow_nan=False).encode()) > MAXIMUM_DOCUMENT_BYTES:
            raise InvalidSceneBuilderRequest(
                "Scene document exceeds the draft size limit."
            )
        return cls(value=result)

    @classmethod
    def _instance(cls, payload: object) -> dict:
        """Normalize a single model's transform and articulation overrides."""
        allowed = {
            BuilderField.ID,
            BuilderField.ASSET_ID,
            BuilderField.NAME,
            BuilderField.POSITION,
            BuilderField.ROTATION,
            BuilderField.SCALE,
            BuilderField.LINK_OFFSETS,
            BuilderField.JOINT_POSITIONS,
            BuilderField.POINT_CLOUD,
        }
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise InvalidSceneBuilderRequest("Invalid scene instance fields.")
        identifier = cls._text(payload.get(BuilderField.ID), maximum=96)
        if not SAFE_NAME.fullmatch(identifier):
            raise InvalidSceneBuilderRequest("Invalid scene instance identifier.")
        scale = cls._vector(payload.get(BuilderField.SCALE, [1, 1, 1]))
        if any(value <= 0 or value > MAXIMUM_SCALE for value in scale):
            raise InvalidSceneBuilderRequest(
                "Object scale must be positive and bounded."
            )
        offsets = payload.get(BuilderField.LINK_OFFSETS, {})
        joints = payload.get(BuilderField.JOINT_POSITIONS, {})
        if (
            not isinstance(offsets, dict)
            or not isinstance(joints, dict)
            or len(offsets) > MAXIMUM_INSTANCES
            or len(joints) > MAXIMUM_INSTANCES
        ):
            raise InvalidSceneBuilderRequest("Invalid articulation overrides.")
        normalized = {
            BuilderField.ID: identifier,
            BuilderField.ASSET_ID: cls._text(
                payload.get(BuilderField.ASSET_ID), maximum=2048
            ),
            BuilderField.NAME: cls._text(payload.get(BuilderField.NAME), maximum=160),
            BuilderField.POSITION: cls._vector(payload.get(BuilderField.POSITION)),
            BuilderField.ROTATION: cls._vector(payload.get(BuilderField.ROTATION)),
            BuilderField.SCALE: scale,
            BuilderField.LINK_OFFSETS: {
                cls._text(key, maximum=256): cls._offset(value)
                for key, value in offsets.items()
            },
            BuilderField.JOINT_POSITIONS: {
                cls._text(key, maximum=256): cls._number(value)
                for key, value in joints.items()
            },
        }
        if BuilderField.POINT_CLOUD in payload:
            normalized[BuilderField.POINT_CLOUD] = cls._point_cloud(
                payload[BuilderField.POINT_CLOUD]
            )
        return normalized

    @classmethod
    def _point_cloud(cls, payload: object) -> dict:
        """Validate optional scan visibility, editing lock and display appearance."""
        defaults = {
            PointCloudField.VISIBLE: True,
            PointCloudField.LOCKED: True,
            PointCloudField.POINT_SIZE: PointCloudLimits.DEFAULT_POINT_SIZE,
            PointCloudField.OPACITY: PointCloudLimits.DEFAULT_OPACITY,
        }
        if not isinstance(payload, dict) or set(payload) - defaults.keys():
            raise InvalidSceneBuilderRequest("Invalid point-cloud display settings.")
        result = defaults | payload
        if any(
            not isinstance(result[key], bool)
            for key in (PointCloudField.VISIBLE, PointCloudField.LOCKED)
        ):
            raise InvalidSceneBuilderRequest(
                "Point-cloud visibility and lock must be booleans."
            )
        size = cls._number(result[PointCloudField.POINT_SIZE])
        opacity = cls._number(result[PointCloudField.OPACITY])
        if (
            not PointCloudLimits.MINIMUM_POINT_SIZE
            <= size
            <= PointCloudLimits.MAXIMUM_POINT_SIZE
            or not PointCloudLimits.MINIMUM_OPACITY
            <= opacity
            <= PointCloudLimits.MAXIMUM_OPACITY
        ):
            raise InvalidSceneBuilderRequest(
                "Point-cloud appearance exceeds its supported bounds."
            )
        result[PointCloudField.POINT_SIZE] = size
        result[PointCloudField.OPACITY] = opacity
        return result

    @classmethod
    def _offset(cls, payload: object) -> dict:
        """Validate a parent-frame delta for an authored link transform."""
        if not isinstance(payload, dict) or set(payload) != {
            BuilderField.POSITION,
            BuilderField.ROTATION,
        }:
            raise InvalidSceneBuilderRequest(
                "A link offset requires position and rotation."
            )
        return {
            BuilderField.POSITION: cls._vector(payload[BuilderField.POSITION]),
            BuilderField.ROTATION: cls._vector(payload[BuilderField.ROTATION]),
        }

    @classmethod
    def _notice(cls, payload: object) -> dict:
        """Retain the name and safe local URL of an attribution document."""
        if not isinstance(payload, dict) or set(payload) != {
            BuilderField.NAME,
            BuilderField.URL,
        }:
            raise InvalidSceneBuilderRequest("Invalid resource notice.")
        url = cls._text(payload[BuilderField.URL], maximum=2048)
        if urlsplit(url).scheme or urlsplit(url).netloc or "\\" in url:
            raise InvalidSceneBuilderRequest("Notices must reference local resources.")
        return {
            BuilderField.NAME: cls._text(payload[BuilderField.NAME], maximum=160),
            BuilderField.URL: url,
        }

    @staticmethod
    def _text(value: object, maximum: int) -> str:
        """Require a nonempty bounded string without control characters."""
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > maximum
            or any(ord(character) < 32 for character in value)
        ):
            raise InvalidSceneBuilderRequest("Invalid text field.")
        return value

    @staticmethod
    def _number(value: object) -> float:
        """Require a finite bounded real number, excluding booleans."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or abs(value) > MAXIMUM_COORDINATE
        ):
            raise InvalidSceneBuilderRequest(
                "Transform values must be finite bounded numbers."
            )
        return float(value)

    @classmethod
    def _vector(cls, value: object) -> list[float]:
        """Read an explicit three-component transform vector."""
        if not isinstance(value, list) or len(value) != 3:
            raise InvalidSceneBuilderRequest(
                "Transform vectors require three components."
            )
        return [cls._number(component) for component in value]


# %% bounded resource library
@dataclass
class SceneBuilderService:
    """Discover configured model resources and store independent scene drafts."""

    roots: Mapping[str, Path] | None = None
    """Explicit named filesystem boundaries; defaults to local workspace resources."""
    drafts_directory: Path | None = None
    """Writable layout directory separate from source bundles."""
    url_prefix: str = "/api/scene-builder/assets/"
    """Same-origin URL prefix including any authenticated server mount."""
    _package_files: list[Path] | None = field(default=None, init=False, repr=False)
    """Allowed files cached for package references until the catalog is refreshed."""

    def __post_init__(self) -> None:
        """Freeze canonical resource roots without creating or altering them."""
        source_roots = self._default_roots() if self.roots is None else self.roots
        if any(not SAFE_NAME.fullmatch(name) for name in source_roots):
            raise InvalidSceneBuilderRequest("Invalid configured resource root name.")
        self.roots = {
            name: Path(directory).expanduser().resolve()
            for name, directory in source_roots.items()
        }
        self.drafts_directory = (
            Path(
                self.drafts_directory
                or paths.data_directory() / "scene-builder" / "drafts"
            )
            .expanduser()
            .resolve()
        )
        uploads = self.drafts_directory / PointCloudField.DIRECTORY
        if uploads.is_symlink() or (
            PointCloudField.ROOT in self.roots
            and self.roots[PointCloudField.ROOT] != uploads
        ):
            raise InvalidSceneBuilderRequest(
                "The point_clouds root is reserved for persistent uploads."
            )
        self.roots[PointCloudField.ROOT] = uploads
        if (
            not self.url_prefix.startswith("/")
            or self.url_prefix.startswith("//")
            or not self.url_prefix.endswith("/")
        ):
            raise InvalidSceneBuilderRequest(
                "Asset URLs require an absolute same-origin prefix."
            )

    @staticmethod
    def _default_roots() -> dict[str, Path]:
        """Use stable library names shared by local and protected deployments."""
        return {
            "resources": Path(__file__).resolve().parent / "resources",
            "scenes": paths.scenes_directory(),
            "local_scenes": paths.local_scenes_directory(),
            "coraplex": paths.repository_root() / "coraplex" / "resources",
        }

    def _resolve(self, identifier: str, *, asset: bool = False) -> Path:
        """Resolve a root-relative ID while rejecting traversal and symlink escapes."""
        if (
            not isinstance(identifier, str)
            or "\\" in identifier
            or "\x00" in identifier
            or unquote(identifier) != identifier
        ):
            raise InvalidSceneBuilderRequest("Invalid resource identifier.")
        segments = identifier.split("/")
        if len(segments) < 2 or any(
            segment in {"", ".", ".."} or segment.startswith(".")
            for segment in segments
        ):
            raise InvalidSceneBuilderRequest(
                "Resource identifiers must be root-relative paths."
            )
        root = self.roots.get(segments[0])
        if root is None:
            raise InvalidSceneBuilderRequest("Unknown resource root.")
        target = root.joinpath(*segments[1:]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise InvalidSceneBuilderRequest(
                "Resource is missing or outside its allowed root."
            )
        if (
            asset
            and target.suffix.lower() not in RESOURCE_SUFFIXES | DEPENDENCY_SUFFIXES
            and target.name not in NOTICE_NAMES
        ):
            raise InvalidSceneBuilderRequest(
                "This file is not a supported model dependency."
            )
        return target

    def resolve_asset(self, identifier: str) -> Path:
        """Return an allowed model, texture, buffer or attribution file."""
        return self._resolve(identifier, asset=True)

    def _identifier(self, path: Path) -> str:
        """Find a stable configured-root identifier for an existing dependency."""
        canonical = path.resolve()
        for name, root in self.roots.items():
            if canonical.is_relative_to(root) and canonical.is_file():
                identifier = f"{name}/{canonical.relative_to(root).as_posix()}"
                self.resolve_asset(identifier)
                return identifier
        raise InvalidSceneBuilderRequest(
            "A model dependency is outside configured resources."
        )

    def _url(self, identifier: str) -> str:
        """Encode an already validated ID beneath the protected asset endpoint."""
        return self.url_prefix + quote(identifier, safe="/")

    def _files(self) -> Iterator[tuple[str, Path]]:
        """Yield regular non-hidden files without following directory symlinks."""
        for root_name, root in self.roots.items():
            if not root.is_dir():
                continue
            for directory, subdirectories, names in os.walk(root, followlinks=False):
                subdirectories[:] = sorted(
                    name for name in subdirectories if not name.startswith(".")
                )
                for name in sorted(names):
                    candidate = Path(directory) / name
                    if (
                        name.startswith(".")
                        or not candidate.resolve().is_relative_to(root)
                        or not candidate.is_file()
                    ):
                        continue
                    yield f"{root_name}/{candidate.relative_to(root).as_posix()}", candidate

    def _notices(self, identifier: str) -> list[dict]:
        """Collect nearby attribution documents through the configured root."""
        root = self.roots[identifier.split("/", 1)[0]]
        directory = self.resolve_asset(identifier).parent
        result = []
        while directory.is_relative_to(root):
            for name in sorted(NOTICE_NAMES):
                candidate = directory / name
                if candidate.is_file() and candidate.resolve().is_relative_to(root):
                    result.append(
                        {
                            BuilderField.NAME: str(name),
                            BuilderField.URL: self._url(self._identifier(candidate)),
                        }
                    )
            if directory == root:
                break
            directory = directory.parent
        return result

    def catalog(self) -> dict:
        """List available assets, initial scene templates and independent drafts."""
        files = list(self._files())
        self._package_files = [path for _, path in files]
        scenes = []
        categories = {}
        for identifier, path in files:
            if path.name != ResourceFile.SCENE:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                continue
            scenes.append(
                {
                    BuilderField.ID: identifier,
                    BuilderField.NAME: document.get("environmentName")
                    or document.get(BuilderField.NAME)
                    or path.parent.name,
                }
            )
            for model in document.get("models", []):
                if isinstance(model, dict) and isinstance(model.get("urdf"), str):
                    categories[(path.parent / model["urdf"]).resolve()] = (
                        ResourceKind.ROBOT
                        if model.get("robot")
                        else ResourceKind.ENVIRONMENT
                    )
        assets = []
        for identifier, path in files:
            if path.suffix.lower() not in RESOURCE_SUFFIXES:
                continue
            category = categories.get(path.resolve(), ResourceKind.OBJECT)
            if "worlds" in path.parts:
                category = ResourceKind.ENVIRONMENT
            if any(part in {"robots", "robot_descriptions"} for part in path.parts):
                category = ResourceKind.ROBOT
            assets.append(self._asset(identifier, path, category=category))
        return {
            BuilderField.ASSETS: assets,
            BuilderField.SCENES: scenes,
            BuilderField.DRAFTS: self.list_drafts(),
        }

    def _asset(
        self,
        identifier: str,
        path: Path,
        *,
        category: ResourceKind = ResourceKind.OBJECT,
    ) -> dict:
        """Describe one local visual resource and its original attribution."""
        kind = (
            ResourceKind.URDF if path.suffix.lower() == ".urdf" else ResourceKind.MESH
        )
        metadata = {}
        if path.suffix.lower() in POINT_CLOUD_SUFFIXES:
            kind = category = ResourceKind.POINT_CLOUD
            if identifier.startswith(PointCloudField.ROOT + "/"):
                metadata_path = path.with_suffix(PointCloudField.METADATA_SUFFIX)
                if metadata_path.is_file() and not metadata_path.is_symlink():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            BuilderField.ID: identifier,
            BuilderField.NAME: (
                path.parent.name
                if path.stem in {"model", "environment", "robot"}
                else path.stem
            ),
            BuilderField.FORMAT: path.suffix.lower().removeprefix("."),
            BuilderField.KIND: kind,
            BuilderField.CATEGORY: category,
            BuilderField.URL: self._url(identifier),
            BuilderField.ROOT: identifier.split("/", 1)[0],
            BuilderField.PATH: identifier.split("/", 1)[1],
            BuilderField.METADATA: metadata,
            BuilderField.NOTICES: self._notices(identifier),
        }

    def import_point_cloud(self, format_name: str, body: bytes) -> dict:
        """Validate and atomically retain original point-cloud bytes for future scenes."""
        source = PointCloudSource.validate(format_name, body)
        directory = self.roots[PointCloudField.ROOT]
        if (
            directory.is_symlink()
            or directory.resolve() != self.drafts_directory / PointCloudField.DIRECTORY
        ):
            raise InvalidSceneBuilderRequest(
                "Point-cloud storage leaves its dedicated directory."
            )
        filename = f"{hashlib.sha256(body).hexdigest()}.{format_name}"
        target = directory / filename
        if target.is_symlink():
            raise InvalidSceneBuilderRequest(
                "Point-cloud storage cannot follow an asset symlink."
            )
        directory.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=".upload-", delete=False
            ) as temporary:
                temporary.write(body)
            os.replace(temporary.name, target)
        elif not target.is_file() or target.read_bytes() != body:
            raise InvalidSceneBuilderRequest(
                "The stored point-cloud content does not match its identifier."
            )
        metadata_path = target.with_suffix(PointCloudField.METADATA_SUFFIX)
        if metadata_path.is_symlink():
            raise InvalidSceneBuilderRequest(
                "Point-cloud metadata cannot follow a symlink."
            )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".metadata-", delete=False
        ) as temporary:
            json.dump(source.metadata(), temporary, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary.name, metadata_path)
        identifier = f"{PointCloudField.ROOT}/{filename}"
        return self._asset(identifier, target)

    def _dependency(self, source: Path, reference: str) -> str:
        """Resolve local and package mesh paths solely within configured roots."""
        parsed = urlsplit(reference)
        if parsed.scheme == "package":
            relative = unquote(parsed.netloc + parsed.path)
            if any(part in {"", ".", ".."} for part in relative.split("/")):
                raise InvalidSceneBuilderRequest("Invalid package resource path.")
            matches = []
            if self._package_files is None:
                self._package_files = [path for _, path in self._files()]
            for candidate in self._package_files:
                if candidate.as_posix().endswith("/" + relative):
                    matches.append(candidate)
            if not matches:
                raise InvalidSceneBuilderRequest(
                    f"Missing package dependency: {relative}"
                )
            return self._identifier(matches[0])
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or Path(unquote(parsed.path)).is_absolute()
        ):
            raise InvalidSceneBuilderRequest(
                "External model dependencies are not allowed."
            )
        return self._identifier(source.parent / unquote(parsed.path))

    def render_asset(self, identifier: str) -> tuple[bytes, str]:
        """Read an asset and rewrite URDF mesh references to protected local URLs."""
        path = self.resolve_asset(identifier)
        if path.suffix.lower() == ".gltf":
            document = json.loads(path.read_text(encoding="utf-8"))
            for collection in ("buffers", "images"):
                for entry in document.get(collection, []):
                    reference = entry.get("uri")
                    if reference is not None and not reference.startswith("data:"):
                        entry["uri"] = self._relative_dependency(
                            identifier, path, reference
                        )
            return json.dumps(document).encode(), "model/gltf+json"
        if path.suffix.lower() != ".urdf":
            return (
                path.read_bytes(),
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
        content = path.read_bytes()
        if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
            raise InvalidSceneBuilderRequest(
                "URDF document declarations are unsupported."
            )
        document = ElementTree.fromstring(content)
        for node in document.iter():
            if node.tag not in {"mesh", "texture"} or "filename" not in node.attrib:
                continue
            node.set(
                "filename",
                self._relative_dependency(identifier, path, node.attrib["filename"]),
            )
        return (
            ElementTree.tostring(document, encoding="utf-8", xml_declaration=True),
            "application/xml",
        )

    def _relative_dependency(
        self, identifier: str, source: Path, reference: str
    ) -> str:
        """Keep a dependency relative to its asset URL for the URDF mesh loader."""
        destination = self._dependency(source, reference)
        relative = posixpath.relpath(destination, posixpath.dirname(identifier))
        return quote(relative, safe="/")

    @staticmethod
    def _pose(pose: list) -> tuple[list[float], list[float]]:
        """Convert recorded xyzw quaternions into URDF roll, pitch and yaw."""
        position = list(pose[:3])
        if len(pose) == 3:
            return position, [0.0, 0.0, 0.0]
        if len(pose) != 7:
            raise InvalidSceneBuilderRequest(
                "A recorded pose requires three or seven values."
            )
        x, y, z, w = pose[3:]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0 or not math.isfinite(norm):
            raise InvalidSceneBuilderRequest("Invalid recorded quaternion.")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        sine_pitch = max(-1.0, min(1.0, 2 * (y * w - x * z)))
        pitch = math.asin(sine_pitch)
        if abs(sine_pitch) >= 1.0 - 1e-12:
            return position, [
                0.0,
                pitch,
                math.atan2(2 * (z * w - x * y), 1 - 2 * (x * x + z * z)),
            ]
        rotation = [
            math.atan2(2 * (x * w + y * z), 1 - 2 * (x * x + y * y)),
            pitch,
            math.atan2(2 * (z * w + x * y), 1 - 2 * (y * y + z * z)),
        ]
        return position, rotation

    def load_scene(self, identifier: str) -> dict:
        """Import the initial model, object and robot configuration from a bundle."""
        path = self._resolve(identifier)
        if path.name != ResourceFile.SCENE:
            raise InvalidSceneBuilderRequest(
                "A template must reference a scene.json bundle."
            )
        source = json.loads(path.read_text(encoding="utf-8"))
        trajectory = {}
        trajectory_reference = source.get("trajectory")
        if isinstance(trajectory_reference, str):
            trajectory_path = path.parent / unquote(trajectory_reference)
            root = self.roots[identifier.split("/", 1)[0]]
            if (
                trajectory_path.resolve().is_relative_to(root)
                and trajectory_path.is_file()
            ):
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        first_frame = (trajectory.get("frames") or [{}])[0]
        bases = (trajectory.get("modelBases") or [{}])[0]
        first_objects = (trajectory.get("objects") or [{}])[0]
        instances = []
        for index, model in enumerate(source.get("models", [])):
            reference = model.get("urdf") or model.get("mesh")
            if not reference:
                continue
            name = model.get("name") or f"model_{index+1}"
            pose = model.get("spawn") or model.get("pose") or list(IDENTITY_POSE)
            if model.get("robot"):
                pose = (trajectory.get("base") or [pose])[0]
            pose = bases.get(name) or pose
            prefix = model.get("prefix", "")
            joints = {
                joint.removeprefix(prefix + "/") if prefix else joint: value
                for joint, value in first_frame.items()
                if not prefix or joint.startswith(prefix + "/")
            }
            instances.append(
                self._instance(
                    f"model_{index+1}",
                    name,
                    self._dependency(path, reference),
                    pose,
                    joints,
                )
            )
        for index, model in enumerate(source.get("objects", [])):
            reference = model.get("urdf") or model.get("mesh")
            if not reference:
                continue
            name = str(model.get("id") or model.get("key") or f"object_{index+1}")
            pose = (
                first_objects.get(model.get("key"))
                or model.get("spawn")
                or list(IDENTITY_POSE)
            )
            instances.append(
                self._instance(
                    f"object_{index+1}",
                    name,
                    self._dependency(path, reference),
                    pose,
                    {},
                )
            )
        result = {
            BuilderField.VERSION: LAYOUT_VERSION,
            BuilderField.NAME: source.get("environmentName")
            or source.get("name")
            or path.parent.name,
            BuilderField.INSTANCES: instances,
            BuilderField.SOURCE_SCENE: identifier,
            BuilderField.NOTICES: [],
        }
        return self._validated(result)

    @classmethod
    def _instance(
        cls, identifier: str, name: str, asset_id: str, pose: list, joints: dict
    ) -> dict:
        """Construct the browser's editable instance from an authored pose."""
        position, rotation = cls._pose(pose)
        return {
            BuilderField.ID: identifier,
            BuilderField.NAME: name,
            BuilderField.ASSET_ID: asset_id,
            BuilderField.POSITION: position,
            BuilderField.ROTATION: rotation,
            BuilderField.SCALE: [1, 1, 1],
            BuilderField.LINK_OFFSETS: {},
            BuilderField.JOINT_POSITIONS: joints,
        }

    def _validated(self, payload: object) -> dict:
        """Verify every model reference and merge its original attribution."""
        document = SceneDocument.from_json(payload).value
        notices = {
            item[BuilderField.URL]: item for item in document[BuilderField.NOTICES]
        }
        for instance in document[BuilderField.INSTANCES]:
            identifier = instance[BuilderField.ASSET_ID]
            if self.resolve_asset(identifier).suffix.lower() not in RESOURCE_SUFFIXES:
                raise InvalidSceneBuilderRequest(
                    "Scene instances must reference model resources."
                )
            for notice in self._notices(identifier):
                notices[notice[BuilderField.URL]] = notice
        document[BuilderField.NOTICES] = list(notices.values())
        return document

    def _draft_path(self, identifier: str) -> Path:
        """Validate one draft ID and its containment in the writable directory."""
        if not isinstance(identifier, str) or not SAFE_NAME.fullmatch(identifier):
            raise InvalidSceneBuilderRequest("Invalid draft identifier.")
        target = self.drafts_directory / f"{identifier}.json"
        if target.is_symlink() or not target.resolve().is_relative_to(
            self.drafts_directory
        ):
            raise InvalidSceneBuilderRequest("Draft path leaves its storage directory.")
        return target

    def list_drafts(self) -> list[dict]:
        """List saved drafts without altering any source scenes."""
        if not self.drafts_directory.is_dir():
            return []
        result = []
        for path in sorted(self.drafts_directory.glob("*.json")):
            if path.is_symlink() or not SAFE_NAME.fullmatch(path.stem):
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            result.append(
                {
                    BuilderField.ID: path.stem,
                    BuilderField.NAME: document[BuilderField.NAME],
                    BuilderField.UPDATED_AT: datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                }
            )
        return result

    def load_draft(self, identifier: str) -> dict:
        """Read and validate a previously saved independent scene document."""
        path = self._draft_path(identifier)
        return {
            BuilderField.ID: identifier,
            BuilderField.DOCUMENT: self._validated(
                json.loads(path.read_text(encoding="utf-8"))
            ),
        }

    def save_draft(self, identifier: str | None, document: object) -> dict:
        """Atomically create or update a draft without writing source assets."""
        normalized = self._validated(document)
        identifier = uuid.uuid4().hex if identifier is None else identifier
        target = self._draft_path(identifier)
        self.drafts_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.drafts_directory,
            prefix=".draft-",
            delete=False,
        ) as temporary:
            json.dump(normalized, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary.name, target)
        return {BuilderField.ID: identifier, BuilderField.DOCUMENT: normalized}
