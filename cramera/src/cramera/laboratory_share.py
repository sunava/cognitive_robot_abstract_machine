"""Share one authenticated laboratory through an independently managed HTTPS tunnel."""

from __future__ import annotations

import hmac
import json
import mimetypes
import re
import secrets
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from html.parser import HTMLParser
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from typing_extensions import ClassVar

from cramera import paths
from cramera.scene_builder import SceneBuilderService
from cramera.scene_builder_http import SceneBuilderHttp, SceneBuilderRoute
from cramera.laboratory_branding import LaboratoryBranding
from cramera.laboratory_devices import LaboratoryDeviceResource, LaboratoryInstrument
from cramera.laboratory_robot_session import RobotPhysicsRoute
from cramera.mesh_format import MeshFormat
from cramera.offline_assets import (
    AssetDependencies,
    AssetFormat,
    AssetReference,
    OfflineSceneAssets,
    SceneAssetField,
)
from cramera.workshop_physics import WorkshopPhysics
from cramera.workshop_session import (
    WorkshopCommand,
    WorkshopField,
    WorkshopRoute,
    WorkshopSession,
    WorkshopSessionInactive,
)


# %% shared configuration
class SharedScene(StrEnum):
    """Scenes explicitly supported by separate protected gateways."""

    LABORATORY = "precision_lab_pr2_physics"
    """Laboratory with its dedicated robot simulation API."""
    WORKSHOP_REPLAY = "aicor_workshop_pr2"
    """Validated native PR2 recording for the workshop."""
    WORKSHOP = "aicor_workshop"
    """Wood workshop with contact physics and recorded robot motion."""


class ShareRoute(StrEnum):
    """Public entry points of the fixed laboratory."""

    ROOT = "/"
    """Canonical viewer redirect or unauthenticated login form."""
    LOGIN = "/login"
    """Password form submission."""
    INDEX = "/index.html"
    """Packaged viewer document."""
    VIEWER = "/index.html?scene=precision_lab_pr2_physics&layout=scene&offline=1"
    """Laboratory page isolated from the general live bridge."""
    SCENE_INDEX = "/scenes/index.json"
    """Filtered scene picker containing only the shared laboratory."""


class ShareFile(StrEnum):
    """Packaged files forming the viewer and login page."""

    INDEX = "index.html"
    """Viewer dependency entry point."""
    LOGIN = "laboratory-login.html"
    """Password form served before authentication."""
    PHYSICS_CLIENT = "core/laboratory-physics.js"
    """Browser client containing the laboratory's absolute API routes."""


class InstrumentRoute(StrEnum):
    """Protected entry points for the standalone instrument collection."""

    ROOT = "/instruments"
    """Exact prefix for the individually allowlisted resource files."""
    CATALOG = "/instruments/catalog/index.html"
    """Gallery and permitted password-login continuation."""


class InstrumentFile(StrEnum):
    """Files intentionally published inside each standalone instrument package."""

    MODEL = "model.urdf"
    """Independent articulated model with local mesh references."""
    BLENDER = "model.blend"
    """Editable source model."""
    PREVIEW = "preview.png"
    """Closed-device preview."""
    OPEN_PREVIEW = "preview_open.png"
    """Inspection-pose preview."""
    README = "README.md"
    """Scope and usage instructions."""
    METADATA = "metadata.json"
    """Articulation and interaction-frame information."""
    DEFINITION = "definition.json"
    """Metric model definition."""
    LICENSE = "LICENSE.txt"
    """Reuse terms accompanying the original models."""
    NOTICE = "NOTICE.md"
    """Authorship and provenance notice."""
    RIGHTS = "rights.json"
    """Machine-readable rights information."""


class SharedRightsField(StrEnum):
    """Explicit rights references carried by workshop manifests and glTF assets."""

    RIGHTS = "assetRights"
    """Metadata documenting the original workshop assets."""
    LICENSE = "licenseFile"
    """Relative location of the declared reuse terms."""
    NOTICE = "noticeFile"
    """Relative location of the declared attribution notice."""
    ASSET = "asset"
    """glTF asset metadata container."""
    EXTRAS = "extras"
    """glTF extension data containing asset rights."""


@dataclass(frozen=True)
class ShareOptions:
    """Loopback listeners and private configuration for one shared laboratory."""

    SCENE_NAME: ClassVar[str] = "precision_lab_pr2_physics"
    """Only scene that the public gateway may expose."""
    MAX_BODY_BYTES: ClassVar[int] = 8192
    """Largest accepted form or simulation command."""
    MAX_MANUAL_BODY_BYTES: ClassVar[int] = 64 * 1024
    """Largest JSON-escaped document request accepted after authentication."""
    MAX_RESPONSE_BYTES: ClassVar[int] = 4 * 1024 * 1024
    """Largest simulation JSON response accepted from the local backend."""
    COOKIE_NAME: ClassVar[str] = "__Host-cramera_lab"
    """Host-only secure browser session cookie."""
    LOOPBACK: ClassVar[str] = "127.0.0.1"
    """Private address used for both HTTP listeners."""

    password_file: Path
    """Private file containing the login password."""
    public_origin_file: Path
    """File containing the exact current HTTPS tunnel origin."""
    port: int = 8717
    """Loopback gateway port."""
    backend_port: int = 8716
    """Existing local simulation server port."""
    scene_directory: Path | None = None
    """Optional fixed bundle directory, otherwise resolved from the scene catalog."""
    scene_name: SharedScene = SharedScene.LABORATORY
    """Single scene exposed by this gateway instance."""
    replay_directory: Path | None = None
    """Optional explicit workshop recording exposed beside the workshop."""
    path_prefix: str = ""
    """Optional public mount path without a trailing slash."""
    builder_directory: Path | None = None
    """Explicit library of named model roots available to workshop authoring."""
    builder_drafts_directory: Path | None = None
    """Writable scene document store independent of immutable model releases."""

    def __post_init__(self) -> None:
        """Reject ambiguous mount paths before they can affect public routes."""
        SharedScene(self.scene_name)
        if (
            self.replay_directory is not None
            and self.scene_name != SharedScene.WORKSHOP
        ):
            raise ValueError("A recording directory is only valid for a workshop share")
        if self.path_prefix and not re.fullmatch(
            r"(?:/[A-Za-z0-9_-]+)+", self.path_prefix
        ):
            raise ValueError(
                "The sharing prefix must contain only canonical path segments"
            )

    @property
    def viewer_route(self) -> str:
        """Select the fixed scene without permitting arbitrary query parameters."""
        return f"/index.html?scene={self.scene_name}&layout=scene&offline=1"

    @property
    def cookie_name(self) -> str:
        """Keep workshop and laboratory browser sessions independent."""
        if self.scene_name == SharedScene.LABORATORY:
            return self.COOKIE_NAME
        return "__Host-cramera_workshop"

    @property
    def robot_controls(self) -> bool:
        """Enable the laboratory API only for its own gateway."""
        return self.scene_name == SharedScene.LABORATORY

    def public_path(self, route: str) -> str:
        """Place an internal route beneath the configured public mount."""
        return self.path_prefix + route

    @property
    def backend_origin(self) -> str:
        """Return the fixed local simulation origin."""
        return f"http://{self.LOOPBACK}:{self.backend_port}"

    def public_origin(self) -> str | None:
        """Read the configured HTTPS origin, failing closed during tunnel startup."""
        try:
            origin = self.public_origin_file.read_text().strip()
        except OSError:
            return None
        if not re.fullmatch(r"https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?", origin):
            return None
        return origin

    @classmethod
    def parse(cls, arguments: list[str] | None = None) -> ShareOptions:
        """Parse the private gateway's listener and authentication files."""
        parser = ArgumentParser(description=__doc__)
        parser.add_argument("--port", type=int, default=8717)
        parser.add_argument("--backend-port", type=int, default=8716)
        parser.add_argument("--password-file", type=Path, required=True)
        parser.add_argument("--public-origin-file", type=Path, required=True)
        parser.add_argument("--path-prefix", default="")
        parser.add_argument("--scene-directory", type=Path)
        parser.add_argument("--builder-directory", type=Path)
        parser.add_argument("--builder-drafts-directory", type=Path)
        parser.add_argument("--replay-directory", type=Path)
        parser.add_argument(
            "--scene-name",
            type=SharedScene,
            choices=list(SharedScene),
            default=SharedScene.LABORATORY,
        )
        options = parser.parse_args(arguments)
        if not 0 <= options.port <= 65535 or not 1 <= options.backend_port <= 65535:
            parser.error("Listener ports must be valid TCP ports")
        return cls(**vars(options))


# %% exact dependency allowlist
@dataclass
class DocumentDependencies(HTMLParser):
    """Collect linked scripts, styles and images from the viewer document."""

    references: list[str] = field(default_factory=list, init=False)
    """Directly linked local web assets."""

    def __post_init__(self) -> None:
        """Initialize the standard HTML parser."""
        super().__init__()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect scripts, images and linked stylesheets."""
        attributes = dict(attrs)
        if tag in ("script", "img") and attributes.get("src"):
            self.references.append(attributes["src"])
        if tag == "link" and attributes.get("rel") == "stylesheet":
            self.references.append(attributes["href"])


@dataclass(frozen=True)
class SharedAsset:
    """One allowlisted file whose resolved location remains beneath its root."""

    path: Path
    """Original allowlisted pathname."""
    root: Path
    """Containing directory that may be exposed."""
    follow_symlinks: bool = True
    """Whether a still-confined symbolic link may supply the file contents."""

    def read(self) -> bytes | None:
        """Read a still-existing file without following an escaping symlink."""
        resolved = self.path.resolve()
        if (
            not resolved.is_relative_to(self.root)
            or not resolved.is_file()
            or not self.follow_symlinks
            and resolved != self.path.absolute()
        ):
            return None
        return resolved.read_bytes()


@dataclass(frozen=True)
class InstrumentAssets:
    """The explicit instrument gallery and only its declared standalone packages."""

    directory: Path
    """Installed instrument collection, independent of scene mesh directories."""

    def files(self) -> dict[str, SharedAsset]:
        """Allow known package documents and URDF-declared embedded-texture meshes."""
        root = self.directory.resolve()
        files: dict[str, SharedAsset] = {}
        self._include(files, InstrumentRoute.CATALOG, root / "catalog/index.html", root)
        for filename in (
            InstrumentFile.README,
            InstrumentFile.LICENSE,
            InstrumentFile.NOTICE,
            InstrumentFile.RIGHTS,
        ):
            self._include(
                files, f"{InstrumentRoute.ROOT}/{filename}", root / filename, root
            )
        for instrument in LaboratoryInstrument:
            package = root / instrument
            if package.is_symlink() or not package.is_dir():
                continue
            prefix = f"{InstrumentRoute.ROOT}/{instrument}"
            for filename in InstrumentFile:
                self._include(
                    files, f"{prefix}/{filename}", package / filename, package
                )
            description_route = f"{prefix}/{InstrumentFile.MODEL}"
            if description_route not in files:
                continue
            description = package / InstrumentFile.MODEL
            for reference in AssetDependencies(description).references():
                if re.fullmatch(r"assets/[A-Za-z_][A-Za-z0-9_-]*\.glb", reference):
                    self._include(
                        files, f"{prefix}/{reference}", package / reference, package
                    )
        return files

    @staticmethod
    def _include(
        files: dict[str, SharedAsset], route: str, path: Path, root: Path
    ) -> None:
        """Keep exact file paths confined, without permitting symbolic-link aliases."""
        resolved = path.resolve()
        if (
            resolved != path
            or not resolved.is_relative_to(root)
            or not resolved.is_file()
        ):
            return
        files[route] = SharedAsset(path, root, follow_symlinks=False)


@dataclass
class LaboratoryAssets:
    """The exact packaged viewer and one scene's transitive dependencies."""

    scene_directory: Path
    """Bundle containing the shared scene."""
    scene_name: SharedScene = SharedScene.LABORATORY
    """Public name of this one scene."""
    replay_directory: Path | None = None
    """Explicit optional second workshop bundle."""
    files: dict[str, SharedAsset] = field(default_factory=dict, init=False)
    """Public request paths mapped to individually validated files."""

    def __post_init__(self) -> None:
        """Build both dependency graphs before opening a listener."""
        self._web_files()
        self._scene_files()
        if self.scene_name == SharedScene.LABORATORY:
            directory = LaboratoryDeviceResource(
                LaboratoryInstrument.VORTEX_MIXER
            ).directory.parent
            self.files.update(InstrumentAssets(directory).files())
        if self.replay_directory is not None:
            recording = LaboratoryAssets(
                self.replay_directory, SharedScene.WORKSHOP_REPLAY
            )
            self.files.update(recording.files)

    def _web_files(self) -> None:
        """Allow the viewer's declared scripts, styles and images."""
        root = paths.WEB_ROOT.resolve()
        manifests = [root / ShareFile.INDEX]
        if self.scene_name == SharedScene.WORKSHOP:
            manifests.append(root / str(SceneBuilderRoute.PAGE).lstrip("/"))
        pending = []
        for manifest in manifests:
            parser = DocumentDependencies()
            parser.feed(manifest.read_text())
            pending.append(AssetReference(manifest, manifest.name))
            pending.extend(AssetReference(manifest, ref) for ref in parser.references)
        resolver = OfflineSceneAssets(root)
        while pending:
            reference = pending.pop()
            path = resolver.resolve_reference(reference.source, reference.location)
            if path is None:
                continue
            route = "/" + path.relative_to(root).as_posix()
            if route in self.files:
                continue
            self.files[route] = SharedAsset(path, root)
            if path.suffix == ".css":
                pending.extend(
                    AssetReference(path, match.strip(" \t\"'"))
                    for match in re.findall(r"url\(([^)]+)\)", path.read_text())
                    if not match.strip(" \t\"'").startswith("data:")
                )

    def _scene_files(self) -> None:
        """Allow the scene manifest and all declared model and texture dependencies."""
        root = self.scene_directory.resolve()
        manifest = root / SceneAssetField.SCENE
        document = json.loads(manifest.read_text())
        references = [SceneAssetField.SCENE, document[SceneAssetField.TRAJECTORY]]
        references.extend(self._rights_references(document))
        references.extend(
            model[SceneAssetField.URDF]
            for model in document.get(SceneAssetField.MODELS, [])
        )
        references.extend(
            item[key]
            for item in document.get(SceneAssetField.OBJECTS, [])
            for key in (SceneAssetField.MESH, SceneAssetField.MATERIAL)
            if key in item
        )
        pending = [AssetReference(manifest, ref) for ref in references]
        resolver = OfflineSceneAssets(root)
        while pending:
            reference = pending.pop()
            path = resolver.resolve_reference(reference.source, reference.location)
            if path is None:
                continue
            route = f"/scenes/{self.scene_name}/{path.relative_to(root).as_posix()}"
            if route in self.files:
                continue
            self.files[route] = SharedAsset(path, root)
            dependencies = AssetDependencies(path)
            pending.extend(
                AssetReference(path, ref) for ref in dependencies.references()
            )
            if self.scene_name != SharedScene.LABORATORY and path.suffix.lower() in (
                AssetFormat.GLTF,
                MeshFormat.GLB,
            ):
                asset_metadata = dependencies.gltf_document().get(
                    SharedRightsField.ASSET, {}
                )
                pending.extend(
                    AssetReference(path, reference)
                    for reference in self._rights_references(
                        asset_metadata.get(SharedRightsField.EXTRAS, {})
                    )
                )

    def _rights_references(self, document: dict) -> list[str]:
        """Expose only declared workshop notice files through the confined resolver."""
        if self.scene_name not in (SharedScene.WORKSHOP, SharedScene.WORKSHOP_REPLAY):
            return []
        rights = document.get(SharedRightsField.RIGHTS, {})
        if not isinstance(rights, dict):
            return []
        return [
            rights[field]
            for field in (SharedRightsField.LICENSE, SharedRightsField.NOTICE)
            if field in rights and isinstance(rights[field], str)
        ]

    @property
    def scene_names(self) -> tuple[SharedScene, ...]:
        """Return exactly the configured main scene and optional recording."""
        if self.replay_directory is not None:
            return self.scene_name, SharedScene.WORKSHOP_REPLAY
        return (self.scene_name,)

    @property
    def viewer_routes(self) -> tuple[str, ...]:
        """Accept canonical viewer queries only for configured scenes."""
        return tuple(
            f"/index.html?scene={name}&layout=scene&offline=1"
            for name in self.scene_names
        )

    def scene_index(self) -> bytes:
        """Return only this gateway's explicitly configured scenes."""
        entries = []
        for name in self.scene_names:
            laboratory = name == SharedScene.LABORATORY
            robot = name != SharedScene.WORKSHOP
            entries.append(
                {
                    "name": name,
                    "robot": "pr2" if robot else "",
                    "environment": (
                        LaboratoryBranding.NAME if laboratory else "AICOR Wood Workshop"
                    ),
                    "task": (
                        "PR2 · Contact physics and liquids"
                        if laboratory
                        else (
                            "PR2 · Screw assembly recording"
                            if robot
                            else "Woodworking and assembly"
                        )
                    ),
                }
            )
        return json.dumps({"default": self.scene_name, "scenes": entries}).encode()


# %% authenticated HTTP gateway
@dataclass(init=False, eq=False)
class LaboratoryShareServer(ThreadingHTTPServer):
    """Own authentication and allowlisted assets independently of the simulation."""

    options: ShareOptions
    """Fixed ports and private configuration files."""
    password: bytes
    """Login secret loaded from the private file."""
    session: str
    """Random process-local browser session token."""
    assets: LaboratoryAssets
    """Allowed public assets."""
    workshop_physics: WorkshopSession | None
    """Dedicated workshop contact world, absent from laboratory gateways."""
    scene_builder: SceneBuilderHttp | None
    """Isolated authoring library and document store for the workshop gateway."""
    daemon_threads: ClassVar[bool] = True
    """Discard incomplete connections when shutting down."""

    def __init__(self, options: ShareOptions) -> None:
        """Validate the scene and password before listening on loopback."""
        self.options = options
        self.password = options.password_file.read_bytes().rstrip(b"\r\n")
        if len(self.password) < 20:
            raise ValueError("The sharing password must contain at least 20 bytes")
        self.session = secrets.token_urlsafe(32)
        directory = options.scene_directory or paths.resolve_scene_directory(
            options.scene_name
        )
        if directory is None:
            raise FileNotFoundError("The laboratory scene has not been generated")
        self.assets = LaboratoryAssets(
            directory, options.scene_name, options.replay_directory
        )
        self.workshop_physics = (
            WorkshopSession(
                factory=partial(WorkshopPhysics, bundle_directory=directory)
            )
            if options.scene_name == SharedScene.WORKSHOP
            else None
        )
        self.scene_builder = None
        if options.scene_name == SharedScene.WORKSHOP:
            roots = {"scene": directory}
            if options.replay_directory is not None:
                roots["replay"] = options.replay_directory
            if options.builder_directory is not None:
                roots = {
                    child.name: child
                    for child in options.builder_directory.iterdir()
                    if child.is_dir() and not child.is_symlink()
                }
            self.scene_builder = SceneBuilderHttp(
                service=SceneBuilderService(
                    roots=roots,
                    drafts_directory=options.builder_drafts_directory
                    or paths.data_directory() / "scene-builder",
                    url_prefix=options.public_path(SceneBuilderRoute.ASSETS),
                )
            )
        super().__init__((ShareOptions.LOOPBACK, options.port), ShareHandler)

    def server_close(self) -> None:
        """Release contact controls before closing the workshop listener."""
        if self.workshop_physics is not None:
            self.workshop_physics.close()
        super().server_close()


class ShareHandler(BaseHTTPRequestHandler):
    """Authenticate requests and expose a fixed viewer and simulation API subset."""

    server: LaboratoryShareServer
    """Gateway configuration and process-local authentication."""
    server_version = "LaboratoryShare"
    """Public server identifier without interpreter details."""
    sys_version = ""
    """Suppress the Python version in response headers."""

    def setup(self) -> None:
        """Bound the lifetime of an incomplete HTTP request."""
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format: str, *args: object) -> None:
        """Avoid logging credentials, query strings or simulation polls."""

    def _reply(
        self,
        status: int,
        body: bytes = b"",
        content_type: str = "text/plain; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Send an uncached response constrained to the tunnel origin."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            "connect-src 'self' blob:; worker-src blob:; object-src 'none'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _route(self) -> str | None:
        """Accept a canonical origin-form path without traversal or ambiguous encoding."""
        parsed = urlsplit(self.path)
        route = unquote(parsed.path)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or not route.startswith("/")
            or "\\" in route
            or "%" in route
            or "\x00" in route
            or any(part in (".", "..") for part in route.split("/"))
            or "//" in route
        ):
            return None
        prefix = self.server.options.path_prefix
        if prefix:
            if not route.startswith(prefix + "/"):
                return None
            route = route.removeprefix(prefix)
        if parsed.query and route not in (ShareRoute.INDEX, SceneBuilderRoute.TEMPLATE):
            return None
        return route

    def _public_payload(self, route: str, payload: bytes) -> bytes:
        """Keep form and simulation requests beneath this laboratory's mount."""
        options = self.server.options
        if (
            route == f"/scenes/{SharedScene.WORKSHOP}/scene.json"
            and options.replay_directory is not None
        ):
            scene = json.loads(payload)
            scene.setdefault("workshop", {})[
                "demoUrl"
            ] = f"index.html?scene={SharedScene.WORKSHOP_REPLAY}&layout=scene&offline=1"
            payload = json.dumps(scene, indent=2).encode()
        if route == ShareFile.LOGIN and options.scene_name == SharedScene.WORKSHOP:
            payload = (
                payload.replace(b"CRAMERA Laboratory", b"AICOR Workshop")
                .replace(
                    b"laboratory with PR2, contact physics and liquids",
                    b"wood workshop, movable drawers and assembly parts",
                )
                .replace(b"Open laboratory", b"Open workshop")
            )
        if not options.path_prefix:
            return payload
        if route == ShareFile.LOGIN:
            return payload.replace(
                f'action="{ShareRoute.LOGIN}"'.encode(),
                f'action="{options.public_path(ShareRoute.LOGIN)}"'.encode(),
            )
        if route == "/" + ShareFile.PHYSICS_CLIENT:
            return payload.replace(
                b"'/api/laboratory/", f"'{options.path_prefix}/api/laboratory/".encode()
            )
        return payload

    def _public_origin(self) -> str | None:
        """Require the currently configured public host before serving any response."""
        origin = self.server.options.public_origin()
        if origin is None:
            self._reply(HTTPStatus.SERVICE_UNAVAILABLE, b"Sharing is starting.")
            return None
        if self.headers.get_all("Host") != [urlsplit(origin).netloc]:
            self._reply(HTTPStatus.FORBIDDEN, b"Unknown sharing host.")
            return None
        return origin

    def _authenticated(self) -> bool:
        """Compare the host-only cookie with the random process-local session."""
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            return False
        if self.server.options.cookie_name not in cookie:
            return False
        return hmac.compare_digest(
            cookie[self.server.options.cookie_name].value.encode(),
            self.server.session.encode(),
        )

    def _same_origin(self, origin: str) -> bool:
        """Require an exact public origin for every form or control mutation."""
        if (
            self.headers.get_all("Origin") != [origin]
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
        ):
            self._reply(HTTPStatus.FORBIDDEN, b"A same-origin request is required.")
            return False
        return True

    def _body(
        self, content_type: str, maximum_bytes: int = ShareOptions.MAX_BODY_BYTES
    ) -> bytes | None:
        """Read one bounded, explicitly typed request without transfer encodings."""
        lengths = self.headers.get_all("Content-Length", [])
        if (
            self.headers.get("Transfer-Encoding") is not None
            or len(lengths) != 1
            or not lengths[0].isdigit()
            or self.headers.get("Content-Type", "").split(";", 1)[0] != content_type
        ):
            self._reply(HTTPStatus.BAD_REQUEST, b"Invalid request body.")
            return None
        length = int(lengths[0])
        if length > maximum_bytes:
            self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"Request is too large.")
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._reply(HTTPStatus.BAD_REQUEST, b"Incomplete request body.")
            return None
        return body

    def _login(self) -> None:
        """Exchange a correct password for a secure, host-only session cookie."""
        body = self._body("application/x-www-form-urlencoded")
        if body is None:
            return
        try:
            form = parse_qs(body.decode("utf-8"), strict_parsing=True)
        except (UnicodeDecodeError, ValueError):
            self._reply(HTTPStatus.BAD_REQUEST, b"Invalid login form.")
            return
        password = form.get("password", [])
        destination = form.get("next", [self.server.options.viewer_route])
        if (
            set(form) - {"password", "next"}
            or len(password) != 1
            or len(destination) != 1
            or destination[0]
            not in (
                self.server.options.viewer_route,
                *(
                    [SceneBuilderRoute.PAGE]
                    if self.server.scene_builder is not None
                    else []
                ),
                *(
                    [InstrumentRoute.CATALOG]
                    if InstrumentRoute.CATALOG in self.server.assets.files
                    else []
                ),
            )
            or not hmac.compare_digest(password[0].encode(), self.server.password)
        ):
            self._reply(HTTPStatus.UNAUTHORIZED, b"Incorrect password.")
            return
        self._reply(
            HTTPStatus.SEE_OTHER,
            headers={
                "Location": self.server.options.public_path(destination[0]),
                "Set-Cookie": (
                    f"{self.server.options.cookie_name}={self.server.session}; "
                    "Path=/; HttpOnly; SameSite=Strict; Secure"
                ),
            },
        )

    def _proxy(self, route: str, body: bytes | None = None) -> None:
        """Forward only a validated laboratory operation to the fixed local backend."""
        options = self.server.options
        connection = HTTPConnection(
            ShareOptions.LOOPBACK, options.backend_port, timeout=30
        )
        try:
            connection.request(
                self.command,
                route,
                body=body,
                headers={
                    "Origin": options.backend_origin,
                    "Sec-Fetch-Site": "same-origin",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            payload = response.read(ShareOptions.MAX_RESPONSE_BYTES + 1)
            if (
                len(payload) > ShareOptions.MAX_RESPONSE_BYTES
                or 300 <= response.status < 400
                or response.getheader("Content-Type", "").split(";", 1)[0]
                != "application/json"
            ):
                self._reply(HTTPStatus.BAD_GATEWAY, b"Invalid simulation response.")
                return
            self._reply(response.status, payload, "application/json")
        except (OSError, HTTPException):
            self._reply(HTTPStatus.BAD_GATEWAY, b"The simulation is unavailable.")
        finally:
            connection.close()

    def do_GET(self) -> None:
        """Serve the login form, authenticated assets or simulation state."""
        if self._public_origin() is None:
            return
        route = self._route()
        if route is None:
            self._reply(HTTPStatus.NOT_FOUND)
            return
        if not self._authenticated():
            builder_login = (
                route == SceneBuilderRoute.PAGE
                and self.server.scene_builder is not None
            )
            instrument_login = (
                route == InstrumentRoute.CATALOG and route in self.server.assets.files
            )
            if (
                route in (ShareRoute.ROOT, ShareRoute.INDEX, ShareRoute.LOGIN)
                or builder_login
                or instrument_login
            ):
                login = (paths.WEB_ROOT / ShareFile.LOGIN).read_bytes()
                if builder_login or instrument_login:
                    login = login.replace(
                        b'<label for="password">',
                        f'<input type="hidden" name="next" value="{route}"><label for="password">'.encode(),
                    )
                self._reply(
                    HTTPStatus.OK,
                    self._public_payload(ShareFile.LOGIN, login),
                    "text/html; charset=utf-8",
                )
            else:
                self._reply(HTTPStatus.UNAUTHORIZED, b"Login required.")
            return
        if route in (ShareRoute.ROOT, ShareRoute.LOGIN) or (
            route == ShareRoute.INDEX
            and self.path
            not in tuple(
                self.server.options.public_path(viewer)
                for viewer in self.server.assets.viewer_routes
            )
        ):
            self._reply(
                HTTPStatus.SEE_OTHER,
                headers={
                    "Location": self.server.options.public_path(
                        self.server.options.viewer_route
                    )
                },
            )
            return
        if self.server.options.robot_controls and route == RobotPhysicsRoute.STATE:
            self._proxy(route)
            return
        if self.server.scene_builder is not None and SceneBuilderHttp.handles(route):
            response = self.server.scene_builder.get(route, urlsplit(self.path).query)
            self._reply(response.status, response.body, response.content_type)
            return
        if self.server.workshop_physics is not None and route == WorkshopRoute.STATE:
            self._reply(
                HTTPStatus.OK,
                json.dumps(self.server.workshop_physics.state()).encode(),
                "application/json",
            )
            return
        if route == ShareRoute.SCENE_INDEX:
            self._reply(
                HTTPStatus.OK, self.server.assets.scene_index(), "application/json"
            )
            return
        asset = self.server.assets.files.get(route)
        payload = asset.read() if asset is not None else None
        if payload is None:
            self._reply(HTTPStatus.NOT_FOUND)
            return
        self._reply(
            HTTPStatus.OK,
            self._public_payload(route, payload),
            mimetypes.guess_type(str(asset.path))[0] or "application/octet-stream",
        )

    def do_POST(self) -> None:
        """Accept authenticated same-origin laboratory controls or a login form."""
        origin = self._public_origin()
        if origin is None:
            return
        route = self._route()
        if route is None:
            self._reply(HTTPStatus.NOT_FOUND)
            return
        if route != ShareRoute.LOGIN and not self._authenticated():
            self._reply(HTTPStatus.UNAUTHORIZED, b"Login required.")
            return
        workshop_command = (
            route == WorkshopRoute.COMMAND and self.server.workshop_physics is not None
        )
        builder_command = (
            SceneBuilderHttp.accepts_post(route)
            and self.server.scene_builder is not None
        )
        if (
            route != ShareRoute.LOGIN
            and (
                not self.server.options.robot_controls
                or route
                not in {
                    entry
                    for entry in RobotPhysicsRoute
                    if entry != RobotPhysicsRoute.STATE
                }
            )
            and not workshop_command
            and not builder_command
        ):
            self._reply(HTTPStatus.NOT_FOUND)
            return
        if not self._same_origin(origin):
            return
        if route == ShareRoute.LOGIN:
            self._login()
            return
        if workshop_command:
            self._workshop_command()
            return
        if builder_command:
            body = self._body(
                SceneBuilderHttp.request_content_type(route),
                SceneBuilderHttp.maximum_body_bytes(route),
            )
            if body is not None:
                response = self.server.scene_builder.post(route, body)
                self._reply(response.status, response.body, response.content_type)
            return
        maximum_bytes = (
            ShareOptions.MAX_MANUAL_BODY_BYTES
            if route == RobotPhysicsRoute.MANUAL
            else ShareOptions.MAX_BODY_BYTES
        )
        body = self._body("application/json", maximum_bytes)
        if body is not None:
            self._proxy(route, body)

    def _workshop_command(self) -> None:
        """Apply a bounded command to this gateway's independent contact world."""
        body = self._body("application/json")
        if body is None:
            return
        try:
            command = WorkshopCommand.from_json(json.loads(body))
            state = self.server.workshop_physics.command(command)
        except (ValueError, TypeError) as error:
            self._reply(
                HTTPStatus.BAD_REQUEST,
                json.dumps({WorkshopField.ERROR: str(error)}).encode(),
                "application/json",
            )
            return
        except WorkshopSessionInactive as error:
            self._reply(
                HTTPStatus.CONFLICT,
                json.dumps({WorkshopField.ERROR: str(error)}).encode(),
                "application/json",
            )
            return
        self._reply(HTTPStatus.OK, json.dumps(state).encode(), "application/json")


def make_server(options: ShareOptions) -> LaboratoryShareServer:
    """Create the authenticated loopback gateway without changing the backend."""
    return LaboratoryShareServer(options)


def main(arguments: list[str] | None = None) -> None:
    """Serve the fixed laboratory until the sharing process is stopped."""
    with make_server(ShareOptions.parse(arguments)) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
