"""Carry laboratory asset attribution into standalone model exports."""

from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from xml.parsers import expat

# %% portable notice vocabulary


class RightsFile(StrEnum):
    """Canonical notices copied into each portable resource directory."""

    METADATA = "rights.json"
    """Machine-readable author and permission information."""
    LICENSE = "LICENSE.txt"
    """Complete asset permission terms."""
    NOTICE = "NOTICE.md"
    """Attribution and third-party scope notice."""


class RightsField(StrEnum):
    """Fields in the canonical laboratory asset rights document."""

    AUTHOR = "principalAuthor"
    """Principal human author of the laboratory models."""
    INSTITUTION = "institution"
    """Associated institution."""
    COPYRIGHT = "copyright"
    """Canonical copyright notice."""
    LICENSE = "license"
    """Identifier of the referenced permission terms."""
    PERMISSION = "permissionNotice"
    """Short statement of the permission requirement."""
    PRIOR_LICENSE = "priorLicenseNotice"
    """Reservation of existing rights and permissions."""
    LICENSE_FILE = "licenseFile"
    """Relative path to the complete permission terms."""
    NOTICE_FILE = "noticeFile"
    """Relative path to the attribution notice."""


class AssetField(StrEnum):
    """Standard glTF fields and namespaced export metadata."""

    ASSET = "asset"
    """Standard glTF asset description."""
    COPYRIGHT = "copyright"
    """Standard glTF copyright field."""
    EXTRAS = "extras"
    """Application metadata retained alongside the glTF asset."""
    RIGHTS = "assetRights"
    """Portable laboratory rights record."""
    PRIOR_COPYRIGHT = "priorCopyright"
    """Distinct attribution already present before laboratory packaging."""


class PackagePath(StrEnum):
    """Locations of independently attributed laboratory assets."""

    CANONICAL = "asset_rights"
    """Notice directory beside this standalone module."""
    SCENE = "scene.json"
    """Complete laboratory scene description."""
    METADATA = "metadata.json"
    """Independent device description."""
    ASSETS = "assets"
    """Embedded-texture model directory."""
    URDFS = "*.urdf"
    """Descriptions directly belonging to the export root."""
    GLBS = "*.glb"
    """Binary glTF assets directly belonging to the export."""
    COMMENT = "CRAMERA laboratory asset rights:"
    """Stable marker identifying the managed XML attribution comment."""


class GlbLayout(IntEnum):
    """Byte layout of a version-two binary glTF file."""

    VERSION = 2
    """Supported binary glTF version."""
    HEADER = 12
    """Length of the file header in bytes."""
    CHUNK_HEADER = 8
    """Length of each chunk header in bytes."""
    ALIGNMENT = 4
    """Required chunk alignment in bytes."""


@dataclass
class InvalidLaboratoryAsset(ValueError):
    """An asset cannot receive attribution without discarding existing data."""

    path: Path
    """Asset whose contents violate the supported packaging contract."""
    reason: str
    """Specific invalid structure or unsupported attribution representation."""

    def __str__(self) -> str:
        """Identify the problematic file and the preservation failure."""
        return f"{self.path}: {self.reason}"


# %% lossless attribution installation


@dataclass(frozen=True)
class LaboratoryAssetRights:
    """Install self-contained notices while preserving physical model contents."""

    source: Path = field(
        default_factory=lambda: Path(__file__).with_name(PackagePath.CANONICAL)
    )
    """Authoritative rights document and companion notice files."""

    @property
    def metadata(self) -> dict[str, str]:
        """Read the canonical rights fields without importing the CRAM runtime."""
        path = self.source / RightsFile.METADATA
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or any(
            not isinstance(metadata.get(key), str) for key in RightsField
        ):
            raise InvalidLaboratoryAsset(path, "Missing textual rights fields")
        return metadata

    def install(self, directory: Path) -> None:
        """Attribute root descriptions and assets without traversing robot bundles."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata
        for filename in RightsFile:
            source = self.source / filename
            target = directory / filename
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        for filename in (PackagePath.SCENE, PackagePath.METADATA):
            path = directory / filename
            if path.is_file():
                document = json.loads(path.read_text(encoding="utf-8"))
                document[AssetField.RIGHTS] = metadata
                path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        for path in sorted(directory.glob(PackagePath.URDFS)):
            self._annotate_urdf(path, metadata)
        embedded = dict(metadata)
        for key in (RightsField.LICENSE_FILE, RightsField.NOTICE_FILE):
            embedded[key] = f"../{metadata[key]}"
        for path in sorted((directory / PackagePath.ASSETS).glob(PackagePath.GLBS)):
            self._annotate_glb(path, embedded)

    @staticmethod
    def _annotate_urdf(path: Path, metadata: dict[str, str]) -> None:
        """Insert or refresh a managed comment while retaining all original XML."""
        content = path.read_bytes()
        comment = (
            f"<!-- {PackagePath.COMMENT}\n"
            f"Principal author: {metadata[RightsField.AUTHOR]}\n"
            f"Institution: {metadata[RightsField.INSTITUTION]}\n"
            f"{metadata[RightsField.COPYRIGHT]}\n"
            f"License: {metadata[RightsField.LICENSE]}\n"
            f"See {metadata[RightsField.LICENSE_FILE]} and {metadata[RightsField.NOTICE_FILE]}.\n"
            "-->"
        ).encode("utf-8")
        if b"--" in comment[4:-3]:
            raise InvalidLaboratoryAsset(
                path, "Rights fields contain an invalid XML comment delimiter"
            )
        marker = rb"<!-- " + re.escape(PackagePath.COMMENT.encode()) + rb".*?-->"
        if re.search(marker, content, flags=re.DOTALL):
            path.write_bytes(
                re.sub(marker, lambda _: comment, content, flags=re.DOTALL)
            )
            return
        parser = expat.ParserCreate()
        root_offsets: list[int] = []

        def record_root(name: str, attributes: dict[str, str]) -> None:
            """Locate the actual root opening tag independently of XML comments."""
            if not root_offsets:
                if name != "robot":
                    raise InvalidLaboratoryAsset(
                        path, "URDF root must be a robot element"
                    )
                root_offsets.append(parser.CurrentByteIndex)

        parser.StartElementHandler = record_root
        parser.Parse(content, True)
        opening = re.compile(rb"<robot\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>").match(
            content, root_offsets[0]
        )
        if opening is None or opening.group().endswith(b"/>"):
            raise InvalidLaboratoryAsset(path, "URDF robot root has no model contents")
        offset = opening.end()
        path.write_bytes(content[:offset] + comment + content[offset:])

    @staticmethod
    def _annotate_glb(path: Path, metadata: dict[str, str]) -> None:
        """Replace only the JSON chunk, preserving every other chunk byte for byte."""
        content = path.read_bytes()
        if len(content) < GlbLayout.HEADER + GlbLayout.CHUNK_HEADER:
            raise InvalidLaboratoryAsset(path, "Truncated binary glTF header")
        magic, version, length = struct.unpack_from("<4sII", content)
        if magic != b"glTF" or version != GlbLayout.VERSION or length != len(content):
            raise InvalidLaboratoryAsset(path, "Invalid version-two binary glTF header")
        offset = GlbLayout.HEADER
        json_end = 0
        while offset < length:
            if offset + GlbLayout.CHUNK_HEADER > length:
                raise InvalidLaboratoryAsset(path, "Truncated glTF chunk header")
            size, kind = struct.unpack_from("<I4s", content, offset)
            end = offset + GlbLayout.CHUNK_HEADER + size
            if size % GlbLayout.ALIGNMENT or end > length:
                raise InvalidLaboratoryAsset(path, "Invalid glTF chunk length")
            if offset == GlbLayout.HEADER:
                if kind != b"JSON":
                    raise InvalidLaboratoryAsset(path, "First glTF chunk is not JSON")
                json_end = end
            offset = end
        document = json.loads(
            content[GlbLayout.HEADER + GlbLayout.CHUNK_HEADER : json_end]
        )
        asset = document.get(AssetField.ASSET)
        if not isinstance(asset, dict):
            raise InvalidLaboratoryAsset(path, "Missing glTF asset object")
        extras = asset.setdefault(AssetField.EXTRAS, {})
        if not isinstance(extras, dict):
            raise InvalidLaboratoryAsset(
                path, "Cannot preserve non-object glTF asset extras"
            )
        previous = asset.get(AssetField.COPYRIGHT)
        copyright = metadata[RightsField.COPYRIGHT]
        if previous and previous != copyright:
            prior = extras.setdefault(AssetField.PRIOR_COPYRIGHT, [])
            if (
                not isinstance(prior, list)
                or not all(isinstance(item, str) for item in prior)
                or not isinstance(previous, str)
            ):
                raise InvalidLaboratoryAsset(
                    path, "Cannot preserve existing copyright representation"
                )
            if previous not in prior:
                prior.append(previous)
        asset[AssetField.COPYRIGHT] = copyright
        extras[AssetField.RIGHTS] = metadata
        encoded = json.dumps(
            document, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        encoded += b" " * (-len(encoded) % GlbLayout.ALIGNMENT)
        trailing = content[json_end:]
        length = (
            GlbLayout.HEADER + GlbLayout.CHUNK_HEADER + len(encoded) + len(trailing)
        )
        path.write_bytes(
            struct.pack("<4sII", b"glTF", GlbLayout.VERSION, length)
            + struct.pack("<I4s", len(encoded), b"JSON")
            + encoded
            + trailing
        )
