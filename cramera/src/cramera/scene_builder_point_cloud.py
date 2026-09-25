"""Validate bounded PLY and PCD reference clouds without altering their samples."""

from __future__ import annotations

import io
import math
import re
import struct
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import ClassVar


# %% supported formats and limits
class PointCloudFormat(StrEnum):
    """Upload formats accepted as persistent scene references."""

    PLY = "ply"
    """Stanford polygon files containing scalar vertex records."""
    PCD = "pcd"
    """Point Cloud Library files with uncompressed point records."""


class PointCloudEncoding(StrEnum):
    """Supported on-disk point record encodings."""

    ASCII = "ascii"
    """Whitespace-delimited scalar records."""
    BINARY = "binary"
    """PCD little-endian interleaved binary records."""
    LITTLE_ENDIAN = "binary_little_endian"
    """PLY little-endian binary records."""
    BIG_ENDIAN = "binary_big_endian"
    """PLY big-endian binary records."""


class PointCloudField(StrEnum):
    """Persistent resource and display metadata field names."""

    ROOT = "point_clouds"
    DIRECTORY = "point-clouds"
    METADATA_SUFFIX = ".metadata.json"
    POINT_COUNT = "pointCount"
    FINITE_POINT_COUNT = "finitePointCount"
    ENCODING = "encoding"
    VISIBLE = "visible"
    LOCKED = "locked"
    POINT_SIZE = "pointSize"
    OPACITY = "opacity"


@dataclass(frozen=True)
class PointCloudLimits:
    """Resource limits shared by upload transport, validation and display settings."""

    MAXIMUM_BYTES: ClassVar[int] = 64 * 1024 * 1024
    """Largest complete uploaded source file."""
    MAXIMUM_POINTS: ClassVar[int] = 2_000_000
    """Largest declared source cloud before display downsampling."""
    MAXIMUM_HEADER_BYTES: ClassVar[int] = 64 * 1024
    """Largest ASCII header before point records begin."""
    MAXIMUM_HEADER_LINE_BYTES: ClassVar[int] = 8192
    """Largest individual header declaration."""
    MAXIMUM_RECORD_BYTES: ClassVar[int] = 64 * 1024
    """Largest textual point record."""
    MAXIMUM_FIELDS: ClassVar[int] = 64
    """Largest number of named scalar fields in a point."""
    MAXIMUM_COMPONENTS: ClassVar[int] = 1024
    """Largest combined PCD field count in a point."""
    FLOAT32_MAXIMUM: ClassVar[float] = float.fromhex("0x1.fffffep+127")
    """Largest finite coordinate supported by a browser Float32 position buffer."""
    MINIMUM_POINT_SIZE: ClassVar[float] = 0.0001
    """Smallest visible point size in metres."""
    MAXIMUM_POINT_SIZE: ClassVar[float] = 0.1
    """Largest visible point size in metres."""
    MINIMUM_OPACITY: ClassVar[float] = 0.05
    """Smallest saved point-cloud opacity."""
    MAXIMUM_OPACITY: ClassVar[float] = 1.0
    """Largest saved point-cloud opacity."""
    DEFAULT_POINT_SIZE: ClassVar[float] = 0.006
    """Initial scan-reference point size in metres."""
    DEFAULT_OPACITY: ClassVar[float] = 0.65
    """Initial translucent scan-reference opacity."""


class InvalidPointCloud(ValueError):
    """A source cloud is malformed, unsupported or exceeds the resource limits."""


class ScalarType(StrEnum):
    """Standard binary scalar representations accepted in source records."""

    SIGNED_BYTE = "b"
    UNSIGNED_BYTE = "B"
    SIGNED_SHORT = "h"
    UNSIGNED_SHORT = "H"
    SIGNED_INTEGER = "i"
    UNSIGNED_INTEGER = "I"
    FLOAT = "f"
    DOUBLE = "d"

    @property
    def size(self) -> int:
        """Return the standard scalar size without native alignment padding."""
        return struct.calcsize("<" + self.value)

    @property
    def floating(self) -> bool:
        """Return whether this scalar represents a floating-point value."""
        return self in (ScalarType.FLOAT, ScalarType.DOUBLE)

    @classmethod
    def polygon(cls, name: str) -> ScalarType:
        """Resolve the standard PLY scalar spelling and its sized aliases."""
        types = {
            "char": cls.SIGNED_BYTE,
            "int8": cls.SIGNED_BYTE,
            "uchar": cls.UNSIGNED_BYTE,
            "uint8": cls.UNSIGNED_BYTE,
            "short": cls.SIGNED_SHORT,
            "int16": cls.SIGNED_SHORT,
            "ushort": cls.UNSIGNED_SHORT,
            "uint16": cls.UNSIGNED_SHORT,
            "int": cls.SIGNED_INTEGER,
            "int32": cls.SIGNED_INTEGER,
            "uint": cls.UNSIGNED_INTEGER,
            "uint32": cls.UNSIGNED_INTEGER,
            "float": cls.FLOAT,
            "float32": cls.FLOAT,
            "double": cls.DOUBLE,
            "float64": cls.DOUBLE,
        }
        if name not in types:
            raise InvalidPointCloud("Unsupported PLY scalar type.")
        return types[name]

    @classmethod
    def cloud(cls, category: str, size: int) -> ScalarType:
        """Resolve one PCD scalar's category and byte width."""
        types = {
            ("I", 1): cls.SIGNED_BYTE,
            ("I", 2): cls.SIGNED_SHORT,
            ("I", 4): cls.SIGNED_INTEGER,
            ("U", 1): cls.UNSIGNED_BYTE,
            ("U", 2): cls.UNSIGNED_SHORT,
            ("U", 4): cls.UNSIGNED_INTEGER,
            ("F", 4): cls.FLOAT,
            ("F", 8): cls.DOUBLE,
        }
        if (category, size) not in types:
            raise InvalidPointCloud("Unsupported PCD field type or size.")
        return types[(category, size)]

    def parse(self, token: bytes) -> int | float:
        """Read a numeric ASCII scalar and enforce its declared integer range."""
        if len(token) > 128:
            raise InvalidPointCloud("A point scalar exceeds the numeric token limit.")
        if self.floating:
            if not re.fullmatch(
                rb"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|infinity)",
                token,
                re.IGNORECASE,
            ):
                raise InvalidPointCloud("A point record contains an invalid number.")
            value = float(token)
            if (
                self == ScalarType.FLOAT
                and math.isfinite(value)
                and abs(value) > PointCloudLimits.FLOAT32_MAXIMUM
            ):
                raise InvalidPointCloud("A scalar exceeds its declared Float32 range.")
            return value
        if not re.fullmatch(rb"[+-]?\d+", token):
            raise InvalidPointCloud(
                "An integer point field contains a non-integer value."
            )
        value = int(token)
        signed = self.value.islower()
        bits = self.size * 8
        minimum = -(1 << (bits - 1)) if signed else 0
        maximum = (1 << (bits - (1 if signed else 0))) - 1
        if not minimum <= value <= maximum:
            raise InvalidPointCloud(
                "An integer point field exceeds its declared range."
            )
        return value


# %% bounded source header
@dataclass(frozen=True)
class PointField:
    """One named source dimension and its contiguous scalar components."""

    name: str
    """Declared field name such as x, z or rgb."""
    scalar: ScalarType
    """Declared scalar representation."""
    count: int = 1
    """Number of contiguous components in this field."""


@dataclass
class HeaderLines:
    """Read a bounded ASCII header while retaining the exact payload byte offset."""

    body: bytes
    """Complete unmodified source file."""
    offset: int = 0
    """Byte offset following the most recently consumed line."""

    def read(self) -> str:
        """Consume one complete ASCII header line within declared byte bounds."""
        end = self.body.find(
            b"\n", self.offset, PointCloudLimits.MAXIMUM_HEADER_BYTES + 1
        )
        if end < 0 or end + 1 > PointCloudLimits.MAXIMUM_HEADER_BYTES:
            raise InvalidPointCloud(
                "Point-cloud header is missing, truncated or too large."
            )
        line = self.body[self.offset : end]
        self.offset = end + 1
        if (
            len(line) > PointCloudLimits.MAXIMUM_HEADER_LINE_BYTES
            or not line.isascii()
            or b"\x00" in line
        ):
            raise InvalidPointCloud(
                "Point-cloud header lines must be bounded ASCII text."
            )
        return line.decode("ascii").strip()


def declared_count(value: str, maximum: int, *, minimum: int = 0) -> int:
    """Read a bounded nonnegative header integer without arbitrary-size conversion."""
    if not re.fullmatch(r"\d{1,10}", value):
        raise InvalidPointCloud("Invalid point-cloud count declaration.")
    count = int(value)
    if not minimum <= count <= maximum:
        raise InvalidPointCloud("Point-cloud count exceeds the supported limits.")
    return count


@dataclass(frozen=True)
class PointCloudHeader:
    """Validated point-record layout and payload location."""

    encoding: PointCloudEncoding
    """Declared text or binary storage mode."""
    point_count: int
    """Number of records declared by the source header."""
    fields: tuple[PointField, ...]
    """Field order used by every source record."""
    offset: int
    """Byte offset at which the first record begins."""

    def __post_init__(self) -> None:
        """Require distinct fields and three scalar coordinate dimensions."""
        names = [entry.name for entry in self.fields]
        if (
            not self.fields
            or len(self.fields) > PointCloudLimits.MAXIMUM_FIELDS
            or len(names) != len(set(names))
        ):
            raise InvalidPointCloud("Point-cloud fields must be distinct and bounded.")
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name) for name in names
        ):
            raise InvalidPointCloud("Invalid point-cloud field name.")
        by_name = {entry.name: entry for entry in self.fields}
        if any(
            axis not in by_name or by_name[axis].count != 1 for axis in ("x", "y", "z")
        ):
            raise InvalidPointCloud(
                "A reference cloud requires scalar x, y and z fields."
            )
        if (
            sum(entry.count for entry in self.fields)
            > PointCloudLimits.MAXIMUM_COMPONENTS
        ):
            raise InvalidPointCloud("Too many scalar components per point.")

    @classmethod
    def polygon(cls, lines: HeaderLines) -> PointCloudHeader:
        """Parse scalar PLY vertex clouds with optional empty extra elements."""
        if lines.read() != "ply":
            raise InvalidPointCloud("A PLY source must begin with its format marker.")
        encoding = None
        point_count = None
        element = None
        fields = []
        while True:
            tokens = lines.read().split()
            if not tokens:
                raise InvalidPointCloud("Unexpected empty PLY header declaration.")
            directive = tokens[0]
            if directive in {"comment", "obj_info"}:
                continue
            if directive == "format":
                if (
                    encoding is not None
                    or len(tokens) != 3
                    or tokens[2] != "1.0"
                    or tokens[1]
                    not in {
                        PointCloudEncoding.ASCII,
                        PointCloudEncoding.LITTLE_ENDIAN,
                        PointCloudEncoding.BIG_ENDIAN,
                    }
                ):
                    raise InvalidPointCloud("Unsupported PLY format declaration.")
                encoding = PointCloudEncoding(tokens[1])
            elif directive == "element":
                if len(tokens) != 3 or encoding is None:
                    raise InvalidPointCloud("Invalid PLY element declaration.")
                element = tokens[1]
                count = declared_count(tokens[2], PointCloudLimits.MAXIMUM_POINTS)
                if element == "vertex":
                    if point_count is not None or count == 0:
                        raise InvalidPointCloud(
                            "A PLY cloud requires one nonempty vertex element."
                        )
                    point_count = count
                elif count != 0:
                    raise InvalidPointCloud(
                        "Only PLY vertex clouds and empty extra elements are supported."
                    )
            elif directive == "property":
                if element is None:
                    raise InvalidPointCloud("A PLY property requires an element.")
                if element != "vertex":
                    continue
                if len(tokens) != 3 or tokens[1] == "list":
                    raise InvalidPointCloud(
                        "PLY vertex list properties are unsupported."
                    )
                fields.append(
                    PointField(name=tokens[2], scalar=ScalarType.polygon(tokens[1]))
                )
                if len(fields) > PointCloudLimits.MAXIMUM_FIELDS:
                    raise InvalidPointCloud("Too many PLY vertex fields.")
            elif directive == "end_header":
                if len(tokens) != 1 or encoding is None or point_count is None:
                    raise InvalidPointCloud("Incomplete PLY header.")
                return cls(
                    encoding=encoding,
                    point_count=point_count,
                    fields=tuple(fields),
                    offset=lines.offset,
                )
            else:
                raise InvalidPointCloud("Unknown PLY header declaration.")

    @classmethod
    def cloud(cls, lines: HeaderLines) -> PointCloudHeader:
        """Parse uncompressed PCD fields and consistent organized dimensions."""
        declarations = {}
        allowed = {
            "VERSION",
            "FIELDS",
            "SIZE",
            "TYPE",
            "COUNT",
            "WIDTH",
            "HEIGHT",
            "VIEWPOINT",
            "POINTS",
            "DATA",
        }
        while True:
            tokens = lines.read().split("#", 1)[0].split()
            if not tokens:
                continue
            directive = tokens[0]
            if directive not in allowed or directive in declarations:
                raise InvalidPointCloud("Unknown or repeated PCD header declaration.")
            declarations[directive] = tokens[1:]
            if directive == "DATA":
                break
        required = {
            "VERSION",
            "FIELDS",
            "SIZE",
            "TYPE",
            "WIDTH",
            "HEIGHT",
            "POINTS",
            "DATA",
        }
        if not required.issubset(declarations):
            raise InvalidPointCloud("Incomplete PCD header.")
        for scalar in ("VERSION", "WIDTH", "HEIGHT", "POINTS", "DATA"):
            if len(declarations[scalar]) != 1:
                raise InvalidPointCloud("Invalid scalar PCD header declaration.")
        if declarations["VERSION"][0] not in {".7", "0.7"}:
            raise InvalidPointCloud("Only PCD version 0.7 is supported.")
        storage = declarations["DATA"][0]
        if storage not in {PointCloudEncoding.ASCII, PointCloudEncoding.BINARY}:
            raise InvalidPointCloud(
                "PCD binary_compressed is unsupported; export ASCII or binary PCD."
            )
        point_count = declared_count(
            declarations["POINTS"][0], PointCloudLimits.MAXIMUM_POINTS, minimum=1
        )
        width = declared_count(
            declarations["WIDTH"][0], PointCloudLimits.MAXIMUM_POINTS, minimum=1
        )
        height = declared_count(
            declarations["HEIGHT"][0], PointCloudLimits.MAXIMUM_POINTS, minimum=1
        )
        if width * height != point_count:
            raise InvalidPointCloud(
                "PCD width and height do not match its point count."
            )
        names = declarations["FIELDS"]
        sizes = declarations["SIZE"]
        categories = declarations["TYPE"]
        counts = declarations.get("COUNT", ["1"] * len(names))
        if not len(names) == len(sizes) == len(categories) == len(counts):
            raise InvalidPointCloud("PCD field declarations have inconsistent lengths.")
        if len(names) > PointCloudLimits.MAXIMUM_FIELDS:
            raise InvalidPointCloud("Too many PCD fields.")
        if "VIEWPOINT" in declarations:
            viewpoint = declarations["VIEWPOINT"]
            if len(viewpoint) != 7 or any(
                not math.isfinite(ScalarType.DOUBLE.parse(token.encode()))
                for token in viewpoint
            ):
                raise InvalidPointCloud("PCD viewpoint requires seven finite values.")
        fields = tuple(
            PointField(
                name=name,
                scalar=ScalarType.cloud(category, declared_count(size, 8, minimum=1)),
                count=declared_count(
                    count, PointCloudLimits.MAXIMUM_COMPONENTS, minimum=1
                ),
            )
            for name, size, category, count in zip(names, sizes, categories, counts)
        )
        return cls(
            encoding=PointCloudEncoding(storage),
            point_count=point_count,
            fields=fields,
            offset=lines.offset,
        )


# %% complete source record validation
@dataclass(frozen=True)
class PointCloudSource:
    """Validate source bytes and count usable XYZ measurements."""

    header: PointCloudHeader
    """Validated on-disk point layout."""
    finite_point_count: int
    """Points whose three coordinates are usable browser positions."""

    @classmethod
    def validate(cls, format_name: str, body: bytes) -> PointCloudSource:
        """Reject malformed, excessive or unsupported input before any file is saved."""
        if (
            format_name not in PointCloudFormat
            or not isinstance(body, bytes)
            or not body
            or len(body) > PointCloudLimits.MAXIMUM_BYTES
        ):
            raise InvalidPointCloud(
                "Unsupported point-cloud format or source byte size."
            )
        lines = HeaderLines(body=body)
        header = (
            PointCloudHeader.polygon(lines)
            if format_name == PointCloudFormat.PLY
            else PointCloudHeader.cloud(lines)
        )
        components = tuple(
            scalar for entry in header.fields for scalar in [entry.scalar] * entry.count
        )
        indices = {}
        offset = 0
        for entry in header.fields:
            if entry.name in {"x", "y", "z"}:
                indices[entry.name] = offset
            offset += entry.count
        coordinates = tuple(indices[axis] for axis in ("x", "y", "z"))
        finite_count = (
            cls._ascii(body, header, components, coordinates)
            if header.encoding == PointCloudEncoding.ASCII
            else cls._binary(body, header, components, coordinates)
        )
        if finite_count == 0:
            raise InvalidPointCloud(
                "A reference cloud requires at least one finite XYZ measurement."
            )
        return cls(header=header, finite_point_count=finite_count)

    @staticmethod
    def _finite(values: tuple | list, coordinates: tuple[int, ...]) -> bool:
        """Accept missing NaNs while rejecting infinities and unrenderable coordinates."""
        finite = True
        for index in coordinates:
            value = values[index]
            if math.isnan(value):
                finite = False
                continue
            if (
                not math.isfinite(value)
                or abs(value) > PointCloudLimits.FLOAT32_MAXIMUM
            ):
                raise InvalidPointCloud(
                    "XYZ coordinates exceed the finite Float32 display range."
                )
        return finite

    @classmethod
    def _ascii(
        cls,
        body: bytes,
        header: PointCloudHeader,
        components: tuple[ScalarType, ...],
        coordinates: tuple[int, ...],
    ) -> int:
        """Validate exactly the declared number of whitespace-delimited point rows."""
        source = io.BytesIO(body)
        source.seek(header.offset)
        count = 0
        finite_count = 0
        while True:
            line = source.readline(PointCloudLimits.MAXIMUM_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > PointCloudLimits.MAXIMUM_RECORD_BYTES:
                raise InvalidPointCloud("Point-cloud data record is too large.")
            tokens = line.split()
            if not tokens:
                continue
            count += 1
            if count > header.point_count or len(tokens) != len(components):
                raise InvalidPointCloud(
                    "ASCII point records do not match their declared layout."
                )
            values = [scalar.parse(token) for scalar, token in zip(components, tokens)]
            finite_count += cls._finite(values, coordinates)
        if count != header.point_count:
            raise InvalidPointCloud("ASCII point-cloud data is truncated.")
        return finite_count

    @classmethod
    def _binary(
        cls,
        body: bytes,
        header: PointCloudHeader,
        components: tuple[ScalarType, ...],
        coordinates: tuple[int, ...],
    ) -> int:
        """Validate exact binary record length and all XYZ measurements."""
        order = ">" if header.encoding == PointCloudEncoding.BIG_ENDIAN else "<"
        record = struct.Struct(order + "".join(components))
        if len(body) - header.offset != header.point_count * record.size:
            raise InvalidPointCloud(
                "Binary point records are truncated or contain trailing bytes."
            )
        return sum(
            cls._finite(values, coordinates)
            for values in record.iter_unpack(memoryview(body)[header.offset :])
        )

    def metadata(self) -> dict:
        """Describe source counts without changing the original sample representation."""
        return {
            PointCloudField.POINT_COUNT: self.header.point_count,
            PointCloudField.FINITE_POINT_COUNT: self.finite_point_count,
            PointCloudField.ENCODING: self.header.encoding,
        }
