"""Runtime types for compiling action phases into pose sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

from geometry_msgs.msg import PoseStamped


class PrimitiveFamily(Enum):
    """Supported primitive families for phase compilation."""

    MATERIAL_TRANSFER_DISCHARGE = auto()
    MATERIAL_TRANSFER_SHAKE = auto()
    SEPARATION_CONTACT = auto()
    SURFACE_SCRUB_CIRCLE = auto()
    SURFACE_WIPE_RASTER_SCRUB = auto()
    VOLUME_AGITATION = auto()


class AnchorKey(Enum):
    """Keys used to resolve anchors for phases."""

    CUT_PLANE = auto()
    TOOL_CONTACT = auto()
    POUR_BOUNDARY = auto()
    CONTAINER_INTERIOR = auto()


class ParamKey(Enum):
    """Keys used to resolve parameter bundles for phases."""

    CUT_SPEC = auto()
    CUT_ALIGN_SPEC = auto()
    CUT_RETREAT_SPEC = auto()
    POUR_DISCHARGE_SPEC = auto()
    POUR_SHAKE_SPEC = auto()
    AGITATION_SPEC = auto()


@dataclass(frozen=True)
class PhaseInstance:
    """Phase instance with bound anchors and parameters."""

    family: PrimitiveFamily
    anchor: Any
    params: Any
    time_span: Optional[Tuple[float, float]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MissingAnchorError(KeyError):
    """Raised when an anchor cannot be resolved."""

    key: AnchorKey

    def __str__(self) -> str:
        return f"Missing anchor for key={self.key}."


@dataclass(frozen=True)
class MissingParameterError(KeyError):
    """Raised when parameters cannot be resolved."""

    key: ParamKey

    def __str__(self) -> str:
        return f"Missing parameters for key={self.key}."


@dataclass(frozen=True)
class MissingCompilerError(KeyError):
    """Raised when no compiler exists for a primitive family."""

    family: PrimitiveFamily

    def __str__(self) -> str:
        return f"No compiler registered for family={self.family}."


K = TypeVar("K", bound=Enum)


class Resolver(Protocol[K]):
    def resolve(self, key: K, **kwargs: Any) -> Any: ...


def _resolve_from(
    obj: Union[Mapping[K, Any], Resolver[K]], key: K, **kwargs: Any
) -> Any:
    """Resolve a key from a mapping or resolver implementation."""
    if hasattr(obj, "resolve"):
        return obj.resolve(key, **kwargs)
    if key not in obj:
        raise KeyError(key)
    return obj[key]


@dataclass(frozen=True)
class PhaseSpec:
    """Specification describing how to bind a phase at runtime."""

    family: PrimitiveFamily
    anchor_key: AnchorKey
    param_key: ParamKey
    time_span: Optional[Tuple[float, float]] = None
    meta: Optional[Dict[str, Any]] = None

    def bind(
        self,
        anchors: Union[Mapping[AnchorKey, Any], Resolver[AnchorKey]],
        params: Union[Mapping[ParamKey, Any], Resolver[ParamKey]],
        **kwargs: Any,
    ) -> PhaseInstance:
        try:
            a = _resolve_from(anchors, self.anchor_key, **kwargs)
        except KeyError as exc:
            raise MissingAnchorError(self.anchor_key) from exc
        try:
            p = _resolve_from(params, self.param_key, **kwargs)
        except KeyError as exc:
            raise MissingParameterError(self.param_key) from exc
        return PhaseInstance(
            family=self.family,
            anchor=a,
            params=p,
            time_span=self.time_span,
            meta=self.meta,
        )


class PhaseCompiler(Protocol):
    """Compiles a bound phase into a pose sequence."""

    def __call__(self, anchor: Any, params: Any) -> Iterable[PoseStamped]: ...


@dataclass(frozen=True)
class CompilerRegistry:
    """Registry for primitive family compilers."""

    compilers: Dict[PrimitiveFamily, PhaseCompiler]

    def compile(self, phase: PhaseInstance) -> Iterable[PoseStamped]:
        if phase.family not in self.compilers:
            raise MissingCompilerError(phase.family)
        return self.compilers[phase.family](phase.anchor, phase.params)


class AnchorResolver(Protocol):
    """Resolves anchors by key."""

    def resolve(self, key: AnchorKey, **kwargs: Any) -> Any: ...


class ParamResolver(Protocol):
    """Resolves parameters by key."""

    def resolve(self, key: ParamKey, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ActionProgram:
    """Ordered phase instances that can be compiled to poses."""

    phases: Tuple[PhaseInstance, ...]

    def compile(self, registry: CompilerRegistry) -> Tuple[PoseStamped, ...]:
        out: list[PoseStamped] = []
        for phase in self.phases:
            out.extend(list(registry.compile(phase)))
        return tuple(out)


# from __future__ import annotations
#
# from dataclasses import dataclass
# from enum import Enum, auto
# from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Sequence, Tuple
#
# import numpy as np
# from geometry_msgs.msg import PoseStamped
#
#
# class PrimitiveFamily(Enum):
#     MATERIAL_TRANSFER_DISCHARGE = auto()
#     MATERIAL_TRANSFER_SHAKE = auto()
#     SEPARATION_CONTACT = auto()
#     SURFACE_SCRUB_CIRCLE = auto()
#     SURFACE_WIPE_RASTER_SCRUB = auto()
#     VOLUME_AGITATION = auto()
#
#
# @dataclass(frozen=True)
# class PhaseInstance:
#     family: PrimitiveFamily
#     anchor: Any
#     params: Any
#     time_span: Optional[Tuple[float, float]] = None
#     meta: Optional[Dict[str, Any]] = None
#
#
# class PhaseCompiler(Protocol):
#     def __call__(self, anchor: Any, params: Any) -> Iterable[PoseStamped]: ...
#
#
# @dataclass
# class CompilerRegistry:
#     compilers: Dict[PrimitiveFamily, PhaseCompiler]
#
#     def compile(self, phase: PhaseInstance) -> Iterable[PoseStamped]:
#         if phase.family not in self.compilers:
#             raise KeyError(f"no compiler for family={phase.family}")
#         return self.compilers[phase.family](phase.anchor, phase.params)
#
#
# class AnchorResolver(Protocol):
#     def resolve(self, key: str, **kwargs: Any) -> Any: ...
#
#
# class ParamResolver(Protocol):
#     def resolve(self, key: str, **kwargs: Any) -> Any: ...
#
#
# @dataclass(frozen=True)
# class PhaseSpec:
#     family: PrimitiveFamily
#     anchor_key: str
#     param_key: str
#     time_span: Optional[Tuple[float, float]] = None
#     meta: Optional[Dict[str, Any]] = None
#
#     def bind(
#         self,
#         anchors: AnchorResolver,
#         params: ParamResolver,
#         **kwargs: Any,
#     ) -> PhaseInstance:
#         a = anchors.resolve(self.anchor_key, **kwargs)
#         p = params.resolve(self.param_key, **kwargs)
#         return PhaseInstance(
#             family=self.family,
#             anchor=a,
#             params=p,
#             time_span=self.time_span,
#             meta=self.meta,
#         )
#
#
# @dataclass(frozen=True)
# class ActionProgram:
#     phases: Tuple[PhaseInstance, ...]
#
#     def compile(self, registry: CompilerRegistry) -> Tuple[PoseStamped, ...]:
#         out: list[PoseStamped] = []
#         for ph in self.phases:
#             out.extend(list(registry.compile(ph)))
#         return tuple(out)
