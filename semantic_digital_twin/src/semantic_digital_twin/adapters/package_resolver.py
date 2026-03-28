from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache

from typing_extensions import List

from semantic_digital_twin.exceptions import ParsingError


class PackageLocator(ABC):
    """
    Abstract base class for package locators.
    """

    @abstractmethod
    def resolve(self, package_name: str) -> str:
        """
        Resolves a package name to its local filesystem path.
        """


@dataclass
class AmentPackageLocator(PackageLocator):
    """
    Resolves packages using ament.
    """

    def resolve(self, package_name: str) -> str:
        try:
            from ament_index_python.packages import get_package_share_directory

            return get_package_share_directory(package_name)
        except (ImportError, LookupError) as error:
            raise ParsingError(
                message=f"Ament could not resolve package '{package_name}': {error}"
            )


@dataclass
class ROSPackagePathLocator(PackageLocator):
    """
    Resolves packages using ROS_PACKAGE_PATH.
    """

    def resolve(self, package_name: str) -> str:
        for root in os.environ.get("ROS_PACKAGE_PATH", "").split(":"):
            if not root:
                continue
            candidate = os.path.join(root, package_name)
            if os.path.isdir(candidate):
                return candidate
        raise ParsingError(
            message=f"Package '{package_name}' not found in ROS_PACKAGE_PATH."
        )


@dataclass
class ColconSourcePackageLocator(PackageLocator):
    """
    Resolves packages directly from a colcon workspace source tree.

    This is a fallback for environments where package overlays are sourced well
    enough to expose install prefixes, but the package itself is only present in
    `src/` and therefore not discoverable via ament or ROS_PACKAGE_PATH.
    """

    def resolve(self, package_name: str) -> str:
        for workspace_src in self._candidate_src_roots():
            package_path = self._find_package_in_src(workspace_src, package_name)
            if package_path is not None:
                return package_path
        raise ParsingError(
            message=(
                f"Package '{package_name}' not found in discovered colcon source "
                "workspaces."
            )
        )

    def _candidate_src_roots(self) -> List[str]:
        prefixes = []
        for env_var in ("COLCON_PREFIX_PATH", "AMENT_PREFIX_PATH"):
            prefixes.extend(
                prefix
                for prefix in os.environ.get(env_var, "").split(":")
                if prefix.strip()
            )

        src_roots = []
        for prefix in prefixes:
            workspace_root = self._workspace_root_from_prefix(prefix)
            if workspace_root is None:
                continue
            src_root = os.path.join(workspace_root, "src")
            if os.path.isdir(src_root) and src_root not in src_roots:
                src_roots.append(src_root)
        return src_roots

    @staticmethod
    def _workspace_root_from_prefix(prefix: str) -> str | None:
        normalized = os.path.abspath(prefix)
        if os.path.basename(normalized) == "install":
            return os.path.dirname(normalized)
        parent = os.path.dirname(normalized)
        if os.path.basename(parent) == "install":
            return os.path.dirname(parent)
        return None

    @staticmethod
    @lru_cache(maxsize=32)
    def _find_package_in_src(src_root: str, package_name: str) -> str | None:
        for root, dirs, files in os.walk(src_root):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in {".git", ".hg", ".svn", "__pycache__", "build", "install", "log"}
            ]

            if "package.xml" not in files:
                continue

            if os.path.basename(root) == package_name:
                return root

            package_xml_path = os.path.join(root, "package.xml")
            try:
                with open(package_xml_path, "r", encoding="utf-8") as package_xml:
                    if f"<name>{package_name}</name>" in package_xml.read():
                        return root
            except OSError:
                continue
        return None


@dataclass
class ROSPackageLocator(PackageLocator):
    """
    Tries multiple package locators in order.
    """

    locators: List[PackageLocator] = field(
        default_factory=lambda: [
            AmentPackageLocator(),
            ROSPackagePathLocator(),
            ColconSourcePackageLocator(),
        ]
    )

    def resolve(self, package_name: str) -> str:
        errors = []
        for locator in self.locators:
            try:
                return locator.resolve(package_name)
            except ParsingError as error:
                errors.append(str(error))
        raise ParsingError(
            message=f"Could not resolve package '{package_name}'. Details: {'; '.join(errors)}"
        )


class PathResolver(ABC):
    """
    Abstract base class for path resolvers.
    """

    @abstractmethod
    def supports(self, uri: str) -> bool:
        """
        Checks if the URI is supported by this resolver.
        """

    @abstractmethod
    def resolve(self, uri: str) -> str:
        """
        Resolves a URI to an absolute local file path.
        """


@dataclass
class PackageUriResolver(PathResolver):
    """
    Resolves package:// URIs.
    """

    locator: PackageLocator = field(default_factory=ROSPackageLocator)

    def supports(self, uri: str) -> bool:
        return uri.startswith("package://")

    def resolve(self, uri: str) -> str:
        rest = uri[len("package://") :]
        if "/" not in rest:
            package_name, relative_path = rest, ""
        else:
            package_name, relative_path = rest.split("/", 1)
        base = self.locator.resolve(package_name)
        return os.path.join(base, relative_path)


@dataclass
class FileUriResolver(PathResolver):
    """
    Resolves file:// URIs and plain filesystem paths.
    """

    def supports(self, uri: str) -> bool:
        return uri.startswith("file://") or uri.startswith("/") or "://" not in uri

    def resolve(self, uri: str) -> str:
        path = uri
        if uri.startswith("file://"):
            path = (
                uri.replace("file://", "./", 1)
                if not uri.startswith("file:///")
                else uri[len("file://") :]
            )
        return path


@dataclass
class CompositePathResolver(PathResolver):
    """
    Tries multiple path resolvers in order.
    """

    resolvers: List[PathResolver] = field(
        default_factory=lambda: [
            FileUriResolver(),
            PackageUriResolver(),
        ]
    )

    def supports(self, uri: str) -> bool:
        """
        Checks if the URI is supported by any of the resolvers.
        """
        return any(resolver.supports(uri) for resolver in self.resolvers)

    def resolve(self, uri: str) -> str:
        """
        Resolves a URI to an absolute local file path.
        """
        errors = []
        for resolver in self.resolvers:
            if not resolver.supports(uri):
                continue
            try:
                return resolver.resolve(uri)
            except ParsingError as error:
                errors.append(str(error))

        raise ParsingError(
            message=f"Could not resolve path '{uri}'. Details: {'; '.join(errors)}"
        )
