#!/usr/bin/env python3
"""
bundle_urdf.py — make a URDF (or xacro) self-contained for the web viewer.

Resolves every mesh reference (package://, file://, absolute or relative),
copies the meshes plus their side assets (textures for .dae, .mtl + textures
for .obj) into <out>/meshes/..., rewrites the references to those relative
paths, and writes <out>/<name>.urdf. The result loads in the browser with no
ROS installed.

Standalone use:
    python3 -m cram_viz.onboard.bundle_urdf path/or/package://...  --name apartment \
        --out static/scenes/my_scene

It is also imported by :mod:`cram_viz.onboard.demo` (the ``cram-viz-onboard``
console script), which feeds it the exact uri->path resolutions recorded
while the demo ran.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

MESH_EXTS = (".dae", ".stl", ".obj")


# %% resolution
def _search_root_candidates() -> list[str]:
    """
    Likely ROS install prefixes to search for a package:// URI: env vars first, then
    common workspace layouts under the home directory and /opt/ros.
    """
    roots = []
    for environment_variable in ("AMENT_PREFIX_PATH", "ROS_PACKAGE_PATH", "CMAKE_PREFIX_PATH"):
        roots += [entry for entry in os.environ.get(environment_variable, "").split(":") if entry]
    home = os.path.expanduser("~")
    roots += glob.glob(os.path.join(home, "*_ws", "install"))
    roots += glob.glob(os.path.join(home, "*", "install"))
    roots += glob.glob("/opt/ros/*")
    return roots


def resolve_uri(
    uri: str, hints: dict[str, str] | None = None, base_dir: str | None = None
) -> str | None:
    """
    Resolve a mesh/urdf reference to an absolute file path (or None).
    """
    if hints and uri in hints:
        return hints[uri]
    if uri.startswith("package://"):
        rest = uri[len("package://"):]
        package_name, _, relative_path = rest.partition("/")
        # 1. the CRAM stack's own resolver (ament index), if importable
        try:
            from semantic_digital_twin.adapters.package_resolver import (
                PackageUriResolver,
            )
            from semantic_digital_twin.exceptions import PackageResolutionError
        except ImportError:
            pass
        else:
            try:
                resolved_path = PackageUriResolver().resolve(uri)
            except PackageResolutionError:
                resolved_path = None
            if resolved_path and os.path.isfile(resolved_path):
                return resolved_path
        # 2. ament index directly
        try:
            from ament_index_python.packages import (
                PackageNotFoundError,
                get_package_share_directory,
            )
        except ImportError:
            pass
        else:
            try:
                package_share_directory = get_package_share_directory(package_name)
            except PackageNotFoundError:
                package_share_directory = None
            if package_share_directory:
                resolved_path = os.path.join(package_share_directory, relative_path)
                if os.path.isfile(resolved_path):
                    return resolved_path
        # 3. filesystem heuristics over common workspace layouts
        for root in _search_root_candidates():
            for candidate in (
                os.path.join(root, package_name, "share", package_name, relative_path),
                os.path.join(root, "share", package_name, relative_path),
                os.path.join(root, package_name, relative_path),
            ):
                if os.path.isfile(candidate):
                    return candidate
        return None
    if uri.startswith("file://"):
        path = uri[len("file://"):]
        return path if os.path.isfile(path) else None
    if os.path.isabs(uri):
        return uri if os.path.isfile(uri) else None
    if base_dir:
        path = os.path.join(base_dir, uri)
        return path if os.path.isfile(path) else None
    return None


def _ref_to_relpath(uri: str, resolved_path: str | None) -> str:
    """
    Where a reference lands inside <out>/meshes/....

    Package refs keep their package/relative-path structure. Local/file/absolute refs
    are disambiguated by a short hash of the resolved source path, since two distinct
    files can otherwise share a basename (e.g. two "book.stl" in different directories)
    and silently overwrite each other.
    """
    if uri.startswith("package://"):
        rest = uri[len("package://"):]
        package_name, _, relative_path = rest.partition("/")
        return os.path.join(package_name, relative_path)
    name = uri[len("file://"):] if uri.startswith("file://") else uri
    basename = os.path.basename(name)
    if not resolved_path:
        return os.path.join("_local", basename)
    digest = hashlib.sha1(resolved_path.encode("utf-8")).hexdigest()[:8]
    return os.path.join("_local", digest, basename)


# %% side assets
def _copy_file(
    source_path: str | None, destination_path: str, copied: dict[str, str], missing: list[str]
) -> bool:
    """
    Copy source_path to destination_path once; record it in copied (memo) or missing on
    failure.
    """
    if source_path in copied:
        return True
    if not source_path or not os.path.isfile(source_path):
        missing.append(source_path or "<unresolved>")
        return False
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)
    copied[source_path] = destination_path
    return True


def _copy_side_assets(
    source_mesh: str, destination_mesh: str, copied: dict[str, str], missing: list[str]
) -> None:
    """
    Textures referenced by a .dae, or .mtl + its textures for a .obj.
    """
    source_directory = os.path.dirname(source_mesh)
    destination_directory = os.path.dirname(destination_mesh)
    extension = source_mesh.lower().rsplit(".", 1)[-1]
    if not os.path.isfile(source_mesh):
        return
    with open(source_mesh, "rb") as mesh_file:
        text = mesh_file.read().decode("utf-8", "replace")
    references = set()
    if extension == "dae":
        references |= set(re.findall(r"[\w./\-]+\.(?:png|jpg|jpeg|tga|tif)", text, re.I))
    elif extension == "obj":
        for match in re.findall(r"mtllib\s+(.+)", text):
            references.add(match.strip())
        for mtl_name in list(references):
            mtl_source_path = os.path.join(source_directory, mtl_name)
            if os.path.isfile(mtl_source_path):
                _copy_file(
                    mtl_source_path,
                    os.path.join(destination_directory, mtl_name),
                    copied,
                    missing,
                )
                with open(mtl_source_path, "rb") as mtl_file:
                    mtl_text = mtl_file.read().decode("utf-8", "replace")
                for texture_match in re.findall(r"map_\w+\s+(.+)", mtl_text):
                    references.add(texture_match.strip())
    for reference in references:
        reference = reference.strip().lstrip("./")
        reference_source_path = os.path.join(source_directory, reference)
        if os.path.isfile(reference_source_path):
            _copy_file(
                reference_source_path,
                os.path.join(destination_directory, reference),
                copied,
                missing,
            )


# %% xacro
def xacro_to_urdf_text(path: str) -> str:
    """
    Run the xacro CLI (needs a sourced ROS environment on PATH).
    """
    result = subprocess.run(["xacro", path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("xacro failed for %s:\n%s" % (path, result.stderr[-2000:]))
    return result.stdout


# %% main
@dataclass
class BundleReport:
    """
    The result of bundling one URDF/xacro for the web viewer.
    """

    name: str
    """
    The model's output name (the ``<name>`` in ``<name>.urdf``).
    """

    urdf: str
    """
    Path to the written, self-contained URDF file.
    """

    source: str
    """
    The resolved absolute path of the original URDF/xacro source.
    """

    links: list[str] = field(default_factory=list)
    """
    Names of every ``<link>`` found in the bundled URDF.
    """

    joints: list[str] = field(default_factory=list)
    """
    Names of every ``<joint>`` found in the bundled URDF.
    """

    movable_joints: list[str] = field(default_factory=list)
    """
    Names of the joints whose type is not ``fixed``.
    """

    meshes_copied: int = 0
    """
    How many distinct mesh/side-asset files were copied.
    """

    mesh_exts: list[str] = field(default_factory=list)
    """
    The distinct file extensions among the copied meshes.
    """

    refs_rewritten: int = 0
    """
    How many ``filename="..."`` references were rewritten in the URDF.
    """

    missing: list[str] = field(default_factory=list)
    """
    Source paths that could not be resolved/copied.
    """

def bundle_urdf(
    source: str, name: str, out_dir: str, hints: dict[str, str] | None = None
) -> BundleReport:
    """
    Bundle one URDF/xacro.
    """
    source_path = resolve_uri(source, hints=hints) or source
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            "URDF source not found: %s (from %s)" % (source_path, source)
        )
    if source_path.endswith(".xacro"):
        text = xacro_to_urdf_text(source_path)
    else:
        with open(source_path, encoding="utf-8", errors="replace") as source_file:
            text = source_file.read()
    base_dir = os.path.dirname(source_path)

    os.makedirs(out_dir, exist_ok=True)
    copied: dict[str, str] = {}
    missing: list[str] = []
    refs_rewritten = 0
    for reference in sorted(set(re.findall(r'filename="([^"]+)"', text))):
        if not reference.lower().endswith(MESH_EXTS):
            continue  # plugins (.so) etc.
        resolved_reference = resolve_uri(reference, hints=hints, base_dir=base_dir)
        relative_path = _ref_to_relpath(reference, resolved_reference)
        destination_path = os.path.join(out_dir, "meshes", relative_path)
        if _copy_file(resolved_reference, destination_path, copied, missing):
            _copy_side_assets(resolved_reference, destination_path, copied, missing)
        text = text.replace(
            '"%s"' % reference, '"meshes/%s"' % relative_path.replace(os.sep, "/")
        )
        refs_rewritten += 1

    urdf_out = os.path.join(out_dir, "%s.urdf" % name)
    with open(urdf_out, "w", encoding="utf-8") as urdf_file:
        urdf_file.write(text)
    links = re.findall(r'<link\s+name="([^"]+)"', text)
    joints = re.findall(r'<joint\s+name="([^"]+)"\s+type="([^"]+)"', text)
    extensions = sorted({os.path.splitext(path)[1].lower() for path in copied})
    return BundleReport(
        name=name,
        urdf=urdf_out,
        source=source_path,
        links=links,
        joints=[joint_name for joint_name, _ in joints],
        movable_joints=[joint_name for joint_name, joint_type in joints if joint_type != "fixed"],
        meshes_copied=len(copied),
        mesh_exts=extensions,
        refs_rewritten=refs_rewritten,
        missing=missing,
    )


def main() -> None:
    """
    Command-line entry point for standalone bundling.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argument_parser.add_argument("source", help="URDF/xacro path or package:// URI")
    argument_parser.add_argument("--name", help="output model name (default: source basename)")
    argument_parser.add_argument("--out", default="static/sim", help="output directory")
    arguments = argument_parser.parse_args()
    name = arguments.name or os.path.splitext(os.path.basename(arguments.source))[0]
    report = bundle_urdf(arguments.source, name, arguments.out)
    logging.info(
        "wrote %s  (%d links, %d joints, %d meshes %s)"
        % (
            report.urdf,
            len(report.links),
            len(report.joints),
            report.meshes_copied,
            report.mesh_exts,
        )
    )
    if report.missing:
        logging.warning("missing %d assets:" % len(report.missing))
        for missing_path in report.missing[:20]:
            logging.warning("   %s", missing_path)
        sys.exit(2)


if __name__ == "__main__":
    main()
