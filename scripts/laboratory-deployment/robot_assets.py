"""
List a robot description's mesh files relative to its deployment checkout.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


# %% collision assets
@dataclass(frozen=True)
class RobotAssets:
    """
    A robot description and the checkout boundary its mesh references must obey.
    """

    repository: Path
    """
    Checkout root used by rsync's file list.
    """

    description: Path
    """
    Source URDF containing collision and visual mesh references.
    """

    def paths(self) -> list[Path]:
        """
        Resolve existing mesh references without admitting external paths.
        """
        root = self.repository.resolve()
        description = self.description.resolve()
        meshes = {
            (description.parent / node.attrib["filename"]).resolve()
            for node in ElementTree.parse(description).getroot().iter("mesh")
        }
        for path in meshes:
            if not path.is_file():
                raise FileNotFoundError(path)
        return [path.relative_to(root) for path in sorted(meshes)]


def main() -> None:
    """
    Print only repository-relative paths for the deployment file manifest.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("description", type=Path)
    arguments = parser.parse_args()
    for path in RobotAssets(arguments.repository, arguments.description).paths():
        print(path.as_posix())


if __name__ == "__main__":
    main()
