"""
Registering a written bundle in its scenes directory's ``index.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing_extensions import Optional

from cramera.scenes import SceneDirectory, SceneSource
from cramera_bridge.documents import DocumentFile
from cramera_protocol.bundle.files import ScenesDirectoryFile
from cramera_protocol.bundle.index import SceneIndex, SceneIndexEntry
from cramera_protocol.exceptions import ProtocolError

BACKUP_TIME_FORMAT = "%Y%m%dT%H%M%S%f"
"""
How the time is written into the name of a broken index's backup.
"""

BACKUP_SUFFIX = ".broken"
"""
What a broken index's backup name starts its suffix with.
"""

# %% index


@dataclass(frozen=True)
class SceneIndexFile:
    """
    The ``index.json`` of one scenes directory.
    """

    directory: Path
    """
    The scenes directory.
    """

    source: SceneSource = SceneSource.LOCAL
    """
    Where the directory's bundles come from, for reading them when the index is new.
    """

    @property
    def path(self) -> Path:
        """
        The index file.
        """
        return self.directory / ScenesDirectoryFile.INDEX

    def register(self, entry: SceneIndexEntry, default: bool = False) -> SceneIndex:
        """
        Add a bundle to the index, replacing an entry of the same name.

        Without a readable index the new one lists every readable bundle of the
        directory; an unreadable index is kept as a backup beside it.

        :param entry: The bundle's entry.
        :param default: Whether the bundle becomes the one the viewer opens first.
        :return: The written index.
        """
        index = self.current()
        index.scenes = [scene for scene in index.scenes if scene.name != entry.name]
        index.scenes.append(entry)
        if default:
            index.default = entry.name
        DocumentFile(self.path).write(index)
        return index

    def current(self) -> SceneIndex:
        """
        :return: The index as it is, or the index of the directory's bundles when there
            is none or it cannot be read.
        """
        existing = self.readable()
        if existing is not None:
            return existing
        return SceneDirectory(self.directory, self.source).derived_index().index

    def readable(self) -> Optional[SceneIndex]:
        """
        A broken index of one directory must not stop a bundle from being registered, so
        this is the one place where reading errors become a backup.

        :return: The index, or None when it is missing or broken; a broken one is
            renamed to a backup first.
        """
        if not self.path.is_file():
            return None
        try:
            return DocumentFile(self.path).read(SceneIndex)
        except (ValueError, ProtocolError):
            self.path.rename(self.backup_path())
            return None

    def backup_path(self) -> Path:
        """
        :return: A file name for a backup of the index that no earlier backup uses.
        """
        stamp = datetime.now().strftime(BACKUP_TIME_FORMAT)
        return self.path.with_name(f"{self.path.name}{BACKUP_SUFFIX}-{stamp}")
