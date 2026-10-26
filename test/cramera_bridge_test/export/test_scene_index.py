"""
Registering bundles in a scenes directory's index.
"""

from __future__ import annotations

import shutil

from cramera_bridge.documents import DocumentFile
from cramera_bridge.export.scene_index import BACKUP_SUFFIX, SceneIndexFile
from cramera_protocol.bundle.files import ScenesDirectoryFile
from cramera_protocol.bundle.index import SceneIndex, SceneIndexEntry

from ..dataset.bundles import ANNOTATED_BUNDLE, BROKEN_INDEX

# %% registering


def test_a_new_index_lists_the_bundles_already_there(tmp_path):
    shutil.copytree(ANNOTATED_BUNDLE, tmp_path / ANNOTATED_BUNDLE.name)
    index = SceneIndexFile(tmp_path).register(SceneIndexEntry(name="run", robot="pr2"))
    assert [entry.name for entry in index.scenes] == [ANNOTATED_BUNDLE.name, "run"]
    assert DocumentFile(tmp_path / ScenesDirectoryFile.INDEX).read(SceneIndex) == index


def test_registering_again_replaces_the_entry_and_may_make_it_the_default(tmp_path):
    index_file = SceneIndexFile(tmp_path)
    index_file.register(SceneIndexEntry(name="run", robot="pr2"))
    index = index_file.register(
        SceneIndexEntry(name="run", robot="tiago"), default=True
    )
    assert index.scenes == [SceneIndexEntry(name="run", robot="tiago")]
    assert index.default == "run"


def test_a_broken_index_is_kept_as_a_backup(tmp_path):
    shutil.copy(BROKEN_INDEX, tmp_path / ScenesDirectoryFile.INDEX)
    index = SceneIndexFile(tmp_path).register(SceneIndexEntry(name="run"))
    [backup] = tmp_path.glob(f"{ScenesDirectoryFile.INDEX}{BACKUP_SUFFIX}-*")
    assert backup.read_bytes() == BROKEN_INDEX.read_bytes()
    assert [entry.name for entry in index.scenes] == ["run"]
