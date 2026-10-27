"""
Protocol documents stored as JSON files.
"""

from __future__ import annotations

import pytest

from cramera_bridge.documents import DocumentFile
from cramera_protocol.bundle.index import SceneIndex, SceneIndexEntry
from cramera_protocol.exceptions import ProtocolError

# %% files


def test_a_written_document_reads_back_without_leftovers(tmp_path):
    index = SceneIndex(scenes=[SceneIndexEntry(name="run", robot="pr2")])
    file = DocumentFile(tmp_path / "scenes" / "index.json")
    file.write(index)
    assert file.read(SceneIndex) == index
    assert [path.name for path in file.path.parent.iterdir()] == ["index.json"]


def test_a_missing_file_reads_as_nothing(tmp_path):
    assert DocumentFile(tmp_path / "index.json").read_if_present(SceneIndex) is None


def test_a_file_of_another_document_is_refused(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ProtocolError):
        DocumentFile(path).read(SceneIndex)
