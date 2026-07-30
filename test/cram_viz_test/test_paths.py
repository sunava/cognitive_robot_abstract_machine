"""
Tests for the scene-bundle search order (env → submodule → data dir) and the
architecture-root discovery used by the knowledge graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cram_viz import paths

# %% scene and data directories


class TestScenesDir:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """
        CRAM_VIZ_SCENES overrides every other step of the search order.
        """
        monkeypatch.setenv("CRAM_VIZ_SCENES", str(tmp_path / "mine"))
        assert paths.scenes_dir() == tmp_path / "mine"

    def test_initialized_submodule_is_used(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        An initialized cram-scenes submodule (index.json present) is used.
        """
        monkeypatch.delenv("CRAM_VIZ_SCENES", raising=False)
        submodule = tmp_path / "scenes"
        submodule.mkdir()
        (submodule / "index.json").write_text(
            json.dumps({"default": None, "scenes": []})
        )
        monkeypatch.setattr(paths, "SCENES_SUBMODULE", submodule)
        assert paths.scenes_dir() == submodule

    def test_empty_submodule_falls_back_to_data_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        An un-initialized submodule (empty directory) is skipped in favor of the data
        dir.
        """
        # an un-initialized submodule is an empty directory — must be skipped
        monkeypatch.delenv("CRAM_VIZ_SCENES", raising=False)
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "data"))
        empty = tmp_path / "scenes"
        empty.mkdir()
        monkeypatch.setattr(paths, "SCENES_SUBMODULE", empty)
        assert paths.scenes_dir() == tmp_path / "data" / "scenes"

    def test_data_dir_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        CRAM_VIZ_DATA overrides the default ``~/.cram_viz`` data directory.
        """
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "d"))
        assert paths.data_dir() == tmp_path / "d"


# %% architecture root discovery


class TestArchitectureRoot:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """
        CRAM_VIZ_ARCHITECTURE takes priority over any walk-up match.
        """
        monkeypatch.setenv("CRAM_VIZ_ARCHITECTURE", str(tmp_path))
        assert paths.architecture_root() == tmp_path

    def test_walks_up_to_sibling_coraplex_and_krrood(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        The nearest ancestor directory containing coraplex/ and krrood/ is returned.
        """
        monkeypatch.delenv("CRAM_VIZ_ARCHITECTURE", raising=False)
        workspace = tmp_path / "workspace"
        nested = workspace / "cram_viz" / "src" / "cram_viz"
        nested.mkdir(parents=True)
        (workspace / "coraplex").mkdir()
        (workspace / "krrood").mkdir()
        assert paths.architecture_root(start=nested / "paths.py") == workspace

    def test_falls_back_to_home_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        With no override and no matching ancestor, the conventional clone location is
        used.
        """
        monkeypatch.delenv("CRAM_VIZ_ARCHITECTURE", raising=False)
        lonely = tmp_path / "somewhere" / "far" / "away"
        lonely.mkdir(parents=True)
        assert (
            paths.architecture_root(start=lonely / "paths.py")
            == Path.home() / "cognitive_robot_abstract_machine"
        )
