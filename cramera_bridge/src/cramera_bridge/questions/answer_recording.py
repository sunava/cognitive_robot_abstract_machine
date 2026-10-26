"""
``cramera-record-answers``: answers the ready-made questions of recorded bundles and
builds their graph views with krrood, and writes them into each bundle as
``recorded_answers.json``, so viewers without krrood, such as the static demo site, can
show them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from typing_extensions import Callable, List, Optional, Sequence

from cramera.scenes import SceneCatalog, SceneDirectory, SceneSource
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.questions import (
    GraphViewName,
    QueryRequest,
    RecordedAnswer,
    RecordedAnswers,
)
from cramera_protocol.version import CURRENT_VERSION
from cramera_bridge.questions.graph.views import GraphViews
from cramera_bridge.questions.recorded.bundle import RecordedBundle
from cramera_bridge.questions.recorded.knowledge_base import RecordedKnowledge
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex

Clock = Callable[[], datetime]
"""
Tells the current time in UTC.
"""

# %% recording


@dataclass
class AnswerRecorder:
    """
    Records the answers of the bundles of one scenes directory.
    """

    scenes: Path
    """
    The scenes directory holding the bundles.
    """

    class_index: WorkspaceClassIndex = field(
        default_factory=WorkspaceClassIndex.of_installed_packages
    )
    """
    The workspace classes a query may name.
    """

    graph_views: GraphViews = field(default_factory=GraphViews)
    """
    Builds the graph views.
    """

    clock: Clock = field(default=lambda: datetime.now(timezone.utc))
    """
    Tells when the answers are recorded.
    """

    def catalog(self) -> SceneCatalog:
        """
        :return: The catalog of the scenes directory.
        """
        return SceneCatalog([SceneDirectory(self.scenes, SceneSource.DEMO)])

    def scene_names(self) -> List[str]:
        """
        :return: The names of every readable bundle of the directory.
        """
        return [entry.name for entry in self.catalog().index().scenes]

    def record(self, name: str) -> RecordedAnswers:
        """
        :param name: A bundle of the directory.
        :return: Its ready-made questions with their answers, and its graph views.
        """
        knowledge = RecordedKnowledge.of_bundle(
            RecordedBundle.of_catalog(self.catalog(), name)
        )
        desk = knowledge.desk(self.class_index)
        presets = desk.preset_catalog()
        answers = [
            RecordedAnswer(
                code=preset.code,
                scope=preset.scope,
                answer=desk.answer_query(
                    QueryRequest(code=preset.code, scope=preset.scope, scene=name)
                ),
            )
            for preset in presets.presets
            if not preset.requires_live
        ]
        return RecordedAnswers(
            format_version=CURRENT_VERSION,
            scene=name,
            recorded_at=self.clock()
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            presets=presets,
            answers=answers,
            graphs=[
                self.graph_views.build(view.value, knowledge) for view in GraphViewName
            ],
        )

    def write(self, name: str) -> Path:
        """
        Record a bundle's answers into the bundle.

        :param name: A bundle of the directory.
        :return: The file written.
        """
        path = self.scenes / name / BundleFile.RECORDED_ANSWERS
        path.write_text(
            json.dumps(self.record(name).to_json(), indent=1) + "\n", encoding="utf-8"
        )
        return path


# %% command


class RecordOption(StrEnum):
    """
    The command's options.
    """

    SCENE = "--scene"
    """
    A bundle to record; every bundle of the directory when none is named.
    """


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """
    Record the answers of a scenes directory's bundles.

    :param arguments: The command-line arguments; the process arguments when None.
    :return: The process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="cramera-record-answers", description=__doc__.strip().splitlines()[0]
    )
    parser.add_argument("scenes", type=Path)
    parser.add_argument(RecordOption.SCENE, action="append", default=None)
    parsed = parser.parse_args(arguments)
    recorder = AnswerRecorder(parsed.scenes)
    for name in parsed.scene or recorder.scene_names():
        print(f"recorded {recorder.write(name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
