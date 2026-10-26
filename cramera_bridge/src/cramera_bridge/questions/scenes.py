"""
Questions about the recorded scenes of a viewer's catalog, answered as the protocol's
documents: what the questions plugin's routes ask of the question engine.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from typing_extensions import Dict, Optional

from cramera.scenes import SceneCatalog
from cramera_bridge.questions.desk import QuestionDesk
from cramera_bridge.questions.graph.views import GraphViews
from cramera_bridge.questions.recorded.bundle import BundleReader, RecordedBundle
from cramera_bridge.questions.recorded.knowledge_base import RecordedKnowledge
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex
from cramera_protocol.questions import (
    GraphView,
    Members,
    PresetCatalog,
    QueryAnswer,
    QueryRequest,
    QueryScope,
    QuestionMatch,
    QuestionRequest,
    Vocabulary,
)

# %% scenes


@dataclass(frozen=True)
class SceneQuestions:
    """
    What one recorded scene answers with, built from the scene's files as they were.
    """

    stamp: int
    """
    When the scene description last changed when this was built.
    """

    knowledge: RecordedKnowledge
    """
    The scene's entities.
    """

    desk: QuestionDesk
    """
    The desk answering questions about the scene.
    """


@dataclass
class SceneQuestionCache:
    """
    The questions of each scene, rebuilt when its scene description changes.
    """

    catalog: SceneCatalog
    """
    The scenes the viewer shows.
    """

    class_index: WorkspaceClassIndex
    """
    The workspace classes a query may name.
    """

    built: Dict[str, SceneQuestions] = field(default_factory=dict)
    """
    The questions of each scene asked about so far.
    """

    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    """
    Makes checking, building and storing one step for concurrent requests.
    """

    def of_scene(self, name: str) -> SceneQuestions:
        """
        :param name: A bundle name.
        :return: The scene's questions, rebuilt when its description changed.
        """
        stamp = BundleReader(self.catalog, name).stamp()
        with self.lock:
            cached = self.built.get(name)
            if cached is not None and cached.stamp == stamp:
                return cached
            knowledge = RecordedKnowledge.of_bundle(
                RecordedBundle.of_catalog(self.catalog, name)
            )
            built = SceneQuestions(
                stamp=stamp, knowledge=knowledge, desk=knowledge.desk(self.class_index)
            )
            self.built[name] = built
            return built


# %% catalog


@dataclass(frozen=True)
class RecordedSceneQuestions:
    """
    Answers questions about the scenes of one catalog.
    """

    cache: SceneQuestionCache
    """
    The questions of each scene.
    """

    graph_views: GraphViews = field(default_factory=GraphViews)
    """
    Builds the graph views.
    """

    @classmethod
    def of_catalog(
        cls, catalog: SceneCatalog, class_index: Optional[WorkspaceClassIndex] = None
    ) -> RecordedSceneQuestions:
        """
        :param catalog: The scenes the viewer shows.
        :param class_index: The workspace classes a query may name; None reads those of
            the installed workspace packages.
        :return: The questions about the catalog's scenes.
        """
        if class_index is None:
            class_index = WorkspaceClassIndex.of_installed_packages()
        return cls(SceneQuestionCache(catalog, class_index))

    def preset_catalog(self, scene: str) -> PresetCatalog:
        """
        :param scene: A bundle name.
        :return: The scene's ready-made questions.
        """
        return self.cache.of_scene(scene).desk.preset_catalog()

    def vocabulary(self, scene: str, scope: QueryScope) -> Vocabulary:
        """
        :param scene: A bundle name.
        :param scope: A body of knowledge.
        :return: The names a query of it may use.
        """
        return self.cache.of_scene(scene).desk.vocabulary(scope)

    def members(self, scene: str, scope: QueryScope, name: str) -> Members:
        """
        :param scene: A bundle name.
        :param scope: A body of knowledge.
        :param name: A type's name or a variable's.
        :return: The members that may follow the name's dot.
        """
        return self.cache.of_scene(scene).desk.members(scope, name)

    def graph_view(self, scene: str, view: str) -> GraphView:
        """
        :param scene: A bundle name.
        :param view: A view name, a :class:`GraphViewName` value.
        :return: The graph view.
        """
        return self.graph_views.build(view, self.cache.of_scene(scene).knowledge)

    def answer_query(self, scene: str, request: QueryRequest) -> QueryAnswer:
        """
        :param scene: The bundle the query is about.
        :param request: A query.
        :return: Its answer.
        """
        return self.cache.of_scene(scene).desk.answer_query(request)

    def answer_question(self, scene: str, request: QuestionRequest) -> QuestionMatch:
        """
        :param scene: The bundle the question is about.
        :param request: A plain-language question.
        :return: The ready-made question it asks, or the reply that none does.
        """
        return self.cache.of_scene(scene).desk.answer_question(request)
