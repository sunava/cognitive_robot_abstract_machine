"""
Reading a query back as English, in plain text and as HTML whose words are coloured by
the role they play and linked to the documentation of the classes they name.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from enum import StrEnum

from krrood.entity_query_language.verbalization.exceptions import (
    UnverbalizableExpressionError,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import (
    ParagraphRenderer,
)
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
    AutoAPIResolver,
)
from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
from typing_extensions import Any, Optional

from cramera_protocol.questions import Verbalization

DOCUMENTATION_SITE = "https://cram2.github.io/cognitive_robot_abstract_machine"
"""
The published documentation of the CRAM packages, one AutoAPI build per package.
"""

# %% links


class DocumentedPackage(StrEnum):
    """
    The CRAM packages whose documentation is published.
    """

    CORAPLEX = "coraplex"
    """
    Plans and actions.
    """

    GISKARDPY = "giskardpy"
    """
    Motion statecharts.
    """

    KRROOD = "krrood"
    """
    The entity query language.
    """

    PROBABILISTIC_MODEL = "probabilistic_model"
    """
    Probabilistic models.
    """

    RANDOM_EVENTS = "random_events"
    """
    Random events.
    """

    SEMANTIC_DIGITAL_TWIN = "semantic_digital_twin"
    """
    Worlds and their annotations.
    """


@dataclass(frozen=True)
class PublishedDocumentation:
    """
    Links a word naming a class to the class's published documentation page.
    """

    site: str = DOCUMENTATION_SITE
    """
    The root of the documentation site.
    """

    def resolve(self, reference: SourceReference) -> Optional[str]:
        """
        :param reference: The class or attribute a word names.
        :return: The page documenting it, or None when its package publishes none.
        """
        if not isinstance(reference.owner_type, type):
            return None
        package = reference.owner_type.__module__.split(".", 1)[0]
        if package not in {member.value for member in DocumentedPackage}:
            return None
        return AutoAPIResolver(base_url=f"{self.site.rstrip('/')}/{package}").resolve(
            reference
        )


# %% rendering


@dataclass
class EscapedHtmlFormatter(HTMLFormatter):
    """
    Krrood's coloured HTML with the words escaped, since a query's literals are whatever
    a person typed.
    """

    def colorize(self, text: str, role: SemanticRole) -> str:
        return super().colorize(html.escape(text), role)


@dataclass(frozen=True)
class QueryVerbalizer:
    """
    Reads queries back as English without evaluating them.
    """

    documentation: PublishedDocumentation = field(
        default_factory=PublishedDocumentation
    )
    """
    Where the words of the HTML rendering link to.
    """

    def verbalize(self, expression: Any) -> Optional[Verbalization]:
        """
        :param expression: A query expression.
        :return: The query in plain text and as HTML, or None when krrood has no
            wording for it; wording leaves the expression evaluable.
        """
        try:
            fragment = EQLVerbalizer().build(expression)
        except UnverbalizableExpressionError:
            # krrood offers no way to ask whether an expression can be worded other
            # than wording it; a query without a wording is still answered.
            return None
        return Verbalization(
            text=ParagraphRenderer().render(fragment),
            html=ParagraphRenderer(
                EscapedHtmlFormatter(), link_resolver=self.documentation
            ).render(fragment),
        )
