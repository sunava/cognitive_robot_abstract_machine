from __future__ import annotations

from typing import Protocol

from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.rendering.colorizer import Colorizer, PlainColorizer


class FragmentRenderer(Protocol):
    """Converts a VerbFragment tree into a string."""

    def render(self, fragment: VerbFragment) -> str: ...


class ParagraphRenderer:
    """
    Flattens the fragment tree into a single prose string.

    BlockFragment headers and items are joined inline; nesting adds no
    visual structure — only content.
    """

    def __init__(self, colorizer: Colorizer = PlainColorizer()):
        self._colorizer = colorizer

    def render(self, fragment: VerbFragment) -> str:
        match fragment:
            case WordFragment(text=text):
                return text
            case RoleFragment(text=text, role=role):
                return self._colorizer.colorize(text, role)
            case PhraseFragment(parts=parts, separator=sep):
                rendered = [self.render(p) for p in parts]
                return sep.join(rendered)
            case BlockFragment(header=header, items=items):
                rendered_items = [self.render(i) for i in items]
                prose = ", ".join(rendered_items)
                if header is None:
                    return prose
                header_str = self.render(header)
                return f"{header_str} {prose}" if prose else header_str
            case _:
                return ""


class HierarchicalRenderer:
    """
    Renders BlockFragments as indented bullet lists.

    Each level of BlockFragment nesting adds one ``indent`` step.
    Non-block fragments are rendered inline using the same colorizer.

    Example output::

        **If:**
          - there's a Handle
          - there's a PrismaticConnection, whose child is …
        **Then:**
          - there's a Drawer
            - whose container is …
    """

    def __init__(
        self,
        colorizer: Colorizer = PlainColorizer(),
        indent: str = "  ",
        bullet: str = "- ",
    ):
        self._colorizer = colorizer
        self._indent = indent
        self._bullet = bullet

    def render(self, fragment: VerbFragment, depth: int = 0) -> str:
        match fragment:
            case BlockFragment(header=header, items=items):
                lines: list[str] = []
                if header is not None:
                    header_str = self._inline(header)
                    lines.append(self._indent * depth + header_str)
                for item in items:
                    lines.append(self._render_item(item, depth + 1))
                return "\n".join(lines)
            case _:
                return self._indent * depth + self._inline(fragment)

    def _render_item(self, fragment: VerbFragment, depth: int) -> str:
        """Render one item, prepending the bullet at its indentation level."""
        match fragment:
            case BlockFragment():
                return self.render(fragment, depth)
            case _:
                prefix = self._indent * depth + self._bullet
                return prefix + self._inline(fragment)

    def _inline(self, fragment: VerbFragment) -> str:
        """Render a non-block fragment as a flat inline string."""
        match fragment:
            case WordFragment(text=text):
                return text
            case RoleFragment(text=text, role=role):
                return self._colorizer.colorize(text, role)
            case PhraseFragment(parts=parts, separator=sep):
                return sep.join(self._inline(p) for p in parts)
            case BlockFragment():
                # Nested block encountered while rendering inline — delegate to render()
                return self.render(fragment, 0)
            case _:
                return ""
