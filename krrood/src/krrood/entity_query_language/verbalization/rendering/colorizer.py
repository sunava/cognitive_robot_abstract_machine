from __future__ import annotations

from typing import Protocol

from krrood.entity_query_language.verbalization.fragments.roles import ROLE_COLORS, SemanticRole


class Colorizer(Protocol):
    """Applies visual styling to a text string based on its semantic role."""

    def colorize(self, text: str, role: SemanticRole) -> str: ...


class PlainColorizer:
    """Returns text unchanged — no color markup."""

    def colorize(self, text: str, role: SemanticRole) -> str:
        return text


class ANSIColorizer:
    """
    Wraps text in true-color ANSI escape sequences (24-bit, ``\\033[38;2;R;G;Bm``).

    Works in VS Code terminal, GNOME Terminal, iTerm2, Windows Terminal, and any
    other terminal that supports the ISO-8613-3 direct-color extension.
    """

    _RESET = "\033[0m"

    def colorize(self, text: str, role: SemanticRole) -> str:
        color = ROLE_COLORS.get(role)
        if color is None:
            return text
        r, g, b = self._hex_to_rgb(color)
        return f"\033[38;2;{r};{g};{b}m{text}{self._RESET}"

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        """Convert ``"#rrggbb"`` or a CSS named color to an ``(R, G, B)`` tuple."""
        if color.startswith("#"):
            h = color.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        # Named colours used in ROLE_COLORS (currently only "cornflowerblue")
        _NAMED: dict[str, tuple[int, int, int]] = {
            "cornflowerblue": (100, 149, 237),
        }
        return _NAMED.get(color.lower(), (255, 255, 255))


class MarkdownColorizer:
    """
    Wraps text in an HTML ``<span style="color: …">`` tag.

    The output is valid inside GitHub-flavored Markdown rendered by any renderer
    that passes through inline HTML (Jupyter, GitLab, most static-site generators).
    """

    def colorize(self, text: str, role: SemanticRole) -> str:
        color = ROLE_COLORS.get(role)
        if color is None:
            return text
        return f'<span style="color:{color}">{text}</span>'
