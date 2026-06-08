"""Shared inflect engine for the verbalization subsystem.

A single :data:`_engine` instance is used by every module in this package
that needs English morphology (plural forms, articles, ordinals).
"""

from __future__ import annotations

import inflect

_engine = inflect.engine()
