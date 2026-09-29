"""Select the authored laboratory served by a dedicated backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


# %% explicit backend configuration
class LaboratoryEnvironmentVariable(StrEnum):
    """Deployment settings for the isolated laboratory simulation."""

    BUNDLE = "CRAMERA_LABORATORY_BUNDLE"
    """Authored room directory replacing the default source bundle."""


@dataclass(frozen=True)
class LaboratoryConfiguration:
    """An optional room selection shared by robot and contact sessions."""

    source_directory: Path | None = None
    """The explicitly configured authored world, or the default laboratory."""

    @classmethod
    def from_environment(cls) -> LaboratoryConfiguration:
        """Resolve the configured resource path without modifying scene catalogs."""
        value = os.environ.get(LaboratoryEnvironmentVariable.BUNDLE)
        return cls(Path(value).expanduser().resolve() if value else None)
