"""Explicit preparation and resume requests for the complete sample workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing_extensions import Any


# %% request and ownership vocabulary
class WorkflowField(StrEnum):
    """Workflow fields shared by the request and observation boundaries."""

    PREPARE = "prepare"
    """Whether to prepare fresh demo samples before execution."""
    WORKFLOW = "workflow"
    """Nested measured workflow status."""
    RESERVED = "reserved"
    """Whether the workflow retains ownership of samples and instruments."""


class WorkflowSource(StrEnum):
    """Controller identity used by the complete laboratory demonstration."""

    PROGRAM = "laboratory_workflow"
    """Physical sample processing from dosing through refrigerated storage."""


@dataclass
class InvalidWorkflowCommand(ValueError):
    """A workflow request does not explicitly state its preparation intent."""

    reason: str
    """Explanation returned to the operator."""

    def __str__(self) -> str:
        """Return the rejected request's reason."""
        return self.reason


@dataclass(frozen=True)
class WorkflowCommand:
    """Distinguish a fresh prepared demonstration from a state-preserving resume."""

    prepare: bool
    """True prepares four empty samples; false resumes the existing workflow."""

    @classmethod
    def from_payload(cls, payload: Any) -> WorkflowCommand:
        """Accept only an explicit boolean preparation flag."""
        if not isinstance(payload, dict) or set(payload) != {WorkflowField.PREPARE}:
            raise InvalidWorkflowCommand("Expected an explicit prepare flag")
        prepare = payload[WorkflowField.PREPARE]
        if not isinstance(prepare, bool):
            raise InvalidWorkflowCommand("The prepare flag must be true or false")
        return cls(prepare)
