"""
The stretch of a recording worth replaying around one answered moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from cramera_protocol.questions import ReplayWindow

REPLAY_LEAD_SECONDS = 1.0
"""
How long before its moment a replay starts.
"""

REPLAY_TAIL_SECONDS = 1.0
"""
How long after its moment a replay ends.
"""

# %% windows


@dataclass(frozen=True)
class ReplayMoment:
    """
    A moment an answer names, which is too short to watch on its own.
    """

    moment: datetime
    """
    When the answered thing happened.
    """

    def window(self) -> ReplayWindow:
        """
        :return: The stretch from shortly before to shortly after the moment.
        """
        at = self.moment.timestamp()
        return ReplayWindow(
            start=at - REPLAY_LEAD_SECONDS, end=at + REPLAY_TAIL_SECONDS
        )
