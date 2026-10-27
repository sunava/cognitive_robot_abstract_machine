"""
A clock the tests advance by hand.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ManualClock:
    """
    Answers the time it was last set to.
    """

    now: float = 0.0
    """
    The time it answers, in seconds.
    """

    def __call__(self) -> float:
        """
        :return: The current time.
        """
        return self.now
