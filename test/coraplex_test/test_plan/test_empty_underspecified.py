"""
What a plan says when an action it left open could not be pinned down to anything that
works.
"""

import pytest

from coraplex.plans.failures import EmptyUnderspecified, PlanFailure
from coraplex.robot_plans.actions.core.navigation import NavigateAction


class TestSayingWhatCouldNotBeGrounded:
    """
    An underspecified action is tried candidate by candidate; when they run out there is
    nothing left to point at unless the failure carries it.
    """

    def test_the_action_that_could_not_be_grounded_is_named(self):
        failure = EmptyUnderspecified(action=NavigateAction)
        assert NavigateAction.__name__ in str(failure)

    def test_the_reason_the_last_candidate_failed_is_carried(self):
        last = PlanFailure()
        failure = EmptyUnderspecified(action=NavigateAction, last_failure=last)
        assert failure.last_failure is last
        assert last.error_message() in str(failure)

    def test_an_action_nothing_matched_is_told_apart_from_one_that_failed(self):
        nothing_matched = EmptyUnderspecified(action=NavigateAction)
        every_candidate_failed = EmptyUnderspecified(
            action=NavigateAction, last_failure=PlanFailure()
        )
        assert str(nothing_matched) != str(every_candidate_failed)

    def test_a_failure_naming_nothing_still_reads(self):
        assert str(EmptyUnderspecified())

    def test_it_is_still_a_plan_failure(self):
        with pytest.raises(PlanFailure):
            raise EmptyUnderspecified(action=NavigateAction)
