"""
Segmind's detectors ticked along a run.
"""

from __future__ import annotations

from cramera_bridge.attachment import AttachmentDetector
from segmind.datastructures.events import SupportEvent, TranslationEvent
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.statecharts.segmind_statechart import SegmindStatechart

from cramera_bridge.detections import (
    EVENT_NAME_SEPARATOR,
    DetectionRecorder,
    DetectionTicks,
)

from ..dataset.onboarding.sliding_box_demo import BOX_HEIGHTS

# %% detectors


def detector_types() -> set:
    """
    :return: The type of every detector a recorder ticks.
    """
    return {type(detector) for detector in DetectionRecorder.detectors()}


def test_picking_up_is_read_from_the_attachment():
    assert AttachmentDetector in detector_types()


def test_picking_up_is_not_also_inferred_from_motion():
    assert not {PickUpDetector, PlacingDetector} & detector_types()


def test_every_other_detector_segmind_offers_is_ticked():
    offered = {
        type(detector) for detector in SegmindStatechart().build_statechart().nodes
    }
    assert offered - {PickUpDetector, PlacingDetector} <= detector_types()


# %% recording


def lower_box(sliding_box) -> None:
    """
    Move the box down onto the table.
    """
    for height in BOX_HEIGHTS:
        sliding_box.move_box(height)


def state_callbacks(world) -> list:
    """
    :return: The detection callbacks registered on the world.
    """
    return [
        callback
        for callback in world.state.state_change_callbacks
        if isinstance(callback, DetectionTicks)
    ]


def test_a_recorder_that_never_started_detects_nothing(sliding_box):
    recorder = DetectionRecorder(sliding_box.world)
    lower_box(sliding_box)
    assert recorder.events() == []


def test_a_started_recorder_detects_what_the_world_does(sliding_box):
    recorder = DetectionRecorder(sliding_box.world).start()
    lower_box(sliding_box)
    recorder.stop()
    event_types = {event.event_type for event in recorder.events()}
    assert {SupportEvent.__name__, TranslationEvent.__name__} <= event_types


def test_an_event_names_the_bodies_it_is_about(sliding_box):
    recorder = DetectionRecorder(sliding_box.world).start()
    lower_box(sliding_box)
    recorder.stop()
    [support] = [
        event
        for event in recorder.events()
        if event.event_type == SupportEvent.__name__
    ]
    assert support.tracked_object == str(sliding_box.box.name)
    assert support.with_object == str(sliding_box.table.name)
    assert support.name == EVENT_NAME_SEPARATOR.join(
        (str(sliding_box.box.name), SupportEvent.__name__)
    )


def test_an_event_without_a_second_body_names_none(sliding_box):
    recorder = DetectionRecorder(sliding_box.world).start()
    lower_box(sliding_box)
    recorder.stop()
    [translation, *_] = [
        event
        for event in recorder.events()
        if event.event_type == TranslationEvent.__name__
    ]
    assert translation.with_object is None


def test_a_stopped_recorder_keeps_its_events_and_detects_no_more(sliding_box):
    recorder = DetectionRecorder(sliding_box.world).start()
    lower_box(sliding_box)
    recorder.stop()
    detected = recorder.events()
    sliding_box.move_box(BOX_HEIGHTS[0])
    lower_box(sliding_box)
    assert recorder.events() == detected
    assert state_callbacks(sliding_box.world) == []
