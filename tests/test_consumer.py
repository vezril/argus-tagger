"""Consumer behavior — the non-wedging failure branch and ack-after-publish.

Uses a recording fake Hermes plus stub reader/taggers, so the whole consume →
tag → publish → ack path runs with no network or ML stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from argus.apollo import StubSampleReader
from argus.consumer import Consumer
from argus.contract import Source
from argus.errors import TransientFailure
from argus.hermes import PulledMessage
from argus.models.base import Prediction, StubTagger
from argus.pipeline import TaggingPipeline


@dataclass
class FakeHermes:
    """Records calls; hands out queued pull batches."""

    to_pull: list[list[PulledMessage]] = field(default_factory=list)
    published: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    acked: list[str] = field(default_factory=list)
    nacked: list[str] = field(default_factory=list)
    subscriptions: list[tuple[str, str]] = field(default_factory=list)
    publish_fail: bool = False

    def ensure_subscription(self, subscription_id: str, topic_id: str) -> None:
        self.subscriptions.append((subscription_id, topic_id))

    def pull(self, subscription_id: str, max_messages: int) -> list[PulledMessage]:
        return self.to_pull.pop(0) if self.to_pull else []

    def publish(self, topic_id: str, payload: str, attributes: dict[str, str]) -> str:
        if self.publish_fail:
            raise TransientFailure("publish down")
        self.published.append((topic_id, payload, attributes))
        return "m-1"

    def ack(self, subscription_id: str, ack_ids: list[str]) -> None:
        self.acked.extend(ack_ids)

    def modify_ack_deadline(self, subscription_id: str, ack_ids: list[str], seconds: int) -> None:
        if seconds == 0:
            self.nacked.extend(ack_ids)


def _job_message(ack_id: str = "ack-1", post_id: str = "p1") -> PulledMessage:
    payload = json.dumps(
        {
            "postId": post_id,
            "sample": {"bucket": "samples", "object": f"{post_id}/s.png"},
            "mediaType": "image/png",
        }
    )
    return PulledMessage(
        ack_id=ack_id, payload=payload, attributes={"postId": post_id}, publish_time=""
    )


def _pipeline() -> TaggingPipeline:
    p = TaggingPipeline(
        [
            StubTagger(Source.WD, [Prediction(tag="1girl", confidence=0.9, category="general")]),
            StubTagger(Source.RAM, [Prediction(tag="beach", confidence=0.7)]),
        ]
    )
    p.load()
    return p


def _consumer(hermes: FakeHermes, reader: StubSampleReader) -> Consumer:
    return Consumer(
        hermes=hermes,  # type: ignore[arg-type]
        reader=reader,
        pipeline=_pipeline(),
        subscription_id="argus.media.tag",
        source_topic="media.tag",
        results_topic="media.suggestions",
    )


def test_happy_path_publishes_then_acks() -> None:
    hermes = FakeHermes(to_pull=[[_job_message()]])
    consumer = _consumer(hermes, StubSampleReader())

    handled = consumer.run_once()

    assert handled == 1
    assert len(hermes.published) == 1
    topic, payload, attrs = hermes.published[0]
    assert topic == "media.suggestions"
    body = json.loads(payload)
    assert body["postId"] == "p1"
    assert {s["tag"] for s in body["suggestions"]} == {"1girl", "beach"}
    assert attrs == {"postId": "p1"}
    assert hermes.acked == ["ack-1"]  # acked AFTER publish
    assert hermes.nacked == []


def test_undecodable_sample_publishes_failed_and_acks() -> None:
    # Permanent failure: retrying can't help → publish failed + ack (no wedge).
    hermes = FakeHermes(to_pull=[[_job_message()]])
    consumer = _consumer(hermes, StubSampleReader(data=b"not-an-image"))

    consumer.run_once()

    assert len(hermes.published) == 1
    assert json.loads(hermes.published[0][1])["status"] == "failed"
    assert hermes.acked == ["ack-1"]
    assert hermes.nacked == []


def test_transient_reader_failure_nacks_for_redelivery() -> None:
    hermes = FakeHermes(to_pull=[[_job_message()]])
    reader = StubSampleReader(fail=TransientFailure("apollo down"))
    consumer = _consumer(hermes, reader)

    consumer.run_once()

    assert hermes.published == []
    assert hermes.acked == []          # NOT acked — it must come back
    assert hermes.nacked == ["ack-1"]  # nacked (deadline → 0)


def test_publish_failure_does_not_ack() -> None:
    # If publish fails, the job must not be acked (ack-after-publish invariant).
    hermes = FakeHermes(to_pull=[[_job_message()]], publish_fail=True)
    consumer = _consumer(hermes, StubSampleReader())

    consumer.run_once()

    assert hermes.acked == []
    assert hermes.nacked == ["ack-1"]


def test_malformed_envelope_is_acked_away() -> None:
    bad = PulledMessage(ack_id="ack-9", payload="{not json", attributes={}, publish_time="")
    hermes = FakeHermes(to_pull=[[bad]])
    consumer = _consumer(hermes, StubSampleReader())

    consumer.run_once()

    assert hermes.published == []
    assert hermes.acked == ["ack-9"]  # don't loop forever on garbage


def test_empty_pull_handled_cleanly() -> None:
    hermes = FakeHermes(to_pull=[[]])
    consumer = _consumer(hermes, StubSampleReader())
    assert consumer.run_once() == 0
