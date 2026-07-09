"""The consume loop (hermes-consumer spec, tasks 4.x + 5.1).

Per job:  pull → parse TagJob → read sample → decode → tag → publish → ack.

Correctness rules the spec demands, all enforced here:

* **Backpressure** — pull tiny batches (``pull_max`` = 1–3) so a serial run fits
  inside the broker-global ack deadline (default 30s).
* **Ack after publish** — a crash before publish → Hermes redelivers → safe
  (tagging is idempotent per ``postId``; Artemis dedups).
* **Failure branch by cause** —
  permanent (undecodable sample) → publish ``failed`` + ack;
  transient (Apollo/publish/Hermes fault) → nack (``modifyAckDeadline → 0``).
* **Lease extension** — long inference can extend its lease (wired via
  ``modify_ack_deadline``); poison jobs are bounded by Hermes's DLQ.
"""

from __future__ import annotations

import logging

from argus.apollo import SampleReader, decode_image
from argus.contract import TagJob, TagSuggestions
from argus.errors import PermanentFailure, TransientFailure
from argus.hermes import HermesClient, PulledMessage
from argus.pipeline import TaggingPipeline

logger = logging.getLogger("argus.consumer")


class Consumer:
    """Pull-based, lease-aware consumer of the ``media.tag`` queue."""

    def __init__(
        self,
        *,
        hermes: HermesClient,
        reader: SampleReader,
        pipeline: TaggingPipeline,
        subscription_id: str,
        source_topic: str,
        results_topic: str,
        pull_max: int = 2,
    ) -> None:
        self._hermes = hermes
        self._reader = reader
        self._pipeline = pipeline
        self._subscription_id = subscription_id
        self._source_topic = source_topic
        self._results_topic = results_topic
        self._pull_max = pull_max

    def ensure_subscription(self) -> None:
        self._hermes.ensure_subscription(self._subscription_id, self._source_topic)

    def run_once(self) -> int:
        """Pull one small batch and process it. Returns messages handled.

        Isolated from the forever-loop so it is directly testable.
        """
        messages = self._hermes.pull(self._subscription_id, self._pull_max)
        for message in messages:
            self._handle(message)
        return len(messages)

    def run_forever(self, *, poll_idle_seconds: float = 1.0) -> None:  # pragma: no cover - loop
        import time

        self.ensure_subscription()
        logger.info("consuming %s (pull_max=%d)", self._subscription_id, self._pull_max)
        while True:
            try:
                handled = self.run_once()
            except TransientFailure:
                logger.exception("transient pull failure; backing off")
                handled = 0
            if handled == 0:
                time.sleep(poll_idle_seconds)

    def _handle(self, message: PulledMessage) -> None:
        """Process one leased message, applying the failure branch."""
        try:
            job = TagJob.from_payload(message.payload)
        except (ValueError, KeyError) as exc:
            # Malformed envelope: permanent — ack it away rather than loop forever.
            logger.warning("dropping malformed TagJob (ack): %s", exc)
            self._hermes.ack(self._subscription_id, [message.ack_id])
            return

        # Step 1: tag. Permanent → publish a `failed` result; transient → nack, done.
        try:
            result = self._tag(job)
        except PermanentFailure as exc:
            logger.info("permanent failure for %s: %s → publish failed + ack", job.post_id, exc)
            result = TagSuggestions.failed(job.post_id)
        except TransientFailure as exc:
            logger.warning("transient failure for %s: %s → nack for redelivery", job.post_id, exc)
            self._nack(message.ack_id)
            return

        # Step 2: publish then ack — ACK ONLY after publish is durable. A transient
        # publish/ack fault must NOT ack (the job has to come back), so nack instead.
        try:
            self._hermes.publish(self._results_topic, result.to_payload(), result.attributes())
            self._hermes.ack(self._subscription_id, [message.ack_id])
        except TransientFailure as exc:
            logger.warning("publish/ack failed for %s: %s → nack", job.post_id, exc)
            self._nack(message.ack_id)

    def _tag(self, job: TagJob) -> TagSuggestions:
        data = self._reader.read(job.sample)  # TransientFailure on Apollo faults
        image = decode_image(data)  # PermanentFailure on undecodable bytes
        return TagSuggestions(post_id=job.post_id, suggestions=self._pipeline.tag(image))

    def _nack(self, ack_id: str) -> None:
        """Force fast redelivery by zeroing the lease (Hermes DLQs after N attempts)."""
        try:
            self._hermes.modify_ack_deadline(self._subscription_id, [ack_id], 0)
        except TransientFailure:
            # Even the nack failed; leave it — the lease will lapse and redeliver.
            logger.exception("nack failed; relying on lease expiry")
