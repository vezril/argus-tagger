# hermes-consumer

Argus's own Python client for HermesMQ — consuming the `media.tag` queue *correctly* given
that inference is slow (seconds/image). Hermes ships a Scala client; Argus speaks its REST
API (create subscription, pull/lease, modify-ack-deadline, ack).

> **Verified against Hermes (`PubSubRoutes.scala`, `RedeliveryConfig.scala`).** The consume
> surface Argus assumes exists over REST, endpoint-for-endpoint:
> `POST /v1/subscriptions` (create) · `POST /v1/subscriptions/{id}/pull` `{max}` ·
> `POST /v1/subscriptions/{id}/modifyAckDeadline` `{ackIds, ackDeadlineSeconds}` ·
> `POST /v1/subscriptions/{id}/ack` `{ackIds}` · publish via `POST /v1/topics/{id}/messages`.
> Pulled messages return `{ackId, payload, attributes, publishTime}`.

## ADDED Requirements

### Requirement: Pull-based consumption with backpressure

Argus SHALL ensure (idempotently) a subscription to the `media.tag` topic and consume by
**pulling small batches** — never leasing more messages than it can process before their ack
deadline — so a slow tagger applies natural backpressure and Hermes buffers the rest.

#### Scenario: Small pulls match processing capacity
- **GIVEN** a backlog on `media.tag` and slow CPU inference
- **WHEN** Argus consumes
- **THEN** it pulls a small batch, processes it fully, then pulls again — rather than leasing the whole backlog at once

#### Scenario: Edge case — an empty queue pull returns nothing and retries
- **GIVEN** no pending tag-jobs
- **WHEN** Argus pulls
- **THEN** it receives an empty result and polls again without error

### Requirement: Lease-aware processing for slow inference

The ack deadline in Hermes is a **broker/subscription-global** setting (`ack-deadline`,
default **30s**, env `HERMESMQ_ACK_DEADLINE`) — the REST `pull` body carries only `max`, so a
deadline **cannot** be requested per pull. Argus SHALL therefore keep the deadline sufficient
by (a) pulling **very small batches** (`max` = 1–3) so a serially-processed batch cannot
outlive the shared 30s lease, and (b) calling `modifyAckDeadline` to **extend the lease** when
an image's inference runs long. It SHALL **acknowledge only after** the `TagSuggestions` are
durably published.

#### Scenario: A long inference extends its lease instead of being redelivered
- **GIVEN** a leased tag-job whose inference approaches the (global) ack deadline
- **WHEN** Argus is still processing
- **THEN** it calls `modifyAckDeadline` to extend it, and the message is not redelivered while in flight

#### Scenario: Edge case — a small pull keeps a serial batch inside the shared lease
- **GIVEN** the 30s global ack deadline and ~seconds-per-image serial inference
- **WHEN** Argus pulls
- **THEN** it pulls only 1–3 messages so the last message in the batch is still processed (or has its deadline extended) before the shared lease expires

#### Scenario: Edge case — ack happens only after publishing suggestions
- **GIVEN** a processed image
- **WHEN** Argus is about to acknowledge
- **THEN** it acks only once the `TagSuggestions` message is confirmed published (a crash before that → Hermes redelivers)

### Requirement: Idempotent, non-wedging consumption

Consumption SHALL be idempotent per `postId` (at-least-once redelivery must be safe —
re-tagging a post produces equivalent suggestions Artemis can dedup), and a message that
fails to process SHALL NOT wedge the queue. Argus SHALL branch failure handling by cause:

- **Permanent** failure (undecodable/corrupt sample — retrying can't help): publish a
  `TagSuggestions{status: failed}` **and ack**, so it is not needlessly redelivered.
- **Transient** failure (Apollo unreachable, a model hiccup): **nack** by
  `modifyAckDeadline → 0` (or let the lease lapse) so Hermes redelivers and Argus self-heals.

Poison messages are bounded by Hermes's `RedeliverySweeper`: after
`max-delivery-attempts` (default **5**, env `HERMESMQ_MAX_DELIVERY_ATTEMPTS`) the message is
dead-lettered — republished to the configured `dead-letter-topic` (**dropped if unset**) with
`x-delivery-attempts` / `x-original-message-id` headers — and the queue keeps flowing.

#### Scenario: Redelivery re-tags safely
- **GIVEN** a tag-job that was processed but crashed before ack
- **WHEN** Hermes redelivers it
- **THEN** Argus re-tags and re-publishes suggestions, which Artemis treats idempotently (no duplication)

#### Scenario: A permanent failure is acked, not retried
- **GIVEN** a `TagJob` whose sample cannot be decoded (retrying cannot succeed)
- **WHEN** Argus processes it
- **THEN** it publishes `status: failed` and acks the message, so it is not redelivered 5×

#### Scenario: Edge case — a poison job does not block the queue
- **GIVEN** a job that repeatedly fails transiently
- **WHEN** its delivery attempts exceed `max-delivery-attempts` (5)
- **THEN** Hermes dead-letters it (to the dead-letter topic, if configured) and the queue keeps flowing
