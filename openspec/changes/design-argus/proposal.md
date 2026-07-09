# Change: design-argus

> **Design capture (explore mode).** Records the design for **Argus**, the auto-tagging
> service. Argus is a new (currently empty) repo. No code is implemented by this change.

## Why

Tagging is the tedious part of any booru. Argus watches uploaded media and **suggests**
tags — turning tagging from "type from scratch" into "review a pre-filled set." It is the
biggest quality-of-life lever in the constellation (design backlog #2).

It is deliberately a **dedicated Python service** — the one non-JVM member of a Scala/Pekko
fleet — because ML inference is Python's ecosystem, and isolating it keeps that dependency
from leaking into the core services.

## Decisions carried in from exploration

| Decision | Choice |
|----------|--------|
| Form | **Dedicated Python service** (own repo, own image), named **Argus** (the all-seeing) |
| Models | **Multi-model** for mixed content: **wd-tagger v3** (anime/illustration) + **RAM++** (photos) |
| Routing | **Run both, merge** — simplest, robust; noise is fine because everything is review-gated |
| Hardware | **CPU, no GPU**; runtime *per model* (wd-tagger→ONNX; RAM++→torch CPU, ONNX later — no official export); ~seconds/image, async → fine at personal scale. Validate RAM++ latency by spike first. |
| Trigger | **Artemis publishes** a `tag-job` to HermesMQ (Artemis owns tags; Hephaestus stays tag-agnostic) |
| Input | Argus **consumes from Hermes** and **reads the sample derivative from Apollo** (~448px is plenty) |
| Output | **Raw suggestions** (both vocabularies, with confidence + source model) → Artemis alias-merges them |
| Policy | **Suggestions-only** — never auto-applied; a human reviews in Muses |

## What Changes

- **suggestion-contract** (new): the `TagJob` message Argus consumes and the `TagSuggestions`
  message it publishes.
- **hermes-consumer** (new): Argus's own **Python HermesMQ client** — pull-based, lease-aware
  consumption of the `media.tag` queue done *correctly* (respect the ack deadline given slow
  inference, backpressure, idempotency).
- **tagging-service** (new): the multi-model inference pipeline — fetch sample from Apollo,
  run wd-tagger + RAM++, threshold, normalize surface form, merge, emit suggestions; warm
  models, HuggingFace model management, CPU ONNX.

## Impact

- Affected specs: `suggestion-contract`, `hermes-consumer`, `tagging-service` are **ADDED**.
- New repo: `argus-tagger` — Python (FastAPI optional for a health/debug endpoint),
  `onnxruntime`, model weights from HuggingFace. Own Docker image → Docker Hub; deployed by
  Codex, pinned to a CPU-headroom node (the Hephaestus-labeled laptop or the NAS).
- Depends on: HermesMQ (the `media.tag` queue + its REST API), Apollo (sample reads), and
  the counterpart Artemis change `design-artemis-auto-tagging` (which publishes jobs and
  alias-merges the suggestions).
- Content assumption: mixed **anime/art + photos** → the two-model approach; wd-tagger alone
  would be wrong for photos.
- Out of scope: auto-applying tags (always review-gated), training/fine-tuning models,
  GPU inference, video-frame sampling beyond the poster frame.
