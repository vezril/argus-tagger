<!-- Badges reference this repo slug; adjust OWNER if the GitHub path differs. -->

# Argus

[![CI](https://github.com/vezril/argus-tagger/actions/workflows/ci.yml/badge.svg)](https://github.com/vezril/argus-tagger/actions/workflows/ci.yml)
[![Release](https://github.com/vezril/argus-tagger/actions/workflows/release.yml/badge.svg)](https://github.com/vezril/argus-tagger/actions/workflows/release.yml)
[![Dev publish](https://github.com/vezril/argus-tagger/actions/workflows/dev.yml/badge.svg)](https://github.com/vezril/argus-tagger/actions/workflows/dev.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The all-seeing tagger.** Argus is the auto-tagging service of the Codex media
constellation — and its one **non-JVM** member. It watches uploaded media and
**suggests** tags, turning tagging from "type from scratch" into "review a
pre-filled set." Suggestions are never auto-applied; a human reviews them in Muses.

```
  Artemis ──TagJob──▶ HermesMQ (media.tag) ──▶ Argus ──▶ read sample (Apollo)
                                                  │        wd-tagger v3 + RAM++
                                                  │        threshold · normalize · merge
                                                  ▼
  Artemis ◀──TagSuggestions── HermesMQ (media.suggestions) ◀──┘
     alias-resolve · dedup(max) · flag needs-review ──▶ Muses review queue
```

## Why Python

ML inference is Python's ecosystem. Isolating it as a dedicated service keeps that
dependency out of the Scala/Pekko fleet. Argus speaks two protocols outward —
**HermesMQ REST** (consume/ack) and **Apollo gRPC** (read the sample) — everything
else is local CPU inference.

## Models (CPU, no GPU)

Runtime is chosen **per model**:

| Model | Content | Runtime | License |
|-------|---------|---------|---------|
| **wd-tagger v3** (SmilingWolf) | anime / illustration | ONNX Runtime | Apache-2.0 |
| **RAM++** (Recognize Anything Plus) | real-world photos | torch (CPU) | Apache-2.0 |

Run-both-and-merge (rather than classify-and-route) is simple and robust to
misclassification — any noise is filtered by the human review anyway. Argus
normalizes only **surface form**; meaning-level alias merging is Artemis's job.

> **RAM++ note.** No official ONNX export exists; torch-CPU is the pragmatic path.
> An ONNX export stays a later optimization (its 4,585 tag embeddings are frozen,
> so inference is a static graph) and is gated on a latency spike on the target node.

## Develop

```bash
uv sync                 # core deps only — no ML stack needed for tests/lint
uv run pytest           # the pure pipeline, Hermes client, and consumer are fully tested
uv run ruff check .
uv run mypy

# End-to-end against a real Hermes, no weights required:
ARGUS_STUB_MODELS=1 uv run argus
```

Install the heavy inference stack only on the target node:

```bash
uv sync --extra ml --extra grpc
```

## Configuration

All via env (prefix `ARGUS_`): `HERMES_BASE_URL`, `SUBSCRIPTION_ID`, `SOURCE_TOPIC`
(`media.tag`), `RESULTS_TOPIC` (`media.suggestions`), `APOLLO_ENDPOINT`, `PULL_MAX`,
`ACK_DEADLINE_SECONDS`, `WD_MODEL_PATH`, `WD_TAGS_PATH`, `RAM_MODEL_PATH`,
`GENERAL_THRESHOLD`, `CHARACTER_THRESHOLD`, `RAM_THRESHOLD`, `NUM_THREADS`,
`HEALTH_PORT`, `STUB_MODELS`.

## Deploy

Own Docker image → Docker Hub (matching siblings). Deployed by Codex, **pinned to a
CPU-headroom node**; model weights mounted from the NAS at `/models`. One replica is
plenty — Hermes buffers if Argus falls behind.

> Set a **dead-letter topic** on Hermes (`HERMESMQ_DEAD_LETTER_TOPIC`, e.g.
> `media.tag.dlq`) or exhausted poison jobs are silently dropped.

## Status

Scaffold from the `design-argus` OpenSpec change. The pure pipeline, the Hermes
REST client, and the consumer are implemented and tested; model inference is wired
but needs weights + the `ml` extra on the target node (see `openspec/` and the
task list for what remains — the RAM++ latency spike gates the two-model design).
