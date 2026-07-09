# Tasks: design-argus

Buildout for the Python auto-tagging service.

> **Scaffold status (this session).** The pure pipeline, the Hermes REST client, and the
> consumer are implemented and verified (33 tests pass, mypy + ruff clean, stub e2e smoke OK).
> Remaining unchecked tasks are **blocked on your hardware / infra / model weights** — noted
> inline. Model inference is wired but needs weights + the `ml` extra on the target node.

## 0. Feasibility spike (gate — do BEFORE committing to the two-model pipeline)

- [x] 0.a RAM++ latency spike — **DONE. Verdict: the two-model latency assumption HOLDS.**
      Measured via `spikes/ram_plus_latency.py`: on an M4 Pro (arm64) floor, **0.75s/image
      single-core, ~0.45s multi** (parallelism plateaus at ~4 threads → per-core-speed bound),
      4.8s warm load. Extrapolated to the target node (QNAP N5105 / old laptops, ~5× slower
      per-core) ≈ **~1–3s/image** — comfortably within "seconds/image, async" at personal scale.
      No redesign needed. ⚠️ Numbers are extrapolated from a dev box, not the target — rerun the
      script on the real node to confirm. Memory: RAM++ resident ≈ 3–5 GB → node needs **≥8 GB**
      (a base-4 GB QNAP won't fit). Also surfaced: the `ml` stack needs pinned deps (see 1.1).
- [x] 0.b Per-model runtime decision **CONFIRMED** (wd→ONNX, RAM++→torch CPU). RAM++ latency is
      fine as-is, so no ONNX export is warranted now. Bonus from the spike: the per-tag logit
      path is validated and wired (`ram_plus.py:_tag_scores`), and RAM++ uses **per-class**
      thresholds (`model.class_threshold`), now used instead of a single value.

## 0. Scaffold

- [x] 0.1 Python project (uv/poetry) + Dockerfile → image published to Docker Hub; CI — **DONE.**
      Scaffold + Dockerfile; OLYMPUS-style CI/dev/release workflows. The `.[ml,grpc]` image builds
      (validated + 5 fixes, PR #4) and **publishes**: the first `development` push had `dev.yml`
      push `calvinference/argus-tagger:dev-<sha>` (~358 MB) to Docker Hub. Base is **py3.11**
      (RAM++'s pinned transformers/tokenizers can't do 3.12).
- [x] 0.2 Config (env): Hermes base URL, topic/subscription, Apollo endpoint, thresholds, model paths — `config.py`
- [x] 0.3 A minimal health/readiness endpoint (models loaded = ready) — `health.py` (readiness = `pipeline.ready`)

## 1. Models

- [ ] 1.1 Pull wd-tagger v3 (base, ONNX) + RAM++ (swin_large, Apache-2.0) weights from HuggingFace; bake or mount from NAS
      — **blocked**: weight download + NAS mount (target node). RAM++ weights = `ram_plus_swin_large_14m.pth` (~1.5 GB) from `xinyu1205/recognize-anything-plus-model`. ml deps now pinned in `pyproject.toml` (transformers==4.25.1, scipy, fairscale) per the §0.a spike.
- [ ] 1.2 Load both **warm** (loaded once): wd-tagger as an **ONNX Runtime CPU** session; RAM++ as a **torch CPU** model (`set_num_threads`); load `selected_tags` (wd) + the frozen `ram_plus_tag_embedding_class_4585_des_51` head (RAM++)
      — *loaders wired* (`wd_tagger.py`, `ram_plus.py`). RAM++ per-tag logit path now **wired + validated** in the §0.a spike (`ram_plus.py:_tag_scores`, per-class thresholds). **Blocked** only on real weights for an on-node integration test.
- [x] 1.3 Preprocess per model (wd ~448 / RAM++ 384: resize/pad/normalize) + postprocess (threshold) — `preprocess.py`

## 2. Tagging pipeline

- [x] 2.1 Run both models; threshold per category; normalize surface form (lowercase, underscores) — `pipeline.py` + `normalize.py` + per-tagger thresholds
- [x] 2.2 Merge: agreement keeps max confidence; keep source; assemble raw suggestions — `merge.py`
- [ ] 2.3 Re-bench per-model inference on the target CPU with the real pipeline (confirms the 0.a spike at scale); tune thread counts / base-model choice for latency
      — **blocked**: target hardware

## 3. Apollo reader

- [ ] 3.1 Fetch the sample derivative by ref (gRPC client) for a `TagJob`
      — *port + streaming stub done* (`apollo.py`); **blocked**: generate Python stubs from the external Lexicon proto (`io.codex:lexicon-grpc`)
- [x] 3.2 Handle unreadable/undecodable samples → `failed` result (no wedge) — `decode_image` → `PermanentFailure` → consumer publishes `failed` + ack

## 4. HermesMQ consumer (the client)

- [x] 4.1 Python client over Hermes REST: ensure subscription, pull, modifyAckDeadline, ack — `hermes.py` (verified vs `PubSubRoutes`)
- [x] 4.2 Small-batch pull (backpressure); generous ack deadline — `pull_max` default 2; `ACK_DEADLINE_SECONDS` config
- [x] 4.3 Extend lease for long inference; ack only after publishing suggestions — `consumer.py` (ack-after-publish invariant, incl. publish-failure → nack)
- [x] 4.4 Idempotent per postId; failed/poison → nack/dead-letter (no wedge) — failure branch in `consumer.py`; DLQ is Hermes-side

## 5. Publish + integrate

- [x] 5.1 Publish `TagSuggestions` to the results topic — `hermes.publish` + `consumer._handle`
- [x] 5.2 End-to-end (mock Artemis): consume TagJob → tag sample → publish suggestions — `test_consumer.py` (FakeHermes + stubs) + stub smoke
- [ ] 5.3 Deploy via Codex: pin to a CPU-headroom node; mount model weights; one replica. Ensure Hermes has a dead-letter topic set (`HERMESMQ_DEAD_LETTER_TOPIC`, e.g. `media.tag.dlq`) or poison jobs are silently dropped
      — **blocked**: Codex GitOps deploy (infra)
- [x] 5.4 README + license (match sibling repos) — `README.md`, `LICENSE` (MIT)
