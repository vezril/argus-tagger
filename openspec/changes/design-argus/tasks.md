# Tasks: design-argus

Buildout for the Python auto-tagging service.

> **Scaffold status (this session).** The pure pipeline, the Hermes REST client, and the
> consumer are implemented and verified (33 tests pass, mypy + ruff clean, stub e2e smoke OK).
> Remaining unchecked tasks are **blocked on your hardware / infra / model weights** — noted
> inline. Model inference is wired but needs weights + the `ml` extra on the target node.

## 0. Feasibility spike (gate — do BEFORE committing to the two-model pipeline)

- [ ] 0.a RAM++ latency spike: run `inference_ram_plus.py` in **torch CPU** on the *target node*
      (the CPU-headroom laptop/NAS) over ~20 representative images; record seconds/image. This
      validates the "~seconds/image, async → fine" assumption that the whole two-model design
      rests on. If it's intolerable → reconsider (wd-only v1, or a lighter photo tagger).
- [ ] 0.b Confirm the per-model runtime decision holds (wd→ONNX, RAM++→torch); only pursue a
      RAM++ ONNX export (frozen tag embeddings → static image-encoder+decoder graph) if 0.a shows
      latency actually hurts.

## 0. Scaffold

- [ ] 0.1 Python project (uv/poetry) + Dockerfile → image published to Docker Hub; CI
      — *code done* (`pyproject.toml`, `Dockerfile`, src layout); **blocked**: Docker Hub publish + CI workflow (infra)
- [x] 0.2 Config (env): Hermes base URL, topic/subscription, Apollo endpoint, thresholds, model paths — `config.py`
- [x] 0.3 A minimal health/readiness endpoint (models loaded = ready) — `health.py` (readiness = `pipeline.ready`)

## 1. Models

- [ ] 1.1 Pull wd-tagger v3 (base, ONNX) + RAM++ (swin_large, Apache-2.0) weights from HuggingFace; bake or mount from NAS
      — **blocked**: weight download + NAS mount (target node)
- [ ] 1.2 Load both **warm** (loaded once): wd-tagger as an **ONNX Runtime CPU** session; RAM++ as a **torch CPU** model (`set_num_threads`); load `selected_tags` (wd) + the frozen `ram_plus_tag_embedding_class_4585_des_51` head (RAM++)
      — *loaders wired* (`wd_tagger.py`, `ram_plus.py`); **blocked**: needs weights to load; RAM++ per-tag logit path (`_tag_logits`) to wire against the pinned `ram` version
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
