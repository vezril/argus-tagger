# Design: Argus (auto-tagging service)

The all-seeing tagger. Captured in explore mode; no implementation.

## Where Argus sits

```
   Hephaestus (media processing) → MediaProcessed → Artemis (post active)
                                        │
                        Artemis publishes TagJob → HermesMQ topic: media.tag
                                        │  (decoupled — Argus at its own pace)
                                        ▼
   Argus (Python):  pull TagJob ── fetch sample from Apollo ── wd-tagger + RAM++ ──
                    threshold → normalize → merge → publish TagSuggestions
                                        │
                    Artemis: alias-resolve → dedup(max conf) → store as canonical
                             suggestions · flag post "needs-review"
                                        │
   Muses Review queue → accept/tweak → applied as tags → reviewed
```

Argus is the one **non-JVM** service — isolated on purpose. It speaks two protocols to the
fleet: **HermesMQ REST** (consume/ack) and **Apollo** (read the sample). Everything else is
local ML.

## Multi-model inference (mixed content)

```
   sample image (Apollo, ~448px)
        ├─▶ wd-tagger v3 (ONNX)   → Danbooru-style tags + rating   (great on anime/art)
        └─▶ RAM++       (ONNX)    → real-world tags                (great on photos)
        │ threshold each (e.g. general ≥0.35, character ≥0.85; RAM per its scores)
        │ normalize surface form (lowercase, spaces→underscores; "teddy bear"→teddy_bear)
        ▼ merge: union, keep MAX confidence where a canonical tag appears from both
   raw suggestions: [{tag, category?, confidence, source: wd|ram}]
```

Run-both-and-merge is chosen over a classify-and-route because it's simpler, robust to
mis-classification, and any noise is filtered by the human review anyway. **Namespace
merging is Artemis's job, not Argus's** — Argus emits raw tags from both vocabularies;
Artemis's alias/implication canonicalization collapses `outdoor→outdoors` etc. (Argus only
normalizes *surface form*, not *meaning*.)

- **Models are warm** — loaded once into memory in a long-lived process, never per-request.
- **Weights from HuggingFace** — pulled once, baked into the image or mounted from the NAS.
- **CPU, no GPU — runtime chosen per model** (revised from "CPU ONNX Runtime for both"):
  - **wd-tagger v3 → ONNX Runtime** — natively ONNX, Apache-2.0, light (ViT/ConvNeXt @448,
    ~sub-second CPU). Keep as-is.
  - **RAM++ → plain torch CPU** (default). RAM++ (Swin-Large @384, ~197M params, Apache-2.0)
    has **no official ONNX** and its cross-attention tag decoder fights a clean trace, so
    mandating ONNX adds risk for no clear win at personal, async, review-gated scale. Its
    4,585-tag head uses **frozen precomputed tag embeddings** (`ram_plus_tag_embedding_...pth`),
    so the text encoder is *not* needed at inference — an ONNX export is feasible *later* as an
    optimization (image-encoder + decoder → 4585 logits), not a prerequisite.
  - Trade-off: two runtimes (onnxruntime + torch) in one image — larger image, fine here.
- **Latency — spike done (task 0.a), assumption HOLDS.** `spikes/ram_plus_latency.py` measured
  RAM++ at **0.75s/image single-core / ~0.45s multi** on an M4 Pro floor (parallelism plateaus
  at ~4 threads — per-core-speed bound), 4.8s warm load. Extrapolated to the target node
  (~5× slower per-core) ≈ **~1–3s/image**; with wd-tagger (~0.3–1s) ≈ ~1.5–4s total — fine for
  async, review-gated, personal scale. Rerun on the real node to confirm; ⚠️ RAM++ resident
  ≈ **3–5 GB** so the node needs **≥8 GB**.
- **RAM++ scoring is wired.** Argus reproduces `ram_plus.generate_tag`'s forward but keeps the
  per-tag sigmoid confidences (`ram_plus.py:_tag_scores`), and applies RAM++'s **per-class**
  thresholds (`model.class_threshold`, ~0.45–1.00) rather than one global value.
- **The `ml` stack is fragile** — `recognize-anything` under-declares its deps; it needs
  `transformers==4.25.1` (5.x breaks it), `scipy`, and `fairscale` pinned (now in `pyproject.toml`).
- Licensing: **both Apache 2.0** (wd-tagger v3, RAM++) — gate closed.

## Consuming HermesMQ *correctly* (the client)

Hermes ships a Scala client; Argus is Python, so it needs its **own client over Hermes's
REST API** — and inference is *slow* (seconds/image), which makes lease handling the crux:

```
   ensure a subscription to media.tag (idempotent create)
   loop:
     pull(max = 1..3)             ← BACKPRESSURE + fits the shared 30s lease
        (ack-deadline is BROKER-GLOBAL, default 30s — NOT settable per pull)
     for each message:
        process (fetch sample, infer, publish suggestions)
        if inference runs long → modifyAckDeadline (extend the lease) so it isn't
           redelivered mid-flight
        ack(ackId)  ← only AFTER suggestions are durably published
     (a crash before ack → Hermes redelivers → safe: tagging is idempotent per postId)
```

- **Tiny pull batches (1–3)** — the ack deadline is a broker-global 30s (`HERMESMQ_ACK_DEADLINE`),
  set once per subscription, not per pull; a small batch keeps a serial run inside that lease.
- **Extend the lease** for long inference (`modifyAckDeadline`) rather than risk redelivery.
- **Ack after publish**, not before — at-least-once means a redelivered job just re-tags
  (Artemis dedups suggestions by post).
- **Failure branch by cause:** permanent (undecodable sample) → publish `failed` + **ack**
  (don't burn 5 redeliveries); transient (Apollo down) → **nack** (`modifyAckDeadline → 0`) →
  redelivers. Hermes dead-letters after `max-delivery-attempts` (default 5) — never wedges.

## Messages (the contract)

```
   TagJob            (Artemis → media.tag → Argus)
     postId · sample { bucket, object }  (the Apollo sample ref) · mediaType

   TagSuggestions    (Argus → media.suggestions → Artemis)
     postId · suggestions [{ tag, category?, confidence, source: wd|ram }] · rating?
     status: ok | failed
```

## Deployment

Own Docker image → Docker Hub (matching siblings). Deployed by Codex, **pinned to a
CPU-headroom node** (the Hephaestus-labeled laptop, or the NAS) — inference is the heavy
part. Model weights mounted from the NAS or baked in. One replica is plenty; it just needs
to keep up with your ingest rate, and Hermes buffers if it falls behind.

- **Configure a dead-letter topic** (`HERMESMQ_DEAD_LETTER_TOPIC`, e.g. `media.tag.dlq`) on
  the Hermes deployment — otherwise exhausted poison tag-jobs are **silently dropped**, not
  inspectable. This is a Hermes-side deploy setting Codex owns, called out here as a dependency.

## Out of scope

Auto-applying tags · training/fine-tuning · GPU · multi-frame video sampling · the
namespace *meaning* merge (that's Artemis's alias system).
