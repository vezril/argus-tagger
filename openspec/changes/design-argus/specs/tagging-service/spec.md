# tagging-service

The multi-model inference pipeline: fetch the sample, run wd-tagger + RAM++ on CPU, threshold,
normalize surface form, and merge into raw suggestions.

## ADDED Requirements

### Requirement: Multi-model inference over mixed content

Argus SHALL run **both** wd-tagger v3 (anime/illustration) and RAM++ (real-world photos) on
each sample and combine their outputs, so mixed anime/art/photo content is covered without a
content classifier. Models SHALL run on **CPU via ONNX Runtime** (no GPU) and SHALL be held
**warm** in memory (loaded once, not per request).

#### Scenario: Both models contribute to the suggestion set
- **GIVEN** a sample image
- **WHEN** Argus tags it
- **THEN** it runs wd-tagger and RAM++ and the merged suggestions may include tags from either model

#### Scenario: Edge case — models stay resident across requests
- **GIVEN** a stream of tag-jobs
- **WHEN** Argus processes them
- **THEN** the model weights are loaded once and reused (no per-image reload)

### Requirement: Thresholding and surface-form normalization

Argus SHALL apply per-model confidence thresholds (e.g. general ≥ 0.35, character ≥ 0.85 for
wd-tagger; RAM++ per its scores) and SHALL normalize each tag's **surface form** — lowercased,
whitespace to single underscores — so both vocabularies feed one consistent form. It SHALL
NOT perform meaning-level merging (aliases/implications are Artemis's job).

#### Scenario: Low-confidence predictions are dropped
- **GIVEN** model outputs with a spread of confidences
- **WHEN** Argus thresholds them
- **THEN** predictions below the per-category threshold are excluded from suggestions

#### Scenario: Edge case — natural-language tags are underscored
- **GIVEN** a RAM++ output like `teddy bear`
- **WHEN** Argus normalizes it
- **THEN** it becomes `teddy_bear` (surface-form only; not aliased to another concept)

### Requirement: Merge with agreement-boosted confidence

When both models yield the same normalized tag, Argus SHALL merge them into a single
suggestion keeping the higher confidence (agreement is a strong signal), while preserving the
source information for downstream use.

#### Scenario: Cross-model agreement keeps the max confidence
- **GIVEN** wd-tagger emits `tree` at 0.7 and RAM++ emits `tree` at 0.9
- **WHEN** Argus merges
- **THEN** the suggestion for `tree` carries confidence 0.9 (and notes both sources agreed)

#### Scenario: Edge case — disjoint tags both pass through
- **GIVEN** `1girl` (wd only) and `beach` (ram only)
- **WHEN** Argus merges
- **THEN** both appear in the suggestion set with their own source and confidence
