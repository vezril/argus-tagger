"""Merge raw model predictions into the suggestion set (tagging-service spec, task 2.2).

Rules (from the spec):

* Predictions are keyed by their **normalized surface form**.
* When both models yield the same normalized tag, merge into one suggestion
  keeping the **higher confidence** (agreement is a strong signal); the source
  recorded is the model that produced that winning confidence.
* Disjoint tags pass through with their own source and confidence.

Pure function — no I/O, no model dependency.
"""

from __future__ import annotations

from collections.abc import Iterable

from argus.contract import Source, Suggestion
from argus.models.base import Prediction
from argus.normalize import normalize_surface_form


def merge(predictions: Iterable[tuple[Source, Prediction]]) -> list[Suggestion]:
    """Collapse ``(source, prediction)`` pairs into a deduped suggestion list.

    Same normalized tag from multiple sources → one suggestion at the max
    confidence, sourced to the winning model. Output is sorted by descending
    confidence so the highest-signal tags surface first in review.
    """
    best: dict[str, Suggestion] = {}
    for source, pred in predictions:
        tag = normalize_surface_form(pred.tag)
        if not tag:
            continue
        candidate = Suggestion(
            tag=tag,
            confidence=pred.confidence,
            source=source,
            category=pred.category,
        )
        current = best.get(tag)
        if current is None or candidate.confidence > current.confidence:
            best[tag] = candidate

    return sorted(best.values(), key=lambda s: s.confidence, reverse=True)
