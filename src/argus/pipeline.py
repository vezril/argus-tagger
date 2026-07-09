"""The tagging pipeline — run every model, normalize, merge (tagging-service spec).

Ties the model layer to the suggestion contract:

    decoded image → each Tagger.predict() → merge (normalize + dedup/max) → [Suggestion]

Thresholding lives inside each tagger (per-category knobs); this layer is the
model-agnostic composition. Models are held **warm** — loaded once via
:meth:`load`, reused across every job.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from argus.contract import Source, Suggestion
from argus.merge import merge
from argus.models.base import Prediction, Tagger

if TYPE_CHECKING:
    from PIL.Image import Image


class TaggingPipeline:
    """A warm, multi-model tagging pipeline."""

    def __init__(self, taggers: Sequence[Tagger]) -> None:
        if not taggers:
            raise ValueError("TaggingPipeline needs at least one tagger")
        self._taggers = list(taggers)

    @property
    def ready(self) -> bool:
        """Ready only once every model's weights are resident (drives readiness)."""
        return all(t.ready for t in self._taggers)

    def load(self) -> None:
        """Warm every model. Called once at startup."""
        for tagger in self._taggers:
            tagger.load()

    def tag(self, image: Image) -> list[Suggestion]:
        """Run all models on ``image`` and return the merged raw suggestions."""
        predictions: list[tuple[Source, Prediction]] = [
            (tagger.source, pred)
            for tagger in self._taggers
            for pred in tagger.predict(image)
        ]
        return merge(predictions)
