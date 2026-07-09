"""The inference port every tagger implements, plus a weightless stub.

Keeping models behind a small :class:`Tagger` protocol lets the pipeline, the
consumer, and the whole test suite run without onnxruntime, torch, or any model
weights — the heavy concrete taggers (:mod:`argus.models.wd_tagger`,
:mod:`argus.models.ram_plus`) import their frameworks lazily inside ``load()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from argus.contract import Source

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class Prediction:
    """One raw model output, before surface-form normalization."""

    tag: str
    confidence: float
    category: str | None = None


@runtime_checkable
class Tagger(Protocol):
    """A warm, single-source image tagger."""

    @property
    def source(self) -> Source: ...

    @property
    def ready(self) -> bool:
        """True once weights are loaded into memory."""
        ...

    def load(self) -> None:
        """Load weights once (warm). Idempotent."""
        ...

    def predict(self, image: Image) -> list[Prediction]:
        """Run inference on a decoded image, returning thresholded predictions."""
        ...


class StubTagger:
    """A deterministic, weightless tagger for local dev and tests.

    Enabled in-process via ``ARGUS_STUB_MODELS=1`` so the full consume → tag →
    publish path (task 5.2) runs end-to-end with no models present.
    """

    def __init__(self, source: Source, predictions: list[Prediction] | None = None) -> None:
        self._source = source
        self._predictions = predictions or [
            Prediction(tag="1girl", confidence=0.98, category="general")
            if source is Source.WD
            else Prediction(tag="teddy bear", confidence=0.72, category=None)
        ]
        self._ready = False

    @property
    def source(self) -> Source:
        return self._source

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        self._ready = True

    def predict(self, image: Image) -> list[Prediction]:  # noqa: ARG002 - stub ignores input
        return list(self._predictions)
