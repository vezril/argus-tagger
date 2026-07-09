"""wd-tagger v3 (SmilingWolf) — Danbooru-style tags via ONNX Runtime, CPU (task 1.2).

Natively ONNX and Apache-2.0, so this is the *ONNX* half of the per-model runtime
split. Weights + ``selected_tags.csv`` are fetched from HuggingFace (task 1.1) and
mounted/baked; this class only loads and runs them.

``onnxruntime`` is imported lazily inside :meth:`load` so importing this module
(and running the test suite) needs no ML stack.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from argus.contract import Source
from argus.models.base import Prediction
from argus.models.preprocess import wd_tensor

if TYPE_CHECKING:
    from PIL.Image import Image

# selected_tags.csv category ids (SmilingWolf convention).
_CATEGORY_GENERAL = 0
_CATEGORY_CHARACTER = 4
_CATEGORY_RATING = 9

_CATEGORY_NAME = {_CATEGORY_GENERAL: "general", _CATEGORY_CHARACTER: "character"}


class WdTagger:
    """Warm wd-tagger v3 ONNX session with per-category thresholding."""

    source = Source.WD

    def __init__(
        self,
        model_path: Path,
        tags_path: Path,
        *,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        num_threads: int = 0,
    ) -> None:
        self._model_path = model_path
        self._tags_path = tags_path
        self._general_threshold = general_threshold
        self._character_threshold = character_threshold
        self._num_threads = num_threads
        self._session: Any | None = None
        self._labels: list[tuple[str, int]] = []  # (name, category)

    @property
    def ready(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"wd-tagger weights missing at {self._model_path} — fetch from HuggingFace "
                "(SmilingWolf/wd-v3) and mount/bake them (tasks 1.1). Install the `ml` extra."
            )
        import onnxruntime as ort  # lazy: only needed on the target node

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._num_threads
        self._session = ort.InferenceSession(
            str(self._model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._labels = _load_labels(self._tags_path)

    def predict(self, image: Image) -> list[Prediction]:
        if self._session is None:
            raise RuntimeError("WdTagger.load() must be called before predict()")

        tensor = wd_tensor(image)
        input_name = self._session.get_inputs()[0].name
        scores: npt.NDArray[np.float32] = self._session.run(None, {input_name: tensor})[0][0]

        predictions: list[Prediction] = []
        for (name, category), score in zip(self._labels, scores, strict=True):
            if category == _CATEGORY_RATING:
                continue
            threshold = (
                self._character_threshold
                if category == _CATEGORY_CHARACTER
                else self._general_threshold
            )
            if score >= threshold:
                predictions.append(
                    Prediction(
                        tag=name,
                        confidence=float(score),
                        category=_CATEGORY_NAME.get(category),
                    )
                )
        return predictions


def _load_labels(tags_path: Path) -> list[tuple[str, int]]:
    """Parse ``selected_tags.csv`` into an ordered ``(name, category)`` list."""
    if not tags_path.exists():
        raise FileNotFoundError(f"wd-tagger label map missing at {tags_path}")
    labels: list[tuple[str, int]] = []
    with tags_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append((row["name"], int(row["category"])))
    return labels
