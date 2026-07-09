"""RAM++ (Recognize Anything Plus) — real-world tags via torch CPU (task 1.2).

RAM++ (Swin-Large @384, Apache-2.0) has **no official ONNX export** and its
cross-attention tag decoder resists a clean trace, so — per the revised runtime
decision in design.md — it runs in **plain torch on CPU**, honoring "no GPU"
without the export gamble. An ONNX export stays a later optimization (the 4,585
tag embeddings are frozen/precomputed, so inference is a static
image-encoder→decoder→logits graph) and is gated on the §0.a latency spike.

``torch`` and the upstream ``ram`` package are imported lazily inside :meth:`load`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from argus.contract import Source
from argus.models.base import Prediction
from argus.models.preprocess import ram_tensor

if TYPE_CHECKING:
    from PIL.Image import Image

_IMAGE_SIZE = 384


class RamPlusTagger:
    """Warm RAM++ torch-CPU model with logit thresholding."""

    source = Source.RAM

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.68,
        num_threads: int = 0,
    ) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._num_threads = num_threads
        self._model: Any | None = None
        self._tag_list: list[str] = []

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"RAM++ weights missing at {self._model_path} — fetch "
                "xinyu1205/recognize-anything-plus-model from HuggingFace (task 1.1) "
                "and install the `ml` extra (torch + recognize-anything)."
            )
        import torch  # lazy: only needed on the target node
        from ram.models import ram_plus

        if self._num_threads:
            torch.set_num_threads(self._num_threads)
        model = ram_plus(pretrained=str(self._model_path), image_size=_IMAGE_SIZE, vit="swin_l")
        model.eval()
        self._model = model
        # The frozen English tag list the 4,585-way head is aligned to.
        self._tag_list = list(model.tag_list)

    def predict(self, image: Image) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("RamPlusTagger.load() must be called before predict()")
        import torch

        tensor = torch.from_numpy(ram_tensor(image, _IMAGE_SIZE))
        with torch.no_grad():
            # NOTE: the upstream `ram` helper (`inference_ram`) returns tag *strings*
            # with no confidences. To get per-tag scores we run the image encoder +
            # tagging head and read the logits directly. The exact attribute path into
            # RAM++'s head must be confirmed against the pinned `ram` version during the
            # §0.a spike / task 1.2 — hence isolated in `_tag_logits`.
            logits = _tag_logits(self._model, tensor)  # [1, num_tags]
        scores: npt.NDArray[np.float32] = torch.sigmoid(logits)[0].cpu().numpy()

        return [
            Prediction(tag=tag, confidence=float(score), category=None)
            for tag, score in zip(self._tag_list, scores, strict=True)
            if score >= self._threshold
        ]


def _tag_logits(model: Any, tensor: Any) -> Any:
    """Extract raw per-tag logits from a RAM++ model for one preprocessed image.

    PLACEHOLDER — the real path (image encoder → cross-attention tagging head)
    depends on the pinned ``ram`` version's internals and must be wired against
    the actual model during task 1.2. Kept as a seam so the rest of the class is
    complete and testable.
    """
    raise NotImplementedError(
        "RAM++ per-tag logit extraction is not yet wired to the `ram` model internals "
        "(task 1.2 / §0.a spike). See the note in RamPlusTagger.predict()."
    )
