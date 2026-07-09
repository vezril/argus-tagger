"""RAM++ (Recognize Anything Plus) — real-world tags via torch CPU (task 1.2).

RAM++ (Swin-Large @384, Apache-2.0) has **no official ONNX export** and its
cross-attention tag decoder resists a clean trace, so — per the revised runtime
decision in design.md — it runs in **plain torch on CPU**, honoring "no GPU"
without the export gamble. Validated by the §0.a spike: ~0.75s/image single-core
on an M4 Pro (~0.45s multi), extrapolating to ~1-3s on the target node — well
within "seconds/image, async" at personal scale.

``torch`` and the upstream ``ram`` package are imported lazily inside :meth:`load`.
The scoring path (:meth:`_tag_scores`) mirrors ``ram.models.ram_plus.generate_tag``
but returns per-tag sigmoid confidences instead of thresholded strings, and applies
RAM++'s own **per-class** thresholds (``model.class_threshold``, 4,585 values).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from argus.contract import Source
from argus.models.base import Prediction
from argus.models.preprocess import ram_tensor

if TYPE_CHECKING:
    from PIL.Image import Image

_IMAGE_SIZE = 384


class RamPlusTagger:
    """Warm RAM++ torch-CPU model with per-class thresholding."""

    source = Source.RAM

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float | None = None,
        num_threads: int = 0,
    ) -> None:
        """``threshold`` overrides RAM++'s per-class thresholds with a single
        uniform value when set; leave ``None`` to use the model's own per-tag
        thresholds (recommended — they range ~0.45–1.00)."""
        self._model_path = model_path
        self._threshold = threshold
        self._num_threads = num_threads
        self._model: Any | None = None
        self._tag_list: list[str] = []
        self._thresholds: Any | None = None  # torch.Tensor[num_class]
        self._delete_index: list[int] = []

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
                "and install the `ml` extra (torch + recognize-anything + the pinned deps)."
            )
        import torch
        from ram.models import ram_plus

        if self._num_threads:
            torch.set_num_threads(self._num_threads)
        model = ram_plus(pretrained=str(self._model_path), image_size=_IMAGE_SIZE, vit="swin_l")
        model.eval()
        self._model = model
        self._tag_list = list(model.tag_list)
        # Per-class thresholds (or a uniform override), and the tags RAM suppresses.
        if self._threshold is None:
            self._thresholds = model.class_threshold.clone()
        else:
            self._thresholds = torch.full((model.num_class,), float(self._threshold))
        self._delete_index = list(model.delete_tag_index)

    def predict(self, image: Image) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("RamPlusTagger.load() must be called before predict()")
        import torch

        tensor = torch.from_numpy(ram_tensor(image, _IMAGE_SIZE))
        with torch.no_grad():
            scores = _tag_scores(self._model, tensor)  # [num_class] sigmoid confidences
        thresholds = self._thresholds
        keep = (scores > thresholds).nonzero().squeeze(-1).tolist()

        delete = set(self._delete_index)
        return [
            Prediction(tag=self._tag_list[i], confidence=float(scores[i]), category=None)
            for i in keep
            if i not in delete
        ]


def _tag_scores(model: Any, image: Any) -> Any:
    """Per-tag sigmoid confidences for one preprocessed image ``[1,3,384,384]``.

    Mirrors ``ram.models.ram_plus.generate_tag`` (validated against the pinned
    ``ram`` version in the §0.a spike) but returns the sigmoid scores rather than
    thresholded tag strings, so Argus keeps per-tag confidence for its suggestions.
    """
    import torch
    import torch.nn.functional as f

    embeds = model.image_proj(model.visual_encoder(image))
    atts = torch.ones(embeds.size()[:-1], dtype=torch.long, device=image.device)
    cls = embeds[:, 0, :]
    cls = cls / cls.norm(dim=-1, keepdim=True)

    des_per_class = int(model.label_embed.shape[0] / model.num_class)
    logits_per_image = (model.reweight_scale.exp() * cls @ model.label_embed.t())
    logits_per_image = logits_per_image.view(1, -1, des_per_class)
    weight = f.softmax(logits_per_image, dim=2)
    reshaped = model.label_embed.view(-1, des_per_class, 512)
    reweighted = (weight[0].unsqueeze(-1) * reshaped).sum(dim=1).unsqueeze(0)
    label_embed = f.relu(model.wordvec_proj(reweighted))

    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=embeds,
        encoder_attention_mask=atts,
        return_dict=False,
        mode="tagging",
    )
    return torch.sigmoid(model.fc(tagging_embed[0]).squeeze(-1))[0]
