"""Image preprocessing shared by the taggers (task 1.3).

Pure numpy/PIL transforms — no framework dependency, fully unit-testable.
Each model has its own input contract:

* wd-tagger v3 — ~448px, letterbox-padded to a square on **white**, BGR, uint8→float.
* RAM++        — 384px, resized (no pad), RGB, ImageNet mean/std normalized.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_square(image: Image.Image, size: int, fill: int = 255) -> Image.Image:
    """Resize preserving aspect ratio, then pad to ``size``×``size`` (wd-tagger style)."""
    rgb = image.convert("RGB")
    rgb.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    offset = ((size - rgb.width) // 2, (size - rgb.height) // 2)
    canvas.paste(rgb, offset)
    return canvas


def wd_tensor(image: Image.Image, size: int = 448) -> npt.NDArray[np.float32]:
    """Preprocess for wd-tagger v3 ONNX: NHWC, BGR, float32, unscaled (0–255)."""
    padded = letterbox_square(image, size)
    arr = np.asarray(padded, dtype=np.float32)  # HWC, RGB
    arr = arr[:, :, ::-1]  # RGB → BGR
    return arr[np.newaxis, ...]  # NHWC


def ram_tensor(image: Image.Image, size: int = 384) -> npt.NDArray[np.float32]:
    """Preprocess for RAM++: NCHW, RGB, ImageNet-normalized float32."""
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, RGB, 0–1
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    return arr[np.newaxis, ...].astype(np.float32)  # NCHW
