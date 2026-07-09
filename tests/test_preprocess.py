import numpy as np
from PIL import Image

from argus.models.preprocess import letterbox_square, ram_tensor, wd_tensor


def test_letterbox_pads_to_square() -> None:
    out = letterbox_square(Image.new("RGB", (100, 40)), size=64)
    assert out.size == (64, 64)


def test_wd_tensor_shape_and_layout() -> None:
    tensor = wd_tensor(Image.new("RGB", (200, 120)), size=448)
    assert tensor.shape == (1, 448, 448, 3)  # NHWC
    assert tensor.dtype == np.float32
    assert tensor.max() <= 255.0


def test_ram_tensor_shape_and_normalization() -> None:
    tensor = ram_tensor(Image.new("RGB", (200, 120)), size=384)
    assert tensor.shape == (1, 3, 384, 384)  # NCHW
    assert tensor.dtype == np.float32
    # ImageNet-normalized values fall outside 0–1.
    assert tensor.min() < 0.0
