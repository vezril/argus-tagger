"""RAM++ CPU-latency spike (OpenSpec task 0.a).

Measures wall-clock seconds/image for RAM++ (Swin-Large @384) running in **torch
CPU**, to validate the "~seconds/image, async → fine" assumption the two-model
design rests on. Latency is content-independent (a fixed 384x384 forward pass),
so synthetic images give the same timing as real media.

Run on the *target node* (the CPU-headroom laptop / QNAP) for the number that
matters; running on a dev box gives a best-case floor.

Usage:
    python spike_ram_plus_latency.py                 # auto-downloads weights, 20 imgs
    python spike_ram_plus_latency.py --n 30 --threads 4
    python spike_ram_plus_latency.py --model-path /models/ram_plus_swin_large_14m.pth
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time

import torch
from PIL import Image

WEIGHTS_REPO = "xinyu1205/recognize-anything-plus-model"
WEIGHTS_FILE = "ram_plus_swin_large_14m.pth"
IMAGE_SIZE = 384


def resolve_weights(model_path: str | None) -> str:
    if model_path:
        return model_path
    from huggingface_hub import hf_hub_download

    print(f"Downloading {WEIGHTS_FILE} from {WEIGHTS_REPO} (~1.5 GB, one-time)…")
    return hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE)


def synthetic_images(n: int) -> list[Image.Image]:
    """Varied-size noise images — content-independent for latency, but exercises
    the resize path at realistic input sizes."""
    import numpy as np

    sizes = [(512, 768), (1024, 1024), (800, 600), (1920, 1080), (448, 448)]
    imgs = []
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        arr = (np.abs(np.sin(np.arange(w * h * 3) * (i + 1) * 0.001)) * 255).astype("uint8")
        imgs.append(Image.fromarray(arr.reshape(h, w, 3), "RGB"))
    return imgs


def load_images(images_dir: str | None, n: int) -> list[Image.Image]:
    if not images_dir:
        return synthetic_images(n)
    import pathlib

    paths = sorted(
        p for p in pathlib.Path(images_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )[:n]
    return [Image.open(p).convert("RGB") for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None, help="path to ram_plus_swin_large_14m.pth")
    ap.add_argument("--images-dir", default=None, help="dir of real images (else synthetic)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default/all)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    print("=" * 64)
    print("RAM++ CPU-latency spike (task 0.a)")
    print(f"  machine   : {platform.platform()} / {platform.machine()}")
    print(f"  torch     : {torch.__version__}  threads={torch.get_num_threads()}")
    print("=" * 64)

    from ram import get_transform
    from ram import inference_ram as inference
    from ram.models import ram_plus

    weights = resolve_weights(args.model_path)
    transform = get_transform(image_size=IMAGE_SIZE)

    t0 = time.perf_counter()
    model = ram_plus(pretrained=weights, image_size=IMAGE_SIZE, vit="swin_l")
    model.eval()
    model = model.to("cpu")
    load_s = time.perf_counter() - t0
    print(f"model load (warm startup): {load_s:.1f}s")

    images = load_images(args.images_dir, args.n)
    tensors = [transform(im).unsqueeze(0).to("cpu") for im in images]

    with torch.no_grad():
        for i in range(min(args.warmup, len(tensors))):
            inference(tensors[i], model)

        times = []
        for i, t in enumerate(tensors):
            s = time.perf_counter()
            inference(t, model)
            dt = time.perf_counter() - s
            times.append(dt)
            print(f"  img {i + 1:2d}/{len(tensors)}: {dt:.2f}s")

    times.sort()
    p95 = times[min(len(times) - 1, int(round(0.95 * len(times))) - 1)]
    print("-" * 64)
    print(f"RESULT over {len(times)} images (threads={torch.get_num_threads()}):")
    print(f"  mean   {statistics.mean(times):.2f}s   median {statistics.median(times):.2f}s")
    print(f"  min    {min(times):.2f}s   max {max(times):.2f}s   p95 {p95:.2f}s")
    print(f"  throughput ~{1 / statistics.mean(times):.2f} img/s single-worker")
    print("=" * 64)


if __name__ == "__main__":
    main()
