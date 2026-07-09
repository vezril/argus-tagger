"""Apollo sample reader (tasks 3.1–3.2).

Argus reads the **sample derivative** (~448px) for a post from Apollo. Apollo
exposes ``GetObject`` as a **server-streaming gRPC** call (header then chunks) on
its ``GRPC_PORT``.

Dependency note: Apollo's gRPC contract is **not** vendored in that repo — it
lives in the external "Lexicon" artifact (``io.codex:lexicon-grpc``, GitHub
Packages, needs a ``read:packages`` token). So :class:`GrpcSampleReader` is a
seam: generate the Python stubs from the Lexicon ``.proto`` and wire them in.
Until then, :class:`StubSampleReader` serves the pipeline/consumer tests and the
``ARGUS_STUB_MODELS`` local path.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from PIL import Image, UnidentifiedImageError

from argus.errors import PermanentFailure, TransientFailure

if TYPE_CHECKING:
    from argus.contract import SampleRef


@runtime_checkable
class SampleReader(Protocol):
    """Reads the raw bytes of a sample derivative referenced by a job."""

    def read(self, ref: SampleRef) -> bytes: ...


def decode_image(data: bytes) -> Image.Image:
    """Decode sample bytes into an image.

    A sample that cannot be decoded is a :class:`PermanentFailure` — retrying
    can't help, so the consumer publishes ``failed`` and acks (no wedge).
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PermanentFailure(f"undecodable sample: {exc}") from exc


class GrpcSampleReader:
    """Apollo ``GetObject`` client. STUB — awaits the Lexicon gRPC stubs.

    Server-streaming: read the header frame then concatenate payload chunks.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def read(self, ref: SampleRef) -> bytes:
        raise NotImplementedError(
            "Apollo GetObject client not wired — generate Python stubs from the Lexicon "
            "proto (io.codex:lexicon-grpc) and stream header+chunks here (tasks 3.1). "
            f"endpoint={self._endpoint!r}, ref={ref!r}"
        )


class StubSampleReader:
    """In-memory reader for tests and ``ARGUS_STUB_MODELS`` — returns a tiny PNG,
    or raises to exercise the consumer's failure branches."""

    def __init__(self, data: bytes | None = None, *, fail: Exception | None = None) -> None:
        self._data = data if data is not None else _tiny_png()
        self._fail = fail

    def read(self, ref: SampleRef) -> bytes:  # noqa: ARG002 - stub ignores ref
        if self._fail is not None:
            raise self._fail
        return self._data


def _tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (127, 127, 127)).save(buf, format="PNG")
    return buf.getvalue()


__all__ = [
    "GrpcSampleReader",
    "PermanentFailure",
    "SampleReader",
    "StubSampleReader",
    "TransientFailure",
    "decode_image",
]
