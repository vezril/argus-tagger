"""The suggestion contract — the messages Argus exchanges with Artemis over HermesMQ.

Implements the ``suggestion-contract`` spec. Two messages:

* ``TagJob``          — consumed from ``media.tag``      (Artemis → Argus)
* ``TagSuggestions``  — published to ``media.suggestions`` (Argus → Artemis)

**Envelope encoding (verified against Hermes ``PubSubRoutes``).** Hermes treats a
message body as an opaque UTF-8 ``payload`` string plus a ``Map<string,string>``
of ``attributes`` — it knows nothing of these types. So the structured message is
JSON-encoded into the ``payload`` string, and ``postId`` is mirrored into
``attributes`` for cheap correlation without decoding the body.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

POST_ID_ATTRIBUTE = "postId"


class Source(str, Enum):
    """Which model produced a suggestion."""

    WD = "wd"
    RAM = "ram"


class Status(str, Enum):
    """Outcome of a tag job."""

    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SampleRef:
    """An Apollo reference to the sample derivative to tag."""

    bucket: str
    object: str


@dataclass(frozen=True, slots=True)
class TagJob:
    """A unit of tagging work. ``post_id`` is the idempotency key."""

    post_id: str
    sample: SampleRef
    media_type: str

    @classmethod
    def from_payload(cls, payload: str) -> TagJob:
        """Decode a ``TagJob`` from the Hermes message ``payload`` (JSON string)."""
        raw = json.loads(payload)
        sample = raw["sample"]
        return cls(
            post_id=str(raw["postId"]),
            sample=SampleRef(bucket=str(sample["bucket"]), object=str(sample["object"])),
            media_type=str(raw["mediaType"]),
        )


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One raw tag suggestion. Surface-form normalized only — meaning-level
    aliasing is Artemis's job, not Argus's."""

    tag: str
    confidence: float
    source: Source
    category: str | None = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "tag": self.tag,
            "confidence": round(self.confidence, 4),
            "source": self.source.value,
        }
        if self.category is not None:
            d["category"] = self.category
        return d


@dataclass(frozen=True, slots=True)
class TagSuggestions:
    """The result Argus publishes for a post."""

    post_id: str
    suggestions: list[Suggestion]
    status: Status = Status.OK
    rating: str | None = None

    @classmethod
    def failed(cls, post_id: str) -> TagSuggestions:
        """A ``failed`` result — no suggestions, does not wedge the queue."""
        return cls(post_id=post_id, suggestions=[], status=Status.FAILED)

    def to_payload(self) -> str:
        """Encode to the Hermes message ``payload`` (JSON string)."""
        body: dict[str, object] = {
            "postId": self.post_id,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "status": self.status.value,
        }
        if self.rating is not None:
            body["rating"] = self.rating
        return json.dumps(body, separators=(",", ":"))

    def attributes(self) -> dict[str, str]:
        """Attributes to publish alongside the payload (``postId`` mirrored for
        cheap filtering without decoding the body)."""
        return {POST_ID_ATTRIBUTE: self.post_id}
