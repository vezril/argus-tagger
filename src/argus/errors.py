"""Failure taxonomy driving the consumer's non-wedging branch (hermes-consumer spec).

The consumer branches on cause:

* :class:`PermanentFailure` — retrying cannot help (corrupt/undecodable sample).
  → publish ``status: failed`` and **ack** (don't burn the redelivery budget).
* :class:`TransientFailure` — a dependency hiccup (Apollo down, publish failed).
  → **nack** (``modifyAckDeadline → 0``) so Hermes redelivers and Argus self-heals.
"""

from __future__ import annotations


class ArgusError(Exception):
    """Base class for Argus processing failures."""


class PermanentFailure(ArgusError):
    """The job can never succeed as-is; ack it with a failed result."""


class TransientFailure(ArgusError):
    """A recoverable dependency failure; nack for redelivery."""
