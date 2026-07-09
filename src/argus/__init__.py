"""Argus — the all-seeing auto-tagging service.

The one non-JVM member of the Codex constellation: it consumes ``TagJob``
messages from HermesMQ, reads the sample derivative from Apollo, runs a
multi-model CPU inference pipeline (wd-tagger v3 + RAM++), and publishes raw
``TagSuggestions`` back for Artemis to alias-merge and Muses to review.

Suggestions-only — never auto-applied. Everything is review-gated.
"""

__version__ = "0.1.0"
