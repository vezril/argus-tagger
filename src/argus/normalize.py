"""Surface-form normalization (tagging-service spec, task 2.1).

Argus normalizes only the *surface form* of a tag — lowercased, whitespace
collapsed to single underscores — so both vocabularies (wd-tagger's Danbooru
underscores and RAM++'s natural-language phrases) feed one consistent form.

It does NOT do meaning-level merging: ``outdoor`` is not aliased to ``outdoors``.
That canonicalization is Artemis's alias/implication system downstream.
"""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def normalize_surface_form(tag: str) -> str:
    """Lowercase, trim, and collapse whitespace/underscore runs to one underscore.

    >>> normalize_surface_form("teddy bear")
    'teddy_bear'
    >>> normalize_surface_form("1girl")
    '1girl'
    >>> normalize_surface_form("  Long   Hair ")
    'long_hair'
    """
    lowered = tag.strip().lower()
    underscored = _WHITESPACE.sub("_", lowered)
    collapsed = _MULTI_UNDERSCORE.sub("_", underscored)
    return collapsed.strip("_")
