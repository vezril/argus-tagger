from argus.normalize import normalize_surface_form


def test_spaces_become_single_underscore() -> None:
    assert normalize_surface_form("teddy bear") == "teddy_bear"


def test_already_danbooru_form_is_stable() -> None:
    assert normalize_surface_form("1girl") == "1girl"
    assert normalize_surface_form("long_hair") == "long_hair"


def test_lowercases_and_trims() -> None:
    assert normalize_surface_form("  Long   Hair ") == "long_hair"


def test_collapses_underscore_runs_and_strips() -> None:
    assert normalize_surface_form("blue__eyes_") == "blue_eyes"


def test_surface_form_only_no_aliasing() -> None:
    # "outdoor" is NOT rewritten to "outdoors" — that's Artemis's job.
    assert normalize_surface_form("outdoor") == "outdoor"
