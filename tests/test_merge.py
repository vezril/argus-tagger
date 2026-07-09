from argus.contract import Source
from argus.merge import merge
from argus.models.base import Prediction


def test_cross_model_agreement_keeps_max_confidence() -> None:
    merged = merge(
        [
            (Source.WD, Prediction(tag="tree", confidence=0.7)),
            (Source.RAM, Prediction(tag="tree", confidence=0.9)),
        ]
    )
    assert len(merged) == 1
    assert merged[0].tag == "tree"
    assert merged[0].confidence == 0.9
    assert merged[0].source is Source.RAM  # the winning model


def test_disjoint_tags_both_pass_through() -> None:
    merged = merge(
        [
            (Source.WD, Prediction(tag="1girl", confidence=0.98, category="general")),
            (Source.RAM, Prediction(tag="beach", confidence=0.6)),
        ]
    )
    tags = {s.tag for s in merged}
    assert tags == {"1girl", "beach"}


def test_normalizes_before_merging() -> None:
    # "teddy bear" (ram) and "teddy_bear" (wd) are the SAME normalized tag.
    merged = merge(
        [
            (Source.WD, Prediction(tag="teddy_bear", confidence=0.5)),
            (Source.RAM, Prediction(tag="teddy bear", confidence=0.8)),
        ]
    )
    assert len(merged) == 1
    assert merged[0].tag == "teddy_bear"
    assert merged[0].confidence == 0.8


def test_output_sorted_by_confidence_desc() -> None:
    merged = merge(
        [
            (Source.WD, Prediction(tag="a", confidence=0.4)),
            (Source.WD, Prediction(tag="b", confidence=0.9)),
            (Source.RAM, Prediction(tag="c", confidence=0.6)),
        ]
    )
    assert [s.tag for s in merged] == ["b", "c", "a"]


def test_blank_tags_dropped() -> None:
    assert merge([(Source.RAM, Prediction(tag="   ", confidence=0.9))]) == []
