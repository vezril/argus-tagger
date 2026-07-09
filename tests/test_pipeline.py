import pytest
from PIL import Image

from argus.contract import Source
from argus.models.base import Prediction, StubTagger
from argus.pipeline import TaggingPipeline


def _image() -> Image.Image:
    return Image.new("RGB", (16, 16), (10, 20, 30))


def test_requires_at_least_one_tagger() -> None:
    with pytest.raises(ValueError):
        TaggingPipeline([])


def test_ready_only_after_all_models_loaded() -> None:
    wd, ram = StubTagger(Source.WD), StubTagger(Source.RAM)
    pipeline = TaggingPipeline([wd, ram])
    assert not pipeline.ready
    wd.load()
    assert not pipeline.ready  # ram still cold
    ram.load()
    assert pipeline.ready


def test_tag_runs_all_models_and_merges() -> None:
    wd = StubTagger(Source.WD, [Prediction(tag="1girl", confidence=0.9, category="general")])
    ram = StubTagger(Source.RAM, [Prediction(tag="beach", confidence=0.7)])
    pipeline = TaggingPipeline([wd, ram])
    pipeline.load()

    suggestions = pipeline.tag(_image())
    by_tag = {s.tag: s for s in suggestions}
    assert set(by_tag) == {"1girl", "beach"}
    assert by_tag["1girl"].source is Source.WD
    assert by_tag["beach"].source is Source.RAM


def test_agreement_across_models_collapses_to_one() -> None:
    wd = StubTagger(Source.WD, [Prediction(tag="tree", confidence=0.6)])
    ram = StubTagger(Source.RAM, [Prediction(tag="tree", confidence=0.85)])
    pipeline = TaggingPipeline([wd, ram])
    pipeline.load()

    suggestions = pipeline.tag(_image())
    assert len(suggestions) == 1
    assert suggestions[0].confidence == 0.85
