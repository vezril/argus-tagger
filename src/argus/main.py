"""Entrypoint — wire config → clients → pipeline → consumer, then run (task 0.3 + 5.x).

``argus`` console script. Builds the object graph from :class:`~argus.config.Config`,
warms the models, starts the health server, and runs the consume loop forever.

With ``ARGUS_STUB_MODELS=1`` it wires weightless stubs (a stub sample reader +
stub taggers) so the full consume → tag → publish path runs against a real Hermes
with no ML stack — the local/e2e path behind task 5.2.
"""

from __future__ import annotations

import logging

from argus.apollo import GrpcSampleReader, SampleReader, StubSampleReader
from argus.config import Config
from argus.consumer import Consumer
from argus.contract import Source
from argus.hermes import HermesClient
from argus.models.base import StubTagger, Tagger
from argus.pipeline import TaggingPipeline


def build_pipeline(config: Config) -> TaggingPipeline:
    """Assemble the tagging pipeline per the per-model runtime decision."""
    if config.stub_models:
        taggers: list[Tagger] = [StubTagger(Source.WD), StubTagger(Source.RAM)]
        return TaggingPipeline(taggers)

    # Heavy imports deferred so the stub path needs no ML stack.
    from argus.models.ram_plus import RamPlusTagger
    from argus.models.wd_tagger import WdTagger

    wd = WdTagger(
        config.wd_model_path,
        config.wd_tags_path,
        general_threshold=config.general_threshold,
        character_threshold=config.character_threshold,
        num_threads=config.num_threads,
    )
    ram = RamPlusTagger(
        config.ram_model_path,
        threshold=config.ram_threshold,
        num_threads=config.num_threads,
    )
    return TaggingPipeline([wd, ram])


def build_reader(config: Config) -> SampleReader:
    return StubSampleReader() if config.stub_models else GrpcSampleReader(config.apollo_endpoint)


def main() -> None:  # pragma: no cover - process wiring
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = Config.from_env()

    pipeline = build_pipeline(config)
    reader = build_reader(config)
    hermes = HermesClient(config.hermes_base_url)

    from argus.health import serve_in_background

    serve_in_background(config.health_port, lambda: pipeline.ready)
    logging.getLogger("argus").info("loading models (stub=%s)…", config.stub_models)
    pipeline.load()

    consumer = Consumer(
        hermes=hermes,
        reader=reader,
        pipeline=pipeline,
        subscription_id=config.subscription_id,
        source_topic=config.source_topic,
        results_topic=config.results_topic,
        pull_max=config.pull_max,
    )
    consumer.run_forever()


if __name__ == "__main__":
    main()
