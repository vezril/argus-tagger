"""Environment configuration (task 0.2).

All knobs come from the environment so the same image runs everywhere; Codex
injects them at deploy. Parsing is fail-fast: a bad value raises at startup with
a clear message rather than misbehaving later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ENV_PREFIX = "ARGUS_"


@dataclass(frozen=True, slots=True)
class Config:
    # Hermes
    hermes_base_url: str
    subscription_id: str
    source_topic: str
    results_topic: str
    pull_max: int
    ack_deadline_seconds: int
    # Apollo
    apollo_endpoint: str
    # Models
    wd_model_path: Path
    wd_tags_path: Path
    ram_model_path: Path
    general_threshold: float
    character_threshold: float
    ram_threshold: float
    num_threads: int
    # Runtime
    health_port: int
    stub_models: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        e = os.environ if env is None else env

        def get(name: str, default: str) -> str:
            return e.get(_ENV_PREFIX + name, default)

        return cls(
            hermes_base_url=get("HERMES_BASE_URL", "http://hermesmq:8080"),
            subscription_id=get("SUBSCRIPTION_ID", "argus.media.tag"),
            source_topic=get("SOURCE_TOPIC", "media.tag"),
            results_topic=get("RESULTS_TOPIC", "media.suggestions"),
            pull_max=_int(get("PULL_MAX", "2"), "PULL_MAX"),
            ack_deadline_seconds=_int(get("ACK_DEADLINE_SECONDS", "60"), "ACK_DEADLINE_SECONDS"),
            apollo_endpoint=get("APOLLO_ENDPOINT", "apollo:9090"),
            wd_model_path=Path(get("WD_MODEL_PATH", "/models/wd-v3/model.onnx")),
            wd_tags_path=Path(get("WD_TAGS_PATH", "/models/wd-v3/selected_tags.csv")),
            ram_model_path=Path(get("RAM_MODEL_PATH", "/models/ram_plus_swin_large_14m.pth")),
            general_threshold=_float(get("GENERAL_THRESHOLD", "0.35"), "GENERAL_THRESHOLD"),
            character_threshold=_float(get("CHARACTER_THRESHOLD", "0.85"), "CHARACTER_THRESHOLD"),
            ram_threshold=_float(get("RAM_THRESHOLD", "0.68"), "RAM_THRESHOLD"),
            num_threads=_int(get("NUM_THREADS", "0"), "NUM_THREADS"),
            health_port=_int(get("HEALTH_PORT", "8081"), "HEALTH_PORT"),
            stub_models=get("STUB_MODELS", "0") not in ("0", "", "false", "False"),
        )


def _int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX}{name} must be an integer, got {value!r}") from exc


def _float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX}{name} must be a float, got {value!r}") from exc
