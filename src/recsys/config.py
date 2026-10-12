"""Typed config. Everything tunable lives in configs/*.yaml; nothing is hardcoded in src/."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Paths(BaseModel):
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    results_dir: Path
    figures_dir: Path


class DataCfg(BaseModel):
    variant: Literal["large", "small"] = "large"
    raw_splits: list[str] = ["train", "dev"]


class SparkCfg(BaseModel):
    master: str = "local[*]"
    driver_memory: str = "8g"
    shuffle_partitions: int = 64
    extra: dict[str, str] = Field(default_factory=dict)


class SplitCfg(BaseModel):
    temporal_cutoff: datetime
    validation_cutoff: datetime
    random_test_fraction: float | None = None
    exclude_train_seen: bool = True
    catalog: Literal["live", "all"] = "live"
    # snapshot: MIND's `history` is a fixed pre-window snapshot (verified on MIND-small: one
    #   version per user across all days, 95% of its items never appear in the log), so it is
    #   known at serving time for every user. The split asserts the one-version property.
    # visible_before_cutoff: only use history attached to impressions before the cutoff.
    history_policy: Literal["snapshot", "visible_before_cutoff"] = "snapshot"
    max_eval_users: int = 20000
    max_val_users: int = 50000

    @model_validator(mode="after")
    def _ordered(self) -> "SplitCfg":
        if self.validation_cutoff >= self.temporal_cutoff:
            raise ValueError("validation_cutoff must be strictly before temporal_cutoff")
        return self

    @staticmethod
    def epoch(dt: datetime) -> int:
        """Naive datetimes are interpreted as UTC, matching the Spark session timezone."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())


class EvalCfg(BaseModel):
    ks: list[int] = [5, 10, 20, 50, 100]
    headline_k: int = 10
    n_negatives: int = 100
    negative_oversample: float = 2.0
    user_batch: int = 512
    tie_noise: float = 1e-9
    popularity_gamma: float = 0.5
    bootstrap: int = 1000

    @model_validator(mode="after")
    def _headline_in_ks(self) -> "EvalCfg":
        if self.headline_k not in self.ks:
            raise ValueError("headline_k must be one of ks")
        return self


class PopularCfg(BaseModel):
    window_hours: list[float | None]


class ItemKNNCfg(BaseModel):
    neighbors: list[int]
    shrink: list[float]
    alpha: list[float]


class GraphCfg(BaseModel):
    neighbors: list[int]
    alpha: list[float]
    beta: list[float]


class TwoTowerCfg(BaseModel):
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"   # frozen (CLAUDE.md decision 1)
    encode_batch_size: int = 256
    dim: int = 64
    history_len: int = 30
    temperature: list[float] = [0.05]
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 3
    max_steps_per_epoch: int = 3000
    logq_correction: bool = True
    device: str = "auto"


class RetrievalCfg(BaseModel):
    candidates_k: int = 200
    tuning_metric: str = "ndcg@10"
    tuning_users: int = 5000
    popular: PopularCfg
    itemknn: ItemKNNCfg
    graph: GraphCfg
    two_tower: TwoTowerCfg


class GBDTCfg(BaseModel):
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    params: dict[str, Any] = Field(default_factory=dict)


class NeuralRankerCfg(BaseModel):
    hidden: list[int] = [128, 64]
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 20
    batch_users: int = 128
    patience: int = 3
    device: str = "auto"


class RankingCfg(BaseModel):
    max_train_users: int = 50000
    valid_fraction: float = 0.2
    gbdt: GBDTCfg
    neural: NeuralRankerCfg


class ServeCfg(BaseModel):
    k: int = 100
    n_queries: int = 5000
    n_latency_queries: int = 1000
    hnsw_m: int = 32
    ef_search: list[int] = [16, 32, 64, 128, 256]
    ivf_nlist: int = 256
    nprobe: list[int] = [1, 4, 16, 64]


class Config(BaseModel):
    seed: int
    paths: Paths
    data: DataCfg
    spark: SparkCfg
    split: SplitCfg
    eval: EvalCfg
    retrieval: RetrievalCfg
    ranking: RankingCfg
    serve: ServeCfg

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def ensure_dirs(self) -> None:
        for p in (self.paths.processed_dir, self.paths.models_dir,
                  self.paths.results_dir, self.paths.figures_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_raw(path: Path) -> dict:
    """YAML with optional `extends: <other.yaml>` (relative to this file). Lets a cluster
    config change only machine settings (Spark memory, cores) while inheriting every protocol
    setting, so the protocol cannot drift between laptop and cluster runs."""
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    return _deep_merge(_load_raw((path.parent / parent).resolve()), raw) if parent else raw


def load_config(path: str | Path) -> Config:
    path = Path(path)
    cfg = Config.model_validate(_load_raw(path))
    # Relative paths are relative to the repo root (the config's parent's parent).
    root = path.resolve().parent.parent
    for name in Paths.model_fields:
        p = getattr(cfg.paths, name)
        if not p.is_absolute():
            setattr(cfg.paths, name, root / p)
    return cfg


@contextmanager
def deterministic_torch():
    """Bit-reproducible torch training. Without this, multi-threaded CPU backward passes that
    accumulate into shared rows (embedding gathers, index_add) sum in a nondeterministic order:
    two identical MIND-small two-tower fits differed by 0.43 in score and ~25% in val NDCG.
    Measured cost of the deterministic kernels on CPU: none. Restores the previous setting."""
    import torch

    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def device_for(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
