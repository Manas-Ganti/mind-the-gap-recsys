"""LightGBM lambdarank over stage-1 candidates. Trained on temporal_rtrain candidates, early
stopped on the later temporal_val day, then refit on temporal_val for the chosen number of
rounds (no validation block => exactly `num_boost_round`). Never sees temporal test.

LightGBM runs in a separate worker process (`python -m recsys.ranking.gbdt fit|predict`).
On macOS arm64, torch and LightGBM each ship their own libomp; with both loaded into one
process, whichever initialises second segfaults (exit 139), regardless of import order.
The worker never imports torch. Arrays cross the boundary as .npz files and the model is
stored in LightGBM's own text format.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from recsys.ranking.features import CandidateBlock


def _save_block(block: CandidateBlock, path: Path) -> None:
    np.savez(path, ptr=block.ptr, X=block.X, y=block.y)


def _run(*args: str) -> None:
    r = subprocess.run([sys.executable, "-m", "recsys.ranking.gbdt", *args])
    if r.returncode:
        raise RuntimeError(f"LightGBM worker failed ({r.returncode}): {args[0]}")


class GBDTRanker:
    name = "gbdt"
    columns: list[int] | None = None      # feature subset; class default keeps old pickles loadable

    def __init__(self, params: dict, num_boost_round: int, early_stopping_rounds: int, seed: int,
                 columns: list[int] | None = None):
        self.params = {**params, "seed": seed, "deterministic": True, "force_row_wise": True}
        self.num_boost_round, self.early_stopping_rounds = num_boost_round, early_stopping_rounds
        self.columns = columns

    def fit(self, train: CandidateBlock, valid: CandidateBlock | None, feature_names: list[str]) -> "GBDTRanker":
        if self.columns is not None:
            feature_names = [feature_names[i] for i in self.columns]
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            _save_block(train.with_columns(self.columns), t / "train.npz")
            if valid is not None:
                _save_block(valid.with_columns(self.columns), t / "valid.npz")
            (t / "spec.json").write_text(json.dumps({
                "params": self.params, "num_boost_round": self.num_boost_round,
                "early_stopping_rounds": self.early_stopping_rounds, "feature_names": feature_names}))
            _run("fit", str(t))
            self.model_str_ = (t / "model.txt").read_text()
            info = json.loads((t / "info.json").read_text())
        self.best_iteration_ = info["best_iteration"]
        self.feature_importance_ = info["feature_importance_gain"]
        return self

    def predict(self, block: CandidateBlock) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            _save_block(block.with_columns(self.columns), t / "block.npz")
            (t / "model.txt").write_text(self.model_str_)
            (t / "spec.json").write_text(json.dumps({"best_iteration": self.best_iteration_}))
            _run("predict", str(t))
            return np.load(t / "scores.npy")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "GBDTRanker":
        return pickle.loads(path.read_bytes())


def _worker(cmd: str, tmp: Path) -> None:
    import lightgbm as lgb

    spec = json.loads((tmp / "spec.json").read_text())
    if cmd == "fit":
        tr = np.load(tmp / "train.npz")
        names = spec["feature_names"]
        dtr = lgb.Dataset(tr["X"], tr["y"], group=np.diff(tr["ptr"]), feature_name=names)
        if (tmp / "valid.npz").exists():
            va = np.load(tmp / "valid.npz")
            dva = lgb.Dataset(va["X"], va["y"], group=np.diff(va["ptr"]), reference=dtr)
            booster = lgb.train(spec["params"], dtr, num_boost_round=spec["num_boost_round"], valid_sets=[dva],
                                valid_names=["valid"],
                                callbacks=[lgb.early_stopping(spec["early_stopping_rounds"], verbose=False),
                                           lgb.log_evaluation(0)])
            best_iteration = int(booster.best_iteration)
        else:
            booster = lgb.train(spec["params"], dtr, num_boost_round=spec["num_boost_round"])
            best_iteration = spec["num_boost_round"]
        booster.save_model(str(tmp / "model.txt"))
        imp = booster.feature_importance("gain")
        (tmp / "info.json").write_text(json.dumps({
            "best_iteration": best_iteration,
            "feature_importance_gain": dict(sorted(zip(names, map(float, imp)), key=lambda x: -x[1]))}))
    elif cmd == "predict":
        booster = lgb.Booster(model_file=str(tmp / "model.txt"))
        X = np.load(tmp / "block.npz")["X"]
        np.save(tmp / "scores.npy", booster.predict(X, num_iteration=spec["best_iteration"] or None))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    _worker(sys.argv[1], Path(sys.argv[2]))
