"""Retriever interface. A retriever is fitted on one split and scores that split's catalog."""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Retriever(ABC):
    name: str = "base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, data) -> "Retriever":
        """Fit on `data.train` only. Must not read `data.test`."""

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """(len(users), len(data.catalog)) float32 scores; higher = better."""

    def attach(self, data) -> "Retriever":
        """Re-bind split data after unpickling (only retrievers that need it override this)."""
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Retriever":
        with path.open("rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
