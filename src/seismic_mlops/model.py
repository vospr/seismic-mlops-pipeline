"""The one model object: scaler + PCA + classifier in a single sklearn Pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def flatten(patches: np.ndarray) -> np.ndarray:
    """(n, h, w) -> (n, h*w). The only transformation outside the Pipeline."""
    return np.asarray(patches, dtype="float32").reshape(len(patches), -1)


def build_pipeline(n_components: int = 32, seed: int = 0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("clf", HistGradientBoostingClassifier(random_state=seed)),
    ])
