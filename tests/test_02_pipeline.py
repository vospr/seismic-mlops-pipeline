"""Acceptance: one sklearn Pipeline, fitted on train only."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seismic_mlops.model import build_pipeline, flatten


def test_is_single_pipeline_scaler_pca_model():
    p = build_pipeline()
    assert isinstance(p, Pipeline)
    names = [n for n, _ in p.steps]
    assert names == ["scaler", "pca", "clf"]
    assert isinstance(p.named_steps["scaler"], StandardScaler)
    assert isinstance(p.named_steps["pca"], PCA)


def test_fit_on_train_only_no_leakage(dataset):
    p = build_pipeline().fit(flatten(dataset.X_train), dataset.y_train)
    scaler = p.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, flatten(dataset.X_train).mean(axis=0), rtol=1e-5, atol=1e-6)
    everything = np.vstack([flatten(dataset.X_train), flatten(dataset.X_test)])
    assert not np.allclose(scaler.mean_, everything.mean(axis=0), rtol=1e-7, atol=0)


def test_accepts_raw_patches_end_to_end(dataset):
    p = build_pipeline().fit(flatten(dataset.X_train), dataset.y_train)
    assert p.predict(flatten(dataset.X_test)).shape == (len(dataset.X_test),)
