"""Acceptance: no model selection on test1. Validation is a spatially disjoint block of the train volume;
the gate and model-family choice use validation only; test1 is scored once per run and only reported."""
import inspect
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from seismic_mlops import data
from seismic_mlops.data import Dataset, block_split
from seismic_mlops.train import run

ROOT = Path(__file__).resolve().parents[1]


def test_block_split_is_contiguous_disjoint_with_gap():
    train, val = block_split(401)
    assert train.start == 0 and val.stop == 401
    assert val.start - train.stop >= 10          # buffer so neighbouring sections never straddle the split
    assert len(val) >= 80 and len(train) >= 250
    assert not set(train) & set(val)


def test_load_f3_train_and_val_patches_come_from_disjoint_section_blocks(tmp_path):
    n, h, w = 40, 64, 64
    for sub, name, count in (("train", "train", n), ("test_once", "test1", 12)):
        d = tmp_path / sub
        d.mkdir()
        # every pixel of section i holds the value i, so a patch reveals which section it came from
        vol = np.broadcast_to(np.arange(count, dtype="float64")[:, None, None], (count, h, w)).copy()
        np.save(d / f"{name}_seismic.npy", vol)
        np.save(d / f"{name}_labels.npy", np.random.default_rng(0).integers(0, 6, (count, h, w)).astype("uint8"))
    ds = data.load_f3(tmp_path, n_train=300, n_val=200, n_test=100)
    train, val = block_split(n)
    tr_sections, va_sections = set(np.unique(ds.X_train)), set(np.unique(ds.X_val))
    assert tr_sections <= set(float(i) for i in train)
    assert va_sections <= set(float(i) for i in val)
    assert not tr_sections & va_sections
    assert len(ds.X_val) == 200 and len(ds.y_val) == 200


def test_fixture_has_validation_split_with_provenance(dataset):
    assert dataset.X_val.shape[1:] == (data.PATCH, data.PATCH) and len(dataset.X_val) >= 150
    assert len(np.unique(dataset.y_val)) >= 3
    assert dataset.meta["val_region"] not in (dataset.meta["train_region"], dataset.meta["test_region"])


def _with_test_labels(ds, y_test):
    return Dataset(ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, y_test, ds.meta)


def test_gate_decision_does_not_depend_on_test1(tmp_path, monkeypatch, dataset):
    monkeypatch.chdir(tmp_path)
    scrambled = _with_test_labels(dataset, np.random.default_rng(1).permutation(dataset.y_test))
    a = run(dataset, tracking_uri=f"sqlite:///{tmp_path}/a.db", model_name="a")
    b = run(scrambled, tracking_uri=f"sqlite:///{tmp_path}/b.db", model_name="b")
    assert a.promoted and b.promoted and a.reasons == b.reasons
    assert a.metrics == b.metrics and a.baseline == b.baseline          # decision inputs identical (validation)
    assert a.test_metrics != b.test_metrics                              # test1 numbers are reported, and differ


def test_test1_scored_exactly_once_per_run(tracking, dataset, monkeypatch):
    ds = Dataset(dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val,
                 dataset.X_test[:250], dataset.y_test[:250], dataset.meta)   # 250 rows: distinguishable from val (300)
    calls = []
    orig = Pipeline.predict

    def spy(self, X, **kw):
        calls.append(len(X))
        return orig(self, X, **kw)

    monkeypatch.setattr(Pipeline, "predict", spy)
    first = run(ds, tracking_uri=tracking, model_name="once")
    assert first.promoted and calls.count(250) == 1
    calls.clear()
    run(ds, tracking_uri=tracking, model_name="once")      # now a champion exists and is re-scored: on validation only
    assert calls.count(250) == 1 and calls.count(len(ds.X_val)) == 2   # challenger + champion on val; test1 once


def test_model_family_comparison_script_never_reads_test1():
    src = (ROOT / "scripts" / "model_family_comparison.py").read_text()
    assert "X_test" not in src and "y_test" not in src
