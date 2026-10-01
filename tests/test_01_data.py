"""Acceptance: real facies labels, documented download, honest split."""
import re
from pathlib import Path

import numpy as np

from seismic_mlops import data

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_is_committed_and_has_real_facies_classes(dataset):
    assert data.FIXTURE.exists()
    assert dataset.X_train.shape[1:] == (data.PATCH, data.PATCH)
    assert set(np.unique(dataset.y_train)) <= set(range(data.N_CLASSES))
    # real benchmark has 6 facies; fixture must contain several of them
    assert len(np.unique(dataset.y_train)) >= 4
    assert len(dataset.X_train) >= 300 and len(dataset.X_test) >= 150


def test_fixture_declares_provenance(dataset):
    assert dataset.meta["source_md5"] == data.F3_MD5
    assert dataset.meta["source_url"] == data.F3_URL
    assert dataset.meta["test_region"] != dataset.meta["train_region"]


def test_readme_documents_download_and_checksum():
    readme = (ROOT / "README.md").read_text()
    assert data.F3_URL in readme
    assert data.F3_MD5 in readme
    assert "Alaudah" in readme


def test_md5_check_rejects_corrupt_file(tmp_path):
    bad = tmp_path / "data.zip"
    bad.write_bytes(b"not the benchmark")
    assert data.md5_matches(bad) is False


def test_patches_use_center_pixel_label_and_are_in_bounds():
    vol = np.arange(2 * 40 * 40, dtype="float32").reshape(2, 40, 40)
    lab = (np.arange(2 * 40 * 40).reshape(2, 40, 40) % 6).astype("uint8")
    X, y = data.make_patches(vol, lab, size=8, n=50, seed=1)
    assert X.shape == (50, 8, 8) and y.shape == (50,)
    for patch, label in zip(X, y):
        # center pixel of patch carries the label we were given
        idx = np.argwhere(vol == patch[4, 4])
        assert lab[tuple(idx[0])] == label
