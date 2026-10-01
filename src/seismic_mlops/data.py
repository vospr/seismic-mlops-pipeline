"""F3 facies benchmark (Alaudah et al. 2019): download check, patch sampling, committed fixture."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

F3_URL = "https://zenodo.org/record/3755060/files/data.zip"
F3_MD5 = "bc5932279831a95c0b244fd765376d85"  # published in the benchmark's README
PATCH = 32
N_CLASSES = 6
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "f3_patches.npz"


@dataclass
class Dataset:
    X_train: np.ndarray  # (n, PATCH, PATCH) raw amplitudes
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict


def md5_matches(path: Path, expected: str = F3_MD5) -> bool:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def make_patches(volume, labels, size: int = PATCH, n: int = 1000, seed: int = 0):
    """Sample n square patches from vertical sections (volume[i] is a 2D section).

    The label of a patch is the facies of its center pixel.
    """
    rng = np.random.default_rng(seed)
    a, h, w = volume.shape
    ii = rng.integers(0, a, n)
    rr = rng.integers(0, h - size + 1, n)
    cc = rng.integers(0, w - size + 1, n)
    X = np.empty((n, size, size), dtype="float32")
    y = np.empty(n, dtype="int64")
    for k, (i, r, c) in enumerate(zip(ii, rr, cc)):
        X[k] = volume[i, r:r + size, c:c + size]
        y[k] = labels[i, r + size // 2, c + size // 2]
    return X, y


def load_fixture(path: Path = FIXTURE) -> Dataset:
    z = np.load(path, allow_pickle=False)
    return Dataset(
        X_train=z["X_train"].astype("float32"), y_train=z["y_train"].astype("int64"),
        X_test=z["X_test"].astype("float32"), y_test=z["y_test"].astype("int64"),
        meta=json.loads(str(z["meta"])),
    )


def load_f3(root: Path, n_train: int = 20000, n_test: int = 5000, seed: int = 0) -> Dataset:
    """Full benchmark: train on the train volume, evaluate on the disjoint test1 volume."""
    root = Path(root)
    tr = make_patches(np.load(root / "train" / "train_seismic.npy", mmap_mode="r"),
                      np.load(root / "train" / "train_labels.npy", mmap_mode="r"), n=n_train, seed=seed)
    te = make_patches(np.load(root / "test_once" / "test1_seismic.npy", mmap_mode="r"),
                      np.load(root / "test_once" / "test1_labels.npy", mmap_mode="r"), n=n_test, seed=seed + 1)
    return Dataset(tr[0], tr[1], te[0], te[1], _meta(seed, n_train, n_test))


def _meta(seed, n_train, n_test) -> dict:
    return {"source_url": F3_URL, "source_md5": F3_MD5, "train_region": "train", "test_region": "test1",
            "patch": PATCH, "seed": seed, "n_train": n_train, "n_test": n_test}


def build_fixture(root: Path, n_train: int = 600, n_test: int = 300, seed: int = 0) -> Path:
    """Reproducible tiny subset of the real benchmark, committed so CI needs no download."""
    ds = load_f3(root, n_train, n_test, seed)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, X_train=ds.X_train.astype("float16"), y_train=ds.y_train.astype("uint8"),
                        X_test=ds.X_test.astype("float16"), y_test=ds.y_test.astype("uint8"),
                        meta=json.dumps(ds.meta))
    return FIXTURE


if __name__ == "__main__":  # python -m seismic_mlops.data fixture <data dir>
    cmd, root = sys.argv[1], Path(sys.argv[2])
    if cmd == "fixture":
        print(build_fixture(root))
