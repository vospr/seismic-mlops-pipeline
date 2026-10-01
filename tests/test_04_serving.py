"""Acceptance: serving features == training features (parity), by construction and by value."""
import inspect

import numpy as np
from fastapi.testclient import TestClient

from seismic_mlops import serve
from seismic_mlops.train import ALIAS, run


def test_served_predictions_equal_training_pipeline(tracking, dataset):
    res = run(dataset, tracking_uri=tracking, model_name="srv")
    assert res.promoted
    app = serve.create_app(model_uri=f"models:/srv@{ALIAS}", tracking_uri=tracking)
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        raw = dataset.X_test[:40]
        r = c.post("/predict", json={"patches": raw.tolist()})
        assert r.status_code == 200
        body = r.json()
        flat = raw.reshape(len(raw), -1)
        np.testing.assert_array_equal(body["facies"], res.pipeline.predict(flat))
        np.testing.assert_allclose(body["proba"], res.pipeline.predict_proba(flat), atol=1e-6)


def test_serving_rejects_wrong_patch_shape(tracking, dataset):
    run(dataset, tracking_uri=tracking, model_name="srv2")
    app = serve.create_app(model_uri=f"models:/srv2@{ALIAS}", tracking_uri=tracking)
    with TestClient(app) as c:
        assert c.post("/predict", json={"patches": [[[0.0] * 3] * 3]}).status_code == 422


def test_serving_does_not_reimplement_features():
    src = inspect.getsource(serve)
    for banned in ("PCA", "StandardScaler", ".mean(", ".std(", "explained_variance"):
        assert banned not in src
