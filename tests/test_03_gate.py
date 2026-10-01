"""Acceptance: gate rejects a worse model (unit + integration through MLflow)."""
import mlflow
import pytest

from seismic_mlops.gate import decide
from seismic_mlops.model import build_pipeline
from seismic_mlops.train import ALIAS, run

BASE = {"accuracy": 0.50, "macro_f1": 0.10}


def m(acc, f1):
    return {"accuracy": acc, "macro_f1": f1}


def test_rejects_when_not_beating_majority_baseline():
    r = decide(m(0.50, 0.30), BASE, champion=None)
    assert not r.promote and "baseline" in " ".join(r.reasons)
    r = decide(m(0.60, 0.10), BASE, champion=None)
    assert not r.promote


def test_first_model_promoted_when_it_beats_baseline():
    assert decide(m(0.70, 0.40), BASE, champion=None).promote


def test_rejects_worse_than_champion_and_ties():
    assert not decide(m(0.70, 0.40), BASE, champion=m(0.80, 0.50)).promote
    assert not decide(m(0.80, 0.50), BASE, champion=m(0.80, 0.50)).promote
    assert decide(m(0.82, 0.55), BASE, champion=m(0.80, 0.50)).promote


def test_integration_worse_challenger_does_not_take_alias(tracking, dataset):
    good = run(dataset, tracking_uri=tracking, model_name="t")
    assert good.promoted, good.reasons
    client = mlflow.MlflowClient(tracking_uri=tracking)
    assert str(client.get_model_version_by_alias("t", ALIAS).version) == good.version

    weak = run(dataset, tracking_uri=tracking, model_name="t",
               pipeline=build_pipeline(n_components=1))
    assert not weak.promoted
    assert weak.metrics["macro_f1"] < good.metrics["macro_f1"]
    # challenger is registered (auditable) but champion alias did not move
    assert str(client.get_model_version_by_alias("t", ALIAS).version) == good.version
    assert int(weak.version) > int(good.version)


def test_integration_baseline_beating_is_enforced(tracking, dataset):
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    useless = Pipeline([("scaler", StandardScaler()), ("pca", __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(2)),
                        ("clf", DummyClassifier(strategy="uniform", random_state=0))])
    res = run(dataset, tracking_uri=tracking, model_name="t2", pipeline=useless)
    assert not res.promoted
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow.MlflowClient(tracking_uri=tracking).get_model_version_by_alias("t2", ALIAS)
