"""Fit on train only, score on held-out region, log the Pipeline to MLflow, promote through the gate."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn

from . import data, gate
from .model import build_pipeline, flatten

ALIAS = "champion"
# skops refuses unknown types on load. This is the one type HistGradientBoosting needs; we load only
# artifacts this repo wrote itself, so we trust exactly it (not get_untrusted_types() wholesale).
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
MODEL_NAME = "seismic-facies"


@dataclass
class RunResult:
    version: str
    promoted: bool
    reasons: list
    metrics: dict
    baseline: dict
    champion: dict | None
    pipeline: object


def _champion_metrics(client, name, X, y):
    """Re-score the current champion on *this* test set so the comparison is like for like."""
    try:
        mv = client.get_model_version_by_alias(name, ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    model = mlflow.sklearn.load_model(f"models:/{name}@{ALIAS}")
    return gate.score(y, model.predict(flatten(X)))


def run(ds: data.Dataset, tracking_uri: str, model_name: str = MODEL_NAME, pipeline=None) -> RunResult:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("seismic-facies")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    pipeline = pipeline if pipeline is not None else build_pipeline()

    Xtr, Xte = flatten(ds.X_train), flatten(ds.X_test)
    pipeline.fit(Xtr, ds.y_train)  # scaler, PCA and classifier all see train only
    metrics = gate.score(ds.y_test, pipeline.predict(Xte))
    baseline = gate.majority_baseline(ds.y_train, ds.y_test)
    champion = _champion_metrics(client, model_name, ds.X_test, ds.y_test)
    verdict = gate.decide(metrics, baseline, champion)

    with mlflow.start_run() as run_:
        mlflow.log_params({"steps": ",".join(n for n, _ in pipeline.steps), "n_train": len(Xtr),
                           "n_test": len(Xte), **{k: v for k, v in ds.meta.items() if k != "n_train"}})
        mlflow.log_metrics({**metrics, **{f"baseline_{k}": v for k, v in baseline.items()}})
        if champion:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion.items()})
        mlflow.set_tags({"gate.promote": str(verdict.promote), "gate.reasons": " | ".join(verdict.reasons)})
        info = mlflow.sklearn.log_model(pipeline, name="model", registered_model_name=model_name,
                                        input_example=Xtr[:2],
                                        skops_trusted_types=TRUSTED_TYPES)
        version = str(info.registered_model_version)

    if verdict.promote:
        client.set_registered_model_alias(model_name, ALIAS, version)
    return RunResult(version, verdict.promote, verdict.reasons, metrics, baseline, champion, pipeline)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fixture", help="'fixture' or path to the unzipped F3 benchmark 'data/' dir")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--n-components", type=int, default=32)
    ap.add_argument("--fail-if-not-promoted", action="store_true")
    a = ap.parse_args(argv)
    ds = data.load_fixture() if a.data == "fixture" else data.load_f3(Path(a.data))
    res = run(ds, a.tracking_uri, a.model_name, build_pipeline(n_components=a.n_components))
    print(json.dumps({"version": res.version, "promoted": res.promoted, "reasons": res.reasons,
                      "metrics": res.metrics, "baseline": res.baseline, "champion": res.champion}, indent=2))
    if a.fail_if_not_promoted and not res.promoted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
