"""Per-class report of the current champion on the held-out F3 test1 region.

    uv run python scripts/per_class_report.py data/f3/data > docs/results/f3_per_class.txt
"""
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

from seismic_mlops import data
from seismic_mlops.model import flatten

mlflow.set_tracking_uri("sqlite:///mlflow.db")
ds = data.load_f3(Path(sys.argv[1]))
model = mlflow.sklearn.load_model("models:/seismic-facies@champion")
print(f"champion on {len(ds.y_test)} test1 patches (train/test volumes are disjoint)")
print(classification_report(ds.y_test, model.predict(flatten(ds.X_test)), zero_division=0, digits=3))
