"""Model-family choice on VALIDATION only (a spatially disjoint block of the train volume). test1 is never read.

    uv run python scripts/model_family_comparison.py > docs/results/model_family_comparison.log
"""
import time
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seismic_mlops import data, gate
from seismic_mlops.model import flatten


def mk(clf, n=32):
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n, random_state=0)), ("clf", clf)])


models = {
    "LR C=1": lambda: mk(LogisticRegression(max_iter=2000)),
    "LR C=.01": lambda: mk(LogisticRegression(C=.01, max_iter=2000)),
    "RF300": lambda: mk(RandomForestClassifier(300, n_jobs=-1, random_state=0)),
    "HGB": lambda: mk(HistGradientBoostingClassifier(random_state=0)),
}
for tag, ds in [("fixture (600 train / 300 val)", data.load_fixture()),
                ("full benchmark sample (6000 train / 3000 val)", data.load_f3(Path("data/f3/data"), 6000, 3000, 10, 0))]:
    base = gate.majority_baseline(ds.y_train, ds.y_val)
    print(f"== {tag}: majority baseline on validation {base}")
    for name, f in models.items():
        t = time.time()
        p = f().fit(flatten(ds.X_train), ds.y_train)
        s = gate.score(ds.y_val, p.predict(flatten(ds.X_val)))
        print(f"  {name:9s} val acc {s['accuracy']:.3f} macro-F1 {s['macro_f1']:.3f}  ({time.time()-t:.0f}s)")
