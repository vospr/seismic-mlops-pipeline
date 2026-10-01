"""Why HistGradientBoosting: compare model families behind the same scaler+PCA, on the fixture and a 6k sample.

    uv run python scripts/model_family_comparison.py > docs/results/model_family_comparison.log

Run after seeing test1 numbers; see README caveats.
"""
import numpy as np, time
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from seismic_mlops import data, gate
from seismic_mlops.model import flatten
fx = data.load_fixture()
full = data.load_f3(Path("data/f3/data"), 6000, 3000, 0)
def mk(clf, n=32): return Pipeline([("scaler",StandardScaler()),("pca",PCA(n,random_state=0)),("clf",clf)])
models = {
 "LR C=1": lambda: mk(LogisticRegression(max_iter=2000)),
 "LR C=.01": lambda: mk(LogisticRegression(C=.01,max_iter=2000)),
 "RF300": lambda: mk(RandomForestClassifier(300,n_jobs=-1,random_state=0)),
 "HGB": lambda: mk(HistGradientBoostingClassifier(random_state=0)),
}
for tag, ds in [("fixture 600", fx), ("full 6000", full)]:
    base = gate.majority_baseline(ds.y_train, ds.y_test)
    print(f"== {tag}: majority baseline {base}")
    for name, f in models.items():
        t=time.time(); p=f().fit(flatten(ds.X_train), ds.y_train)
        s=gate.score(ds.y_test, p.predict(flatten(ds.X_test)))
        print(f"  {name:9s} acc {s['accuracy']:.3f} f1 {s['macro_f1']:.3f}  ({time.time()-t:.0f}s)")
