"""Start the API on the champion model, send real fixture patches, check the answers."""
import json
import subprocess
import sys
import time
import urllib.request

from seismic_mlops.data import load_fixture

PORT = 8765
ds = load_fixture()
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "--factory", "seismic_mlops.serve:create_app", "--port", str(PORT)],
    env={**__import__("os").environ, "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"})
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit("API did not come up")
    patches = ds.X_test[:20]
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/predict", data=json.dumps({"patches": patches.tolist()}).encode(),
                                 headers={"Content-Type": "application/json"})
    body = json.load(urllib.request.urlopen(req))
    assert len(body["facies"]) == 20 and all(0 <= f <= 5 for f in body["facies"]), body
    acc = sum(int(a == b) for a, b in zip(body["facies"], ds.y_test[:20])) / 20
    print(f"smoke ok: 20 patches served, agreement with true facies {acc:.2f}")
finally:
    proc.terminate()
    proc.wait(10)
