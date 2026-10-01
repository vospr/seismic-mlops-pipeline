"""Acceptance: CI can fail; scope cut is real; clone is reproducible."""
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CUT = ("feast", "faiss")


def test_ci_workflow_cannot_silently_pass():
    wf = ROOT / ".github" / "workflows"
    files = list(wf.glob("*.y*ml"))
    assert files, "no workflow"
    for f in files:
        text = f.read_text()
        assert "continue-on-error" not in text, f
        assert "|| true" not in text, f
    ci = yaml.safe_load((wf / "ci.yml").read_text())
    triggers = ci.get("on") or ci.get(True)
    assert "pull_request" in triggers and "push" in triggers
    steps = " ".join(s.get("run", "") for j in ci["jobs"].values() for s in j["steps"])
    assert "make ci" in steps


def test_make_ci_runs_pipeline_and_pytest():
    mk = (ROOT / "Makefile").read_text()
    assert re.search(r"^ci:.*", mk, re.M) and "pytest" in mk and "seismic_mlops.train" in mk
    assert re.search(r"^demo:", mk, re.M)


def test_dependencies_are_pyproject_plus_lockfile():
    assert (ROOT / "pyproject.toml").exists()
    assert (ROOT / "uv.lock").exists()
    assert not (ROOT / "requirements.txt").exists()


def test_feast_faiss_rag_and_extra_envs_removed_from_code():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    for t in tracked:
        low = t.lower()
        assert not any(c in low for c in CUT + ("rag_pipeline", "docker-compose.staging", "docker-compose.prod")), t
    for py in (ROOT / "src").rglob("*.py"):
        text = py.read_text().lower()
        assert not any(re.search(rf"\b(import|from)\s+{c}", text) for c in CUT), py


def test_readme_mentions_cut_scope_only_under_how_it_extends():
    readme = (ROOT / "README.md").read_text()
    assert "## How it extends" in readme
    before, after = readme.split("## How it extends", 1)
    next_h = re.search(r"^## ", after, re.M)
    outside = before + (after[next_h.start():] if next_h else "")
    for c in CUT + ("three environments",):
        assert c not in outside.lower(), c
    assert ">85%" not in readme and "geological formations" not in readme
