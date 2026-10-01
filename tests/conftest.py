import pytest

from seismic_mlops.data import load_fixture


@pytest.fixture(scope="session")
def dataset():
    return load_fixture()


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    """Isolated MLflow backend + artifact store per test."""
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path}/mlflow.db"
