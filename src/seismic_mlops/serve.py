"""Serve the registered Pipeline exactly as logged. No feature code lives here."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from .data import PATCH


class PredictRequest(BaseModel):
    patches: list[list[list[float]]]

    @field_validator("patches")
    @classmethod
    def _shape(cls, v):
        if not v or any(len(p) != PATCH or any(len(r) != PATCH for r in p) for p in v):
            raise ValueError(f"expected a non-empty list of {PATCH}x{PATCH} patches")
        return v


def create_app(model_uri: str | None = None, tracking_uri: str | None = None) -> FastAPI:
    model_uri = model_uri or os.environ.get("MODEL_URI", "models:/seismic-facies@champion")
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mlflow.set_tracking_uri(tracking_uri)
        app.state.model = mlflow.sklearn.load_model(model_uri)
        yield

    app = FastAPI(title="seismic-facies", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_uri": model_uri}

    @app.post("/predict")
    def predict(req: PredictRequest):
        flat = np.asarray(req.patches, dtype="float32").reshape(len(req.patches), -1)
        model = app.state.model
        return {"facies": model.predict(flat).tolist(), "proba": model.predict_proba(flat).tolist()}

    return app


app = None  # `uvicorn --factory seismic_mlops.serve:create_app`
