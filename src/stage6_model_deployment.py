"""
Stage 6: Model Deployment

Deploy model with multiple serving modes:
- REST API (FastAPI) for real-time inference
- Batch inference for large-scale processing
- MLflow model loading with version control

Updated to support:
- 40 features (8 handcrafted + 32 PCA embeddings)
- Models trained with tuned hyperparameters (Optuna TPE)
- Batch inference mode for production workloads
- Model versioning and staging
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn
from datetime import datetime
import json
from sklearn.decomposition import PCA

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ============================================================
# Pydantic Models for API
# ============================================================

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    traces: List[List[float]]  # List of trace data arrays
    include_embeddings: bool = True  # Whether to compute embeddings
    metadata: Optional[Dict[str, Any]] = {}


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Dict[str, Any]]
    timestamp: str
    model_version: str
    num_features: int


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    input_path: str  # Path to input file (Parquet/CSV)
    output_path: str  # Path for output predictions
    include_probabilities: bool = True


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    num_predictions: int
    output_path: str
    timestamp: str
    model_version: str


# ============================================================
# Model Serving Class
# ============================================================

class ModelServing:
    """
    Model serving class that loads and serves the model.
    
    Supports:
    - Real-time inference via REST API
    - Batch inference for large datasets
    - 40 features (8 handcrafted + 32 PCA embeddings)
    - MLflow model loading with version control
    """
    
    def __init__(self, 
                 model_path: str = "models/seismic_classifier.pkl",
                 scaler_path: str = "data/silver/feature_scaler.pkl",
                 pca_path: str = "data/silver/pca_model.pkl",
                 mlflow_uri: str = "file:./mlruns",
                 model_stage: str = None):
        """
        Initialize model serving.
        
        Args:
            model_path: Path to pickled model
            scaler_path: Path to feature scaler
            pca_path: Path to PCA model for embeddings
            mlflow_uri: MLflow tracking URI
            model_stage: MLflow model stage (None, Staging, Production)
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.pca_path = Path(pca_path)
        self.model = None
        self.scaler = None
        self.pca_model = None
        self.model_version = "unknown"
        self.model_type = "unknown"
        self.num_features = 40
        self.model_stage = model_stage
        
        mlflow.set_tracking_uri(mlflow_uri)
        self._load_model()
        self._load_preprocessors()
    
    def _load_model(self):
        """Load model from file or MLflow."""
        # Try loading from MLflow first if stage specified
        if self.model_stage:
            try:
                model_uri = f"models:/SeismicClassifier/{self.model_stage}"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = f"mlflow-{self.model_stage}"
                self.model_type = type(self.model).__name__
                print(f"Model loaded from MLflow: {model_uri}")
                return
            except Exception as e:
                print(f"MLflow loading failed: {e}, trying local file...")
        
        # Load from local file
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = type(self.model).__name__
            print(f"Model loaded from: {self.model_path}")
            print(f"Model type: {self.model_type}")
            
            # Try to get version from features file
            features_path = self.model_path.parent / "seismic_classifier_features.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    info = json.load(f)
                    self.num_features = info.get('num_features', 40)
                    self.model_version = f"local-{info.get('model_type', 'unknown')}"
        else:
            # Try loading latest from MLflow
            try:
                model_uri = "models:/SeismicClassifier/latest"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = "mlflow-latest"
                self.model_type = type(self.model).__name__
                print(f"Model loaded from MLflow: {model_uri}")
            except Exception as e:
                raise FileNotFoundError(f"Model not found: {e}")
    
    def _load_preprocessors(self):
        """Load scaler and PCA model."""
        # Load scaler
        if self.scaler_path.exists():
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from: {self.scaler_path}")
        else:
            print("Warning: Scaler not found, using raw features")
        
        # Load PCA model for embeddings
        if self.pca_path.exists():
            with open(self.pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            print(f"PCA model loaded from: {self.pca_path}")
        else:
            print("Warning: PCA model not found, embeddings will be zeros")
    
    def extract_handcrafted_features(self, trace_data: np.ndarray) -> np.ndarray:
        """
        Extract 8 handcrafted features from trace data.
        
        Args:
            trace_data: Array of amplitude values
            
        Returns:
            Array of 8 features
        """
        trace_array = np.array(trace_data, dtype=np.float32)
        
        features = np.array([
            np.mean(trace_array),                              # mean_amplitude
            np.std(trace_array),                               # std_amplitude
            np.min(trace_array),                               # min_amplitude
            np.max(trace_array),                               # max_amplitude
            np.sqrt(np.mean(trace_array**2)),                  # rms_amplitude
            np.sum(trace_array**2),                            # energy
            int(np.sum(np.diff(np.signbit(trace_array)))),     # zero_crossings
            self._dominant_frequency(trace_array)              # dominant_frequency
        ])
        
        return features
    
    def _dominant_frequency(self, trace: np.ndarray) -> float:
        """Calculate dominant frequency using FFT."""
        fft = np.fft.fft(trace)
        freqs = np.fft.fftfreq(len(trace))
        power = np.abs(fft)**2
        
        positive_freqs = freqs[1:len(freqs)//2]
        positive_power = power[1:len(power)//2]
        
        if len(positive_power) > 0:
            dominant_idx = np.argmax(positive_power)
            return float(abs(positive_freqs[dominant_idx]))
        return 0.0
    
    def extract_embeddings(self, trace_data: np.ndarray) -> np.ndarray:
        """
        Extract 32-dimensional PCA embeddings from trace data.
        
        Args:
            trace_data: Array of amplitude values
            
        Returns:
            Array of 32 embedding features
        """
        if self.pca_model is None:
            # Return zeros if PCA not available
            return np.zeros(32)
        
        trace_array = np.array(trace_data, dtype=np.float32).reshape(1, -1)
        
        # Ensure trace has correct length for PCA
        expected_length = self.pca_model.n_features_in_
        if trace_array.shape[1] != expected_length:
            # Resample or pad
            if trace_array.shape[1] > expected_length:
                trace_array = trace_array[:, :expected_length]
            else:
                padded = np.zeros((1, expected_length))
                padded[:, :trace_array.shape[1]] = trace_array
                trace_array = padded
        
        embeddings = self.pca_model.transform(trace_array)
        return embeddings.flatten()
    
    def extract_all_features(self, trace_data: np.ndarray, 
                            include_embeddings: bool = True) -> np.ndarray:
        """
        Extract all 40 features (8 handcrafted + 32 embeddings).
        
        Args:
            trace_data: Array of amplitude values
            include_embeddings: Whether to include PCA embeddings
            
        Returns:
            Array of 40 features (or 8 if embeddings disabled)
        """
        handcrafted = self.extract_handcrafted_features(trace_data)
        
        if include_embeddings:
            embeddings = self.extract_embeddings(trace_data)
            return np.concatenate([handcrafted, embeddings])
        else:
            return handcrafted
    
    def predict(self, traces: List[List[float]], 
               include_embeddings: bool = True) -> List[Dict[str, Any]]:
        """
        Make predictions on traces (real-time inference).
        
        Args:
            traces: List of trace data arrays
            include_embeddings: Whether to use embeddings
            
        Returns:
            List of prediction dictionaries
        """
        # Extract features for all traces
        features_list = []
        for trace in traces:
            features = self.extract_all_features(
                np.array(trace), 
                include_embeddings=include_embeddings
            )
            features_list.append(features)
        
        X = np.array(features_list)
        
        # Scale features if scaler available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(X)
            except:
                pass
        
        # Format results
        results = []
        class_names = ['normal', 'anomaly', 'boundary']
        
        for i, pred in enumerate(predictions):
            result = {
                'trace_id': i,
                'predicted_class': int(pred),
                'predicted_class_name': class_names[int(pred)],
            }
            
            if probabilities is not None:
                result['probabilities'] = {
                    class_names[j]: float(prob) 
                    for j, prob in enumerate(probabilities[i])
                }
                result['confidence'] = float(np.max(probabilities[i]))
            
            results.append(result)
        
        return results
    
    def batch_predict(self, input_path: str, output_path: str,
                     include_probabilities: bool = True) -> Dict[str, Any]:
        """
        Batch inference for large-scale processing.
        
        Args:
            input_path: Path to input file (Parquet with features or raw traces)
            output_path: Path for output predictions
            include_probabilities: Whether to include class probabilities
            
        Returns:
            Summary of batch prediction
        """
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load input data
        if input_file.suffix == '.parquet':
            df = pd.read_parquet(input_file)
        elif input_file.suffix == '.csv':
            df = pd.read_csv(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
        print(f"Loaded {len(df)} samples from {input_path}")
        
        # Define expected feature columns (40 features: 8 handcrafted + 32 embeddings)
        expected_feature_cols = [
            'mean_amplitude_scaled', 'std_amplitude_scaled', 'min_amplitude_scaled',
            'max_amplitude_scaled', 'rms_amplitude_scaled', 'energy_scaled',
            'zero_crossings_scaled', 'dominant_frequency_scaled'
        ] + [f'embedding_{i}_scaled' for i in range(32)]
        
        # Check which expected features are available
        available_cols = [col for col in expected_feature_cols if col in df.columns]
        
        if len(available_cols) >= 40:
            # Use the 40 expected features
            X = df[available_cols].values
            print(f"Using {len(available_cols)} features (40 expected)")
        elif len(available_cols) >= 8:
            # At least handcrafted features available
            X = df[available_cols].values
            print(f"Using {len(available_cols)} available features")
        else:
            # Fallback: try any scaled/embedding columns
            feature_cols = [col for col in df.columns if 'scaled' in col or 'embedding' in col]
            if len(feature_cols) >= 8:
                X = df[feature_cols[:40]].values  # Take first 40
                print(f"Using {min(len(feature_cols), 40)} pre-extracted features")
            else:
                raise ValueError("Input must contain pre-extracted features for batch mode")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create output dataframe
        output_df = df.copy()
        output_df['predicted_class'] = predictions
        output_df['predicted_class_name'] = [
            ['normal', 'anomaly', 'boundary'][int(p)] for p in predictions
        ]
        
        # Add probabilities if requested
        if include_probabilities and hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(X)
                output_df['prob_normal'] = proba[:, 0]
                output_df['prob_anomaly'] = proba[:, 1]
                output_df['prob_boundary'] = proba[:, 2]
                output_df['confidence'] = np.max(proba, axis=1)
            except:
                pass
        
        # Save output
        if output_file.suffix == '.parquet':
            output_df.to_parquet(output_file, index=False)
        else:
            output_df.to_csv(output_file, index=False)
        
        print(f"Predictions saved to: {output_path}")
        
        return {
            'num_predictions': len(predictions),
            'output_path': str(output_file),
            'class_distribution': {
                'normal': int(np.sum(predictions == 0)),
                'anomaly': int(np.sum(predictions == 1)),
                'boundary': int(np.sum(predictions == 2))
            }
        }


# ============================================================
# FastAPI Application
# ============================================================

# Initialize FastAPI app
app = FastAPI(
    title="Seismic Classification API",
    description="""
    ML Model Serving API for Seismic Data Classification
    
    Features:
    - Real-time inference via /predict endpoint
    - Batch inference via /batch_predict endpoint
    - 40 features (8 handcrafted + 32 PCA embeddings)
    - Model trained with Optuna TPE hyperparameter optimization
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model serving instance (lazy initialization)
model_serving: Optional[ModelServing] = None


def get_model_serving() -> ModelServing:
    """Get or initialize model serving instance."""
    global model_serving
    if model_serving is None:
        model_serving = ModelServing()
    return model_serving


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Seismic Classification API",
        "version": "2.0.0",
        "features": "40 (8 handcrafted + 32 PCA embeddings)",
        "endpoints": {
            "/predict": "POST - Real-time predictions",
            "/batch_predict": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/model_info": "GET - Model information"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    serving = get_model_serving()
    return {
        "status": "healthy",
        "model_loaded": serving.model is not None,
        "scaler_loaded": serving.scaler is not None,
        "pca_loaded": serving.pca_model is not None,
        "model_type": serving.model_type,
        "num_features": serving.num_features,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model_info")
async def model_info():
    """Get model information."""
    serving = get_model_serving()
    
    info = {
        "model_type": serving.model_type,
        "model_version": serving.model_version,
        "num_features": serving.num_features,
        "feature_breakdown": {
            "handcrafted": 8,
            "pca_embeddings": 32
        },
        "classes": ["normal", "anomaly", "boundary"],
        "deployment_modes": ["real-time", "batch"]
    }
    
    # Add tuning info if available
    tuning_path = Path("models/hyperparameter_tuning_results.json")
    if tuning_path.exists():
        with open(tuning_path, 'r') as f:
            tuning = json.load(f)
            info["hyperparameter_tuning"] = {
                "method": "Optuna TPE",
                "best_model_type": tuning.get("best_model_type"),
                "best_cv_score": tuning.get("best_cv_score")
            }
    
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make real-time predictions on seismic traces.
    
    Args:
        request: Prediction request with traces
        
    Returns:
        Prediction response with class labels and probabilities
    """
    try:
        serving = get_model_serving()
        
        if not serving.model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not request.traces:
            raise HTTPException(status_code=400, detail="No traces provided")
        
        # Make predictions
        predictions = serving.predict(
            request.traces,
            include_embeddings=request.include_embeddings
        )
        
        return PredictionResponse(
            predictions=predictions,
            timestamp=datetime.now().isoformat(),
            model_version=serving.model_version,
            num_features=serving.num_features
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch inference for large-scale processing.
    
    Args:
        request: Batch prediction request with input/output paths
        
    Returns:
        Batch prediction response with summary
    """
    try:
        serving = get_model_serving()
        
        if not serving.model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Run batch prediction
        result = serving.batch_predict(
            request.input_path,
            request.output_path,
            request.include_probabilities
        )
        
        return BatchPredictionResponse(
            num_predictions=result['num_predictions'],
            output_path=result['output_path'],
            timestamp=datetime.now().isoformat(),
            model_version=serving.model_version
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Batch Inference CLI
# ============================================================

def run_batch_inference(input_path: str, output_path: str):
    """
    Run batch inference from command line.
    
    Args:
        input_path: Path to input features file
        output_path: Path for output predictions
    """
    print("=" * 60)
    print("Batch Inference Mode")
    print("=" * 60)
    
    serving = ModelServing()
    
    print(f"\nModel: {serving.model_type}")
    print(f"Features: {serving.num_features}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    result = serving.batch_predict(input_path, output_path)
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Batch inference complete!")
    print(f"{'='*60}")
    print(f"  Predictions: {result['num_predictions']}")
    print(f"  Class distribution:")
    for cls, count in result['class_distribution'].items():
        print(f"    {cls}: {count}")
    print(f"  Output: {result['output_path']}")


def main():
    """Run the FastAPI server or batch inference."""
    import sys
    
    # Check for batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("Usage: python stage6_model_deployment.py --batch <input_path> <output_path>")
            sys.exit(1)
        run_batch_inference(sys.argv[2], sys.argv[3])
        return
    
    # API mode
    print("=" * 60)
    print("Stage 6: Model Deployment")
    print("=" * 60)
    print("\nDeployment Modes:")
    print("  1. REST API (default): Real-time inference")
    print("  2. Batch: python stage6_model_deployment.py --batch <input> <output>")
    
    if not UVICORN_AVAILABLE:
        print("\nError: uvicorn not installed. Install with: pip install uvicorn")
        return
    
    print("\nStarting FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("Documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
