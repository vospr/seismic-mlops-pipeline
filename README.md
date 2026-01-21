# Seismic MLOps Pipeline

A production-ready MLOps pipeline for seismic data classification demonstrating end-to-end machine learning lifecycle management.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **40 Features**: 8 handcrafted + 32 PCA embeddings from seismic traces
- **Hyperparameter Optimization**: Optuna with TPE (Tree of Parzen Estimators)
- **Experiment Tracking**: MLflow for metrics, parameters, and artifacts
- **Model Registry**: Versioned model management with MLflow
- **Feature Store**: Feast integration for feature management
- **RAG Pipeline**: FAISS vector store with TF-IDF embeddings
- **LLM Integration**: Ollama for intelligent analysis and insights
- **Model Serving**: FastAPI REST API with batch inference support
- **Monitoring**: Prometheus metrics and drift detection
- **CI/CD**: GitHub Actions workflow with validation gates

## Quick Start

### Option 1: Docker (Recommended for Teams)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/seismic-mlops-pipeline.git
cd seismic-mlops-pipeline

# Quick start (Linux/Mac)
chmod +x scripts/docker-start.sh
./scripts/docker-start.sh

# Quick start (Windows)
scripts\docker-start.bat

# Or manually:
docker-compose up -d
docker-compose exec mlops python run_all_stages.py
```

**Services available after startup:**
- API Server: http://localhost:8000
- MLflow UI: http://localhost:5000
- Metrics: http://localhost:8001/metrics

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/seismic-mlops-pipeline.git
cd seismic-mlops-pipeline

# Quick setup (Linux/Mac)
chmod +x scripts/setup-local.sh
./scripts/setup-local.sh

# Quick setup (Windows)
scripts\setup-local.bat

# Or manually:
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run the pipeline
python run_all_stages.py
```

### Option 3: Quick Validation

```bash
# Validate all stages completed successfully
python src/stage8_cicd.py

# Run pipeline in quick mode (skip hyperparameter tuning)
python run_all_stages.py --quick

# Start from specific stage
python run_all_stages.py --from 3
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Seismic MLOps Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 0          Stage 1           Stage 2          Stage 3    │
│  ┌──────────┐    ┌──────────┐      ┌──────────┐    ┌──────────┐ │
│  │ Sampling │───▶│ Ingest   │─────▶│ Features │───▶│ Training │ │
│  │ KNN      │    │ Quality  │      │ 40 feat  │    │ MLflow   │ │
│  └──────────┘    └──────────┘      └──────────┘    └──────────┘ │
│       │               │                 │               │        │
│       ▼               ▼                 ▼               ▼        │
│  data/raw/       data/bronze/      data/silver/     models/     │
│                                                                  │
│  Stage 3b         Stage 4           Stage 5          Stage 6    │
│  ┌──────────┐    ┌──────────┐      ┌──────────┐    ┌──────────┐ │
│  │ Optuna   │───▶│ Evaluate │─────▶│ Registry │───▶│ Deploy   │ │
│  │ TPE      │    │ Drift    │      │ Version  │    │ FastAPI  │ │
│  └──────────┘    └──────────┘      └──────────┘    └──────────┘ │
│       │               │                 │               │        │
│       ▼               ▼                 ▼               ▼        │
│  tuning.json     data/gold/        mlruns/        API:8000     │
│                                                                  │
│  Stage 7          Stage 8                                        │
│  ┌──────────┐    ┌──────────┐                                   │
│  │ Monitor  │    │ CI/CD    │                                   │
│  │ Metrics  │    │ Validate │                                   │
│  └──────────┘    └──────────┘                                   │
│       │               │                                          │
│       ▼               ▼                                          │
│  Prometheus      GitHub Actions                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `stage0_data_sampling.py` | Sample F3 dataset, KNN imputation |
| 1 | `stage1_data_ingestion.py` | Ingest SGY/SEGY, quality validation |
| 2 | `stage2_feature_engineering.py` | Extract 40 features (8 handcrafted + 32 PCA) |
| 3 | `stage3_model_training.py` | Train model with tuned hyperparameters |
| 3b | `stage3_hyperparameter_tuning.py` | Optuna TPE optimization |
| 4 | `stage4_model_evaluation.py` | Evaluate model, detect drift |
| 5 | `stage5_model_registry.py` | Register model in MLflow |
| 6 | `stage6_model_deployment.py` | FastAPI server + batch inference |
| 7 | `stage7_monitoring.py` | Prometheus metrics, alerts |
| 8 | `stage8_cicd.py` | CI/CD validation |

## Project Structure

```
seismic-mlops-pipeline/
├── data/
│   ├── raw/                    # Sampled SGY files
│   ├── bronze/                 # Ingested data (Delta Lake)
│   ├── silver/                 # Engineered features
│   └── gold/                   # Predictions & results
├── models/                     # Trained models & artifacts
├── mlruns/                     # MLflow tracking
├── feature_store/              # Feast feature store
├── src/                        # Pipeline stages
│   ├── stage0_data_sampling.py
│   ├── stage1_data_ingestion.py
│   ├── stage2_feature_engineering.py
│   ├── stage3_model_training.py
│   ├── stage3_hyperparameter_tuning.py
│   ├── stage4_model_evaluation.py
│   ├── stage5_model_registry.py
│   ├── stage6_model_deployment.py
│   ├── stage7_monitoring.py
│   ├── stage8_cicd.py
│   └── Code explanation.md     # Detailed documentation
├── scripts/                    # Setup & utility scripts
│   ├── docker-start.sh         # Docker quick start (Linux/Mac)
│   ├── docker-start.bat        # Docker quick start (Windows)
│   ├── setup-local.sh          # Local setup (Linux/Mac)
│   └── setup-local.bat         # Local setup (Windows)
├── .github/workflows/          # CI/CD workflows
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service setup
├── requirements.txt            # Python dependencies
├── run_all_stages.py           # Run complete pipeline
├── QUICK_START.md              # Quick start guide
└── README.md                   # This file
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Data Format | SGY/SEGY (segyio) |
| Data Storage | Delta Lake, Parquet |
| Feature Store | Feast |
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow |
| Hyperparameter Tuning | Optuna (TPE) |
| LLM Integration | Ollama (llama3.1:8b) |
| Vector Store | FAISS |
| Model Serving | FastAPI |
| Monitoring | Prometheus |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## API Endpoints

Start the server: `python src/stage6_model_deployment.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Batch predictions |
| `/model_info` | GET | Model metadata |

## MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Access at: http://localhost:5000

## Prometheus Metrics

After running Stage 7: http://localhost:8001/metrics

| Metric | Description |
|--------|-------------|
| `seismic_model_accuracy` | Current model accuracy |
| `seismic_drift_score` | Data drift score (0-1) |
| `seismic_predictions_total` | Total predictions |

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Detailed quick start guide
- **[src/Code explanation.md](src/Code%20explanation.md)** - Complete architecture documentation
- **[PROJECT_DESCRIPTION_MLOPS.md](PROJECT_DESCRIPTION_MLOPS.md)** - MLOps requirements reference

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- F3 Netherlands seismic dataset
- MLflow for experiment tracking
- Optuna for hyperparameter optimization
- Ollama for local LLM inference
