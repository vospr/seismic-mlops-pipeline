# Quick Start Guide - MLOps Seismic Classification Pipeline

## Overview

This pipeline implements a complete MLOps workflow for seismic data classification with:
- **40 features** (8 handcrafted + 32 PCA embeddings)
- **Optuna TPE** hyperparameter optimization
- **MLflow** experiment tracking and model registry
- **Feast** feature store integration
- **FAISS** vector store for RAG
- **Ollama LLM** for intelligent analysis
- **Docker** containerization for easy deployment

---

## Quick Start Options

### Option 1: Docker (Recommended for Teams)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/seismic-mlops-pipeline.git
cd seismic-mlops-pipeline

# Start all services
docker-compose up -d

# Wait for services to start (MLflow takes ~60s to install)
docker-compose ps

# Run CI/CD validation
docker-compose exec mlops python src/stage8_cicd.py

# Run full pipeline
docker-compose exec mlops python run_all_stages.py
```

**Services available:**
| Service | URL | Description |
|---------|-----|-------------|
| API Server | http://localhost:8000 | FastAPI with Swagger UI |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| Metrics | http://localhost:8001/metrics | Prometheus metrics |

**Docker Commands:**
```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose build --no-cache

# Start with LLM support (optional)
docker-compose --profile llm up -d
docker-compose exec ollama ollama pull llama3.1:8b
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/seismic-mlops-pipeline.git
cd seismic-mlops-pipeline

# Windows
scripts\setup-local.bat

# Linux/Mac
chmod +x scripts/setup-local.sh
./scripts/setup-local.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Run the pipeline
python run_all_stages.py
```

---

## Prerequisites

### For Docker
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 4GB+ RAM available for containers

### For Local Installation
- Python 3.11+
- Ollama (optional, for LLM features)

### Start Ollama (for LLM features - Local only)

```bash
# Start Ollama service
ollama serve

# Pull the model (first time only)
ollama pull llama3.1:8b
```

---

## Pipeline Stages

| Stage | Script | Description | Output |
|-------|--------|-------------|--------|
| 0 | `stage0_data_sampling.py` | Sample F3 dataset, KNN imputation | `data/raw/*.sgy` |
| 1 | `stage1_data_ingestion.py` | Ingest SGY, quality validation | `data/bronze/` |
| 2 | `stage2_feature_engineering.py` | Extract 40 features, embeddings | `data/silver/` |
| 3b | `stage3_hyperparameter_tuning.py` | Optuna TPE optimization | `models/hyperparameter_tuning_results.json` |
| 3 | `stage3_model_training.py` | Train with tuned hyperparameters | `models/*.pkl` |
| 4 | `stage4_model_evaluation.py` | Evaluate, drift detection | `data/gold/evaluation_results.json` |
| 5 | `stage5_model_registry.py` | MLflow model registry | `models/SeismicClassifier_v*/` |
| 6 | `stage6_model_deployment.py` | FastAPI server, batch inference | `data/gold/batch_predictions.parquet` |
| 7 | `stage7_monitoring.py` | Prometheus metrics, drift alerts | `data/gold/monitoring_report.json` |
| 8 | `stage8_cicd.py` | CI/CD validation | `cicd_results.json` |

---

## Run Individual Stages

### Using Docker

```bash
# Run specific stage
docker-compose exec mlops python src/stage1_data_ingestion.py

# Run all stages
docker-compose exec mlops python run_all_stages.py

# Run with options
docker-compose exec mlops python run_all_stages.py --quick      # Skip HP tuning
docker-compose exec mlops python run_all_stages.py --from 3     # Start from stage 3
```

### Using Local Installation

```bash
# Stage 0: Data Sampling (from F3 dataset)
python src/stage0_data_sampling.py

# Stage 1: Data Ingestion & Quality Assurance
python src/stage1_data_ingestion.py

# Stage 2: Feature Engineering (40 features)
python src/stage2_feature_engineering.py

# Stage 3b: Hyperparameter Tuning (Optuna TPE)
python src/stage3_hyperparameter_tuning.py

# Stage 3: Model Training (uses tuned parameters)
python src/stage3_model_training.py

# Stage 4: Model Evaluation & Drift Detection
python src/stage4_model_evaluation.py

# Stage 5: Model Registry (MLflow)
python src/stage5_model_registry.py

# Stage 6: Model Deployment
python src/stage6_model_deployment.py           # Start API server
python src/stage6_model_deployment.py --batch   # Run batch inference

# Stage 7: Monitoring & Observability
python src/stage7_monitoring.py

# Stage 8: CI/CD Validation
python src/stage8_cicd.py                       # Quick validation
python src/stage8_cicd.py --full                # Full pipeline run
python src/stage8_cicd.py --workflow            # Generate GitHub Actions
```

---

## Quick Validation (CI/CD)

Validate all stages completed successfully:

```bash
# Docker
docker-compose exec mlops python src/stage8_cicd.py

# Local
python src/stage8_cicd.py
```

Expected output:
```
[OK] Overall Status: SUCCESS

Stage Results:
  [OK] stage0_sampling: success
  [OK] stage1_ingestion: success
  [OK] stage2_features: success
  [OK] stage3_training: success
  [OK] stage3b_tuning: success (cv=0.422)
  [OK] stage4_evaluation: success (acc=0.490)
  [OK] stage5_registry: success
  [OK] stage6_deployment: success (n=500)
  [OK] stage7_monitoring: success (alerts=0)
```

---

## Test API (Stage 6)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Batch predictions |
| `/model_info` | GET | Model metadata |
| `/docs` | GET | Swagger UI |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model_info

# Make prediction (with trace data)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trace_data": [1.0, 2.0, 3.0, ...]}'
```

### PowerShell (Windows)

```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8000/health

# Model info
Invoke-RestMethod -Uri http://localhost:8000/model_info
```

---

## View MLflow

### Access

- **URL**: http://localhost:5000
- **Experiments**: `seismic_classification`, `seismic_hyperparameter_tuning`
- **Models**: `SeismicClassifier` (multiple versions)

### Start MLflow UI (Local only)

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

### What You'll See

| Tab | Content |
|-----|---------|
| Experiments | Training runs with metrics, parameters |
| Models | Registered model versions |
| Artifacts | Model files, confusion matrices, reports |

---

## View Metrics (Stage 7)

### Prometheus Metrics

- **URL**: http://localhost:8001/metrics

### Key Metrics

| Metric | Description |
|--------|-------------|
| `seismic_model_accuracy` | Current model accuracy |
| `seismic_drift_score` | Data drift score (0-1) |
| `seismic_features_drifted_count` | Number of drifted features |
| `seismic_predictions_total` | Total predictions counter |

---

## Project Structure

```
seismic-mlops-pipeline/
├── data/
│   ├── raw/                    # Stage 0: Sampled SGY files
│   ├── bronze/                 # Stage 1: Ingested data (Delta Lake)
│   ├── silver/                 # Stage 2: Features
│   └── gold/                   # Stage 4-7: Results
├── models/                     # Trained models & artifacts
├── mlruns/                     # MLflow tracking
├── feature_store/              # Feast feature store
├── scripts/                    # Setup scripts
│   ├── docker-start.sh/.bat   # Docker quick start
│   └── setup-local.sh/.bat    # Local setup
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
│   └── Code explanation.md
├── .github/workflows/          # CI/CD workflows
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service setup
├── requirements.txt            # Python dependencies
├── run_all_stages.py           # Run complete pipeline
└── README.md                   # Project overview
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Data Format | SGY/SEGY (segyio) |
| Data Storage | Delta Lake, Parquet |
| Feature Store | Feast |
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow |
| Hyperparameter Optimization | Optuna (TPE sampler) |
| LLM Integration | Ollama (llama3.1:8b) |
| Vector Store | FAISS |
| Model Serving | FastAPI |
| Monitoring | Prometheus |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Troubleshooting

### Docker Issues

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs mlops
docker-compose logs mlflow

# Restart services
docker-compose restart

# Full rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Ollama Not Available (Local)

```bash
# Check Ollama is running
ollama list

# Start if needed
ollama serve
```

### MLflow UI Not Loading (Local)

```bash
# Check mlruns directory exists
ls mlruns/

# Start with explicit path
mlflow ui --backend-store-uri file:./mlruns
```

### Port Already in Use

```bash
# Find process using port (Linux/Mac)
lsof -i :8000

# Find process using port (Windows)
netstat -an | findstr "8000"

# Use different port
python src/stage6_model_deployment.py --port 8080
```

---

## Next Steps

1. **Start Services**: `docker-compose up -d`
2. **Validate Pipeline**: `docker-compose exec mlops python src/stage8_cicd.py`
3. **Explore MLflow**: http://localhost:5000
4. **Test API**: http://localhost:8000/docs
5. **Check Metrics**: http://localhost:8001/metrics
6. **Read Documentation**: See `src/Code explanation.md` for detailed architecture
