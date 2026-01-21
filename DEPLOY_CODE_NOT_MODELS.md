# Deploy Code Not Models - Implementation Guide

> **Repository:** [https://github.com/vospr/seismic-mlops-pipeline](https://github.com/vospr/seismic-mlops-pipeline)

---

## Principle Overview

**"Deploy Code Not Models"** means:
- ✅ Deploy the **training code** to each environment
- ✅ Each environment **trains its own model** using environment-specific data
- ✅ Models are **registered in environment-specific MLflow** instances
- ❌ Do NOT copy model files between environments
- ❌ Do NOT share MLflow databases between environments

---

## Why This Matters

| Approach | Deploy Models | Deploy Code |
|----------|--------------|-------------|
| Reproducibility | ❌ Hard to reproduce | ✅ Fully reproducible |
| Auditability | ❌ "Where did this model come from?" | ✅ Clear lineage |
| Data governance | ❌ Prod data in dev? | ✅ Environment isolation |
| Rollback | ❌ Which version? | ✅ Git commit = model version |
| Testing | ❌ Test the artifact | ✅ Test the process |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Git Repository (Single Source)                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  src/stage0-8_*.py  │  config/  │  Dockerfile  │  requirements  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│    DEVELOPMENT      │ │      STAGING        │ │     PRODUCTION      │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ docker-compose.yml  │ │ docker-compose.     │ │ docker-compose.     │
│                     │ │ staging.yml         │ │ prod.yml            │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ ENVIRONMENT=dev     │ │ ENVIRONMENT=staging │ │ ENVIRONMENT=prod    │
│ MLflow: localhost   │ │ MLflow: staging:5001│ │ MLflow: prod:5002   │
│ Data: ./data        │ │ Data: staging-data  │ │ Data: prod-data     │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│                     │ │                     │ │                     │
│  ┌───────────────┐  │ │  ┌───────────────┐  │ │  ┌───────────────┐  │
│  │ Train Model   │  │ │  │ Train Model   │  │ │  │ Train Model   │  │
│  │ (dev data)    │  │ │  │ (staging data)│  │ │  │ (prod data)   │  │
│  └───────────────┘  │ │  └───────────────┘  │ │  └───────────────┘  │
│         │           │ │         │           │ │         │           │
│         ▼           │ │         ▼           │ │         ▼           │
│  ┌───────────────┐  │ │  ┌───────────────┐  │ │  ┌───────────────┐  │
│  │ MLflow        │  │ │  │ MLflow        │  │ │  │ MLflow        │  │
│  │ Experiment:   │  │ │  │ Experiment:   │  │ │  │ Experiment:   │  │
│  │ _dev          │  │ │  │ _staging      │  │ │  │ _prod         │  │
│  └───────────────┘  │ │  └───────────────┘  │ │  └───────────────┘  │
│         │           │ │         │           │ │         │           │
│         ▼           │ │         ▼           │ │         ▼           │
│  Model Stage:       │ │  Model Stage:       │ │  Model Stage:       │
│  "None"             │ │  "Staging"          │ │  "Production"       │
│                     │ │                     │ │                     │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

---

## File Structure

```
seismic-mlops-pipeline/
├── config/
│   └── environments.py          # Environment configurations
├── docker-compose.yml           # Development environment
├── docker-compose.staging.yml   # Staging environment
├── docker-compose.prod.yml      # Production environment
├── scripts/
│   └── deploy.py               # Deployment script
├── src/
│   └── stage*.py               # Same code for all environments
└── .env.staging                # Staging secrets (not in git)
└── .env.prod                   # Production secrets (not in git)
```

---

## Environment Configuration

### config/environments.py

```python
from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    name: str
    data_path: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_stage: str  # "None", "Staging", "Production"
    enable_prometheus: bool
    log_level: str

DEVELOPMENT = EnvironmentConfig(
    name="development",
    data_path="./data",
    mlflow_tracking_uri="file:./mlruns",
    mlflow_experiment_name="seismic_classification_dev",
    model_stage="None",
    enable_prometheus=False,
    log_level="DEBUG",
)

STAGING = EnvironmentConfig(
    name="staging",
    data_path="/app/data",
    mlflow_tracking_uri="http://mlflow-staging:5000",
    mlflow_experiment_name="seismic_classification_staging",
    model_stage="Staging",
    enable_prometheus=True,
    log_level="INFO",
)

PRODUCTION = EnvironmentConfig(
    name="production",
    data_path="/app/data",
    mlflow_tracking_uri="http://mlflow-prod:5000",
    mlflow_experiment_name="seismic_classification_prod",
    model_stage="Production",
    enable_prometheus=True,
    log_level="WARNING",
)
```

---

## Deployment Commands

### Deploy to Development

```bash
# Uses docker-compose.yml
docker-compose up -d
docker-compose exec mlops python run_all_stages.py
```

### Deploy to Staging

```bash
# Uses docker-compose.staging.yml
docker-compose -f docker-compose.staging.yml up -d
docker-compose -f docker-compose.staging.yml exec mlops python run_all_stages.py
```

### Deploy to Production

```bash
# Uses docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml exec mlops python run_all_stages.py
```

### Using the Deploy Script

```bash
# Deploy to staging with training
python scripts/deploy.py --env staging

# Deploy to production without retraining
python scripts/deploy.py --env production --skip-training

# Promote model from staging to production
python scripts/deploy.py --env production --promote-from staging
```

---

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy Pipeline

on:
  push:
    branches:
      - main        # Deploy to staging
      - release/*   # Deploy to production

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Staging
        run: |
          # Same code, staging environment
          docker-compose -f docker-compose.staging.yml build
          docker-compose -f docker-compose.staging.yml up -d
          docker-compose -f docker-compose.staging.yml exec -T mlops \
            python run_all_stages.py
      
      - name: Validate Staging
        run: |
          docker-compose -f docker-compose.staging.yml exec -T mlops \
            python src/stage8_cicd.py

  deploy-production:
    if: startsWith(github.ref, 'refs/heads/release/')
    runs-on: ubuntu-latest
    environment: production  # Requires approval
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Production
        run: |
          # Same code, production environment
          docker-compose -f docker-compose.prod.yml build
          docker-compose -f docker-compose.prod.yml up -d
          docker-compose -f docker-compose.prod.yml exec -T mlops \
            python run_all_stages.py
```

---

## Model Promotion Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Development │────▶│   Staging   │────▶│ Production  │
│             │     │             │     │             │
│ Stage: None │     │ Stage:      │     │ Stage:      │
│             │     │ Staging     │     │ Production  │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      │                   │                   │
      ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Dev MLflow  │     │ Staging     │     │ Prod MLflow │
│ :5000       │     │ MLflow :5001│     │ :5002       │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Promotion via MLflow Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote model from Staging to Production
client.transition_model_version_stage(
    name="SeismicClassifier",
    version=8,
    stage="Production"
)
```

---

## Key Differences by Environment

| Aspect | Development | Staging | Production |
|--------|-------------|---------|------------|
| **Compose file** | `docker-compose.yml` | `docker-compose.staging.yml` | `docker-compose.prod.yml` |
| **MLflow port** | 5000 | 5001 | 5002 |
| **Model stage** | None | Staging | Production |
| **Replicas** | 1 | 1 | 2+ |
| **Resource limits** | None | Soft | Hard |
| **LLM enabled** | Yes | Yes | Optional |
| **Prometheus** | Optional | Yes | Yes |
| **Alerting** | None | Dev channel | Prod channel |
| **Log level** | DEBUG | INFO | WARNING |
| **Source mount** | Yes (hot reload) | Yes | No (baked in) |

---

## Benefits Achieved

### 1. Reproducibility
```bash
# Any commit can be deployed to any environment
git checkout v1.2.3
docker-compose -f docker-compose.staging.yml up -d
docker-compose -f docker-compose.staging.yml exec mlops python run_all_stages.py
# Result: Exact same model as v1.2.3 in staging
```

### 2. Auditability
```
MLflow Experiment: seismic_classification_prod
├── Run: 2026-01-21_training
│   ├── Git commit: a699b8d
│   ├── Parameters: {C: 0.1, max_iter: 200}
│   ├── Metrics: {accuracy: 0.49, f1: 0.48}
│   └── Artifacts: model.pkl, scaler.pkl
```

### 3. Environment Isolation
- Dev data never touches production
- Each environment has its own MLflow instance
- Secrets are environment-specific

### 4. Easy Rollback
```bash
# Rollback to previous version
git checkout v1.1.0
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml exec mlops python run_all_stages.py
```

---

## Summary

**"Deploy Code Not Models" in this project means:**

1. **One codebase** in Git
2. **Three compose files** for three environments
3. **Environment-specific configs** via `ENVIRONMENT` variable
4. **Each environment trains its own model**
5. **MLflow tracks everything** per environment
6. **Model promotion** via MLflow stages, not file copying

This ensures **reproducibility**, **auditability**, and **environment isolation** - the core principles of production MLOps.

---

*Document created: DEPLOY_CODE_NOT_MODELS.md*
