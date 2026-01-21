"""
Environment Configuration for Seismic MLOps Pipeline

Implements "Deploy Code Not Models" principle:
- Same code deployed to all environments
- Each environment trains its own model
- Environment-specific configurations
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Configuration for a specific environment."""
    
    # Environment identification
    name: str
    
    # Data paths
    data_path: str
    models_path: str
    mlruns_path: str
    
    # MLflow settings
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    
    # Model settings
    model_stage: str  # "None", "Staging", "Production"
    
    # API settings
    api_host: str
    api_port: int
    
    # Monitoring settings
    prometheus_port: int
    enable_prometheus: bool
    
    # Alerting
    slack_webhook: Optional[str] = None
    alert_email: Optional[str] = None
    
    # Feature flags
    enable_llm: bool = True
    enable_drift_detection: bool = True
    log_level: str = "INFO"


# =============================================================================
# ENVIRONMENT DEFINITIONS
# =============================================================================

DEVELOPMENT = EnvironmentConfig(
    name="development",
    
    # Local paths for development
    data_path="./data",
    models_path="./models",
    mlruns_path="./mlruns",
    
    # Local MLflow
    mlflow_tracking_uri="file:./mlruns",
    mlflow_experiment_name="seismic_classification_dev",
    
    # Models not promoted
    model_stage="None",
    
    # Local API
    api_host="127.0.0.1",
    api_port=8000,
    
    # Prometheus disabled by default
    prometheus_port=8001,
    enable_prometheus=False,
    
    # No alerting in dev
    slack_webhook=None,
    alert_email=None,
    
    # All features enabled for testing
    enable_llm=True,
    enable_drift_detection=True,
    log_level="DEBUG",
)


STAGING = EnvironmentConfig(
    name="staging",
    
    # Staging paths (could be mounted volumes)
    data_path="/app/data",
    models_path="/app/models",
    mlruns_path="/app/mlruns",
    
    # Staging MLflow server
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-staging:5000"),
    mlflow_experiment_name="seismic_classification_staging",
    
    # Models in staging
    model_stage="Staging",
    
    # Staging API
    api_host="0.0.0.0",
    api_port=8000,
    
    # Prometheus enabled
    prometheus_port=8001,
    enable_prometheus=True,
    
    # Alerting to dev channel
    slack_webhook=os.getenv("SLACK_WEBHOOK_STAGING"),
    alert_email=os.getenv("ALERT_EMAIL_STAGING"),
    
    # All features enabled
    enable_llm=True,
    enable_drift_detection=True,
    log_level="INFO",
)


PRODUCTION = EnvironmentConfig(
    name="production",
    
    # Production paths (mounted volumes)
    data_path="/app/data",
    models_path="/app/models",
    mlruns_path="/app/mlruns",
    
    # Production MLflow server
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-prod:5000"),
    mlflow_experiment_name="seismic_classification_prod",
    
    # Models promoted to production
    model_stage="Production",
    
    # Production API
    api_host="0.0.0.0",
    api_port=8000,
    
    # Prometheus enabled
    prometheus_port=8001,
    enable_prometheus=True,
    
    # Production alerting
    slack_webhook=os.getenv("SLACK_WEBHOOK_PROD"),
    alert_email=os.getenv("ALERT_EMAIL_PROD"),
    
    # LLM may be disabled for cost
    enable_llm=os.getenv("ENABLE_LLM", "false").lower() == "true",
    enable_drift_detection=True,
    log_level="WARNING",
)


# =============================================================================
# ENVIRONMENT SELECTION
# =============================================================================

ENVIRONMENTS = {
    "development": DEVELOPMENT,
    "dev": DEVELOPMENT,
    "staging": STAGING,
    "stg": STAGING,
    "production": PRODUCTION,
    "prod": PRODUCTION,
}


def get_environment() -> EnvironmentConfig:
    """Get configuration for current environment."""
    env_name = os.getenv("ENVIRONMENT", "development").lower()
    
    if env_name not in ENVIRONMENTS:
        print(f"Warning: Unknown environment '{env_name}', defaulting to development")
        env_name = "development"
    
    config = ENVIRONMENTS[env_name]
    print(f"[ENV] Running in {config.name} environment")
    return config


def get_environment_by_name(name: str) -> EnvironmentConfig:
    """Get configuration for a specific environment by name."""
    return ENVIRONMENTS.get(name.lower(), DEVELOPMENT)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Show current environment config
    config = get_environment()
    print(f"\nEnvironment: {config.name}")
    print(f"  Data path: {config.data_path}")
    print(f"  MLflow URI: {config.mlflow_tracking_uri}")
    print(f"  Model stage: {config.model_stage}")
    print(f"  API: {config.api_host}:{config.api_port}")
    print(f"  Prometheus: {config.enable_prometheus}")
    print(f"  Log level: {config.log_level}")
