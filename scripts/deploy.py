#!/usr/bin/env python3
"""
Deploy Code Not Models - Deployment Script

This script implements the "Deploy Code Not Models" MLOps principle:
1. Deploy the same code to all environments
2. Each environment trains its own model using environment-specific data
3. Models are registered in environment-specific MLflow instances
4. Model promotion happens through MLflow stages, not file copying

Usage:
    python scripts/deploy.py --env staging
    python scripts/deploy.py --env production --skip-training
    python scripts/deploy.py --env staging --promote-from development
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd: list, env_vars: dict = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with optional environment variables."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    
    return result


def deploy_to_environment(env: str, skip_training: bool = False, promote_from: str = None):
    """
    Deploy code to specified environment.
    
    "Deploy Code Not Models" workflow:
    1. Build Docker image with code
    2. Deploy containers to environment
    3. Run training pipeline (unless skipped)
    4. Optionally promote model from another environment
    """
    
    print(f"\n{'='*60}")
    print(f"DEPLOYING TO: {env.upper()}")
    print(f"{'='*60}\n")
    
    compose_file = f"docker-compose.{env}.yml"
    if env == "development":
        compose_file = "docker-compose.yml"
    
    # Check compose file exists
    if not Path(compose_file).exists():
        raise FileNotFoundError(f"Compose file not found: {compose_file}")
    
    # Step 1: Build Docker image (same code for all environments)
    print("\n[STEP 1] Building Docker image...")
    run_command(["docker-compose", "-f", compose_file, "build", "--no-cache"])
    
    # Step 2: Start services
    print("\n[STEP 2] Starting services...")
    run_command(["docker-compose", "-f", compose_file, "up", "-d"])
    
    # Step 3: Wait for services to be healthy
    print("\n[STEP 3] Waiting for services to be healthy...")
    import time
    time.sleep(30)  # Wait for MLflow to install
    
    # Step 4: Run training pipeline (Deploy Code Not Models)
    if not skip_training:
        print("\n[STEP 4] Running training pipeline in environment...")
        print("         (This trains a NEW model using environment-specific data)")
        
        run_command([
            "docker-compose", "-f", compose_file, "exec", "-T", "mlops",
            "python", "run_all_stages.py", "--skip-deploy"
        ])
    else:
        print("\n[STEP 4] Skipping training (--skip-training flag)")
    
    # Step 5: Optionally promote model from another environment
    if promote_from:
        print(f"\n[STEP 5] Promoting model from {promote_from}...")
        promote_model(from_env=promote_from, to_env=env)
    
    # Step 6: Validate deployment
    print("\n[STEP 6] Validating deployment...")
    run_command([
        "docker-compose", "-f", compose_file, "exec", "-T", "mlops",
        "python", "src/stage8_cicd.py"
    ])
    
    print(f"\n{'='*60}")
    print(f"DEPLOYMENT TO {env.upper()} COMPLETE!")
    print(f"{'='*60}")
    
    # Print access URLs
    ports = {
        "development": {"api": 8000, "mlflow": 5000},
        "staging": {"api": 8000, "mlflow": 5001},
        "production": {"api": 8000, "mlflow": 5002},
    }
    
    p = ports.get(env, ports["development"])
    print(f"\nAccess URLs:")
    print(f"  API:    http://localhost:{p['api']}")
    print(f"  MLflow: http://localhost:{p['mlflow']}")
    print(f"  Health: http://localhost:{p['api']}/health")


def promote_model(from_env: str, to_env: str):
    """
    Promote a model from one environment to another.
    
    This uses MLflow model stages rather than copying files:
    - Development -> Staging: Model moves to "Staging" stage
    - Staging -> Production: Model moves to "Production" stage
    """
    
    print(f"\nPromoting model: {from_env} -> {to_env}")
    
    # In a real implementation, this would:
    # 1. Connect to source MLflow
    # 2. Get the latest model version
    # 3. Export model artifacts
    # 4. Import to target MLflow
    # 5. Register with appropriate stage
    
    promotion_script = f'''
import mlflow
from mlflow.tracking import MlflowClient

# This would connect to the source environment's MLflow
# and promote the model to the target environment

client = MlflowClient()

# Get latest model from source
model_name = "SeismicClassifier"
latest_version = client.get_latest_versions(model_name, stages=["None", "Staging"])[0]

# Transition to new stage
stage_map = {{
    ("development", "staging"): "Staging",
    ("staging", "production"): "Production",
}}

new_stage = stage_map.get(("{from_env}", "{to_env}"), "None")

client.transition_model_version_stage(
    name=model_name,
    version=latest_version.version,
    stage=new_stage
)

print(f"Model {{model_name}} v{{latest_version.version}} promoted to {{new_stage}}")
'''
    
    print("Model promotion would execute the following:")
    print(promotion_script)
    print("\n(Actual promotion requires MLflow connectivity between environments)")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Code Not Models - Environment Deployment Script"
    )
    parser.add_argument(
        "--env", "-e",
        choices=["development", "staging", "production"],
        default="development",
        help="Target environment"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training pipeline (use existing model)"
    )
    parser.add_argument(
        "--promote-from",
        choices=["development", "staging"],
        help="Promote model from another environment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print(f"DRY RUN: Would deploy to {args.env}")
        print(f"  Skip training: {args.skip_training}")
        print(f"  Promote from: {args.promote_from}")
        return
    
    try:
        deploy_to_environment(
            env=args.env,
            skip_training=args.skip_training,
            promote_from=args.promote_from
        )
    except Exception as e:
        print(f"\n[ERROR] Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
