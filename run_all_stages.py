"""
Run all MLOps pipeline stages sequentially.

Usage:
    python run_all_stages.py           # Run all stages
    python run_all_stages.py --quick   # Skip long-running stages (tuning, LLM)
    python run_all_stages.py --from 3  # Start from stage 3
"""
import subprocess
import sys
import argparse
from pathlib import Path
import time


def run_stage(stage_id: str, stage_name: str, script_name: str, timeout: int = 600):
    """
    Run a single stage.
    
    Args:
        stage_id: Stage identifier (e.g., "0", "3b")
        stage_name: Human-readable stage name
        script_name: Python script filename
        timeout: Maximum execution time in seconds
        
    Returns:
        bool: True if successful
    """
    print("\n" + "=" * 60)
    print(f"STAGE {stage_id}: {stage_name}")
    print("=" * 60)
    
    script_path = Path("src") / script_name
    
    if not script_path.exists():
        print(f"Warning: {script_path} not found, skipping...")
        return True  # Don't fail pipeline for missing optional stages
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=Path.cwd(),
            timeout=timeout
        )
        duration = time.time() - start_time
        print(f"\n[SUCCESS] Stage {stage_id} completed in {duration:.1f}s")
        return True
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Stage {stage_id} timed out after {timeout}s")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] Stage {stage_id} failed with exit code {e.returncode}")
        return False


def main():
    """Run all stages sequentially."""
    parser = argparse.ArgumentParser(description='Run MLOps Pipeline Stages')
    parser.add_argument('--quick', action='store_true', 
                       help='Skip long-running stages (hyperparameter tuning)')
    parser.add_argument('--from', dest='start_from', type=str, default='0',
                       help='Start from specific stage (e.g., 0, 1, 3b)')
    parser.add_argument('--skip-deploy', action='store_true',
                       help='Skip deployment stage (Stage 6 starts a server)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLOps Pipeline - Running All Stages")
    print("=" * 60)
    
    # Define all stages with their timeouts
    stages = [
        ("0", "Data Sampling & Preprocessing", "stage0_data_sampling.py", 300),
        ("1", "Data Ingestion & Quality", "stage1_data_ingestion.py", 300),
        ("2", "Feature Engineering", "stage2_feature_engineering.py", 300),
        ("3b", "Hyperparameter Tuning (Optuna TPE)", "stage3_hyperparameter_tuning.py", 600),
        ("3", "Model Training", "stage3_model_training.py", 300),
        ("4", "Model Evaluation", "stage4_model_evaluation.py", 300),
        ("5", "Model Registry", "stage5_model_registry.py", 300),
        ("6", "Model Deployment (Batch)", "stage6_model_deployment.py", 120),
        ("7", "Monitoring & Observability", "stage7_monitoring.py", 300),
        ("8", "CI/CD Validation", "stage8_cicd.py", 120),
    ]
    
    # Filter stages based on arguments
    if args.quick:
        # Skip hyperparameter tuning in quick mode
        stages = [s for s in stages if s[0] != "3b"]
        print("Quick mode: Skipping hyperparameter tuning")
    
    if args.skip_deploy:
        # Skip deployment stage (it starts a server)
        stages = [s for s in stages if s[0] != "6"]
        print("Skipping deployment stage")
    
    # Find starting point
    start_idx = 0
    for i, (stage_id, _, _, _) in enumerate(stages):
        if stage_id == args.start_from:
            start_idx = i
            break
    
    if start_idx > 0:
        print(f"Starting from Stage {args.start_from}")
        stages = stages[start_idx:]
    
    results = {}
    total_start = time.time()
    
    for stage_id, stage_name, script_name, timeout in stages:
        # Special handling for Stage 6 - run batch mode instead of server
        if stage_id == "6":
            # Run batch inference instead of starting server
            print("\n" + "=" * 60)
            print(f"STAGE {stage_id}: {stage_name}")
            print("=" * 60)
            
            try:
                result = subprocess.run(
                    [sys.executable, "src/stage6_model_deployment.py", "--batch"],
                    check=True,
                    cwd=Path.cwd(),
                    timeout=timeout
                )
                print(f"\n[SUCCESS] Stage {stage_id} completed")
                results[stage_id] = {'name': stage_name, 'success': True}
            except Exception as e:
                print(f"\n[FAILED] Stage {stage_id}: {e}")
                results[stage_id] = {'name': stage_name, 'success': False}
            continue
        
        success = run_stage(stage_id, stage_name, script_name, timeout)
        results[stage_id] = {
            'name': stage_name,
            'success': success
        }
        
        if not success:
            print(f"\nPipeline stopped at Stage {stage_id}")
            break
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    
    for stage_id, result in results.items():
        status = "[OK]" if result['success'] else "[FAIL]"
        print(f"  Stage {stage_id}: {result['name']} - {status}")
    
    print(f"\nTotal duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    
    all_success = all(r['success'] for r in results.values())
    
    if all_success:
        print("\n" + "=" * 60)
        print("[SUCCESS] All stages completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - View MLflow: mlflow ui --backend-store-uri file:./mlruns")
        print("  - Start API: python src/stage6_model_deployment.py")
        print("  - View metrics: http://localhost:8001/metrics")
    else:
        print("\n[FAILED] Some stages failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
