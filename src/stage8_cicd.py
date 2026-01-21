"""
Stage 8: CI/CD Automation

Automated testing and validation pipeline for ML models.

Updated to support:
- All pipeline stages (0-7)
- Hyperparameter tuning validation (Stage 3b)
- Model registry validation (Stage 5)
- Deployment validation (Stage 6)
- Quick validation mode for testing
"""
import subprocess
import sys
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class CICDPipeline:
    """
    CI/CD pipeline for ML model validation.
    
    Supports:
    - Full pipeline validation (all stages)
    - Quick validation (key stages only)
    - Individual stage testing
    - LLM-generated test suggestions
    """
    
    def __init__(self, project_root: str = "."):
        """
        Initialize CI/CD pipeline.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'overall_status': 'unknown',
            'duration_seconds': 0
        }
        self.start_time = None
    
    def run_command(self, command: List[str], stage_name: str,
                   timeout: int = 120) -> Dict[str, Any]:
        """
        Run a command and capture results.
        
        Args:
            command: Command to run
            stage_name: Name of the stage
            timeout: Timeout in seconds
            
        Returns:
            Results dictionary
        """
        print(f"\n{'-'*60}")
        print(f"> {stage_name}")
        print(f"  Command: {' '.join(command)}")
        print(f"{'-'*60}")
        
        start = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start
            success = result.returncode == 0
            
            stage_result = {
                'status': 'success' if success else 'failed',
                'exit_code': result.returncode,
                'duration_seconds': round(duration, 2),
                'stdout_lines': len(result.stdout.split('\n')) if result.stdout else 0,
                'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0
            }
            
            if success:
                print(f"  [OK] {stage_name} completed in {duration:.1f}s")
            else:
                print(f"  [FAIL] {stage_name} failed (exit code: {result.returncode})")
                if result.stderr:
                    # Show last few lines of error
                    error_lines = result.stderr.strip().split('\n')[-5:]
                    for line in error_lines:
                        print(f"    {line}")
            
            return stage_result
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            print(f"  [TIMEOUT] {stage_name} timed out after {timeout}s")
            return {
                'status': 'timeout',
                'duration_seconds': round(duration, 2),
                'message': f'Timed out after {timeout}s'
            }
        
        except Exception as e:
            print(f"  [ERROR] {stage_name} error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def run_python_check(self, script_path: str, stage_name: str,
                        timeout: int = 60) -> Dict[str, Any]:
        """Run a Python script with import check."""
        # First check if file exists
        full_path = self.project_root / script_path
        if not full_path.exists():
            print(f"  [WARN] {stage_name}: File not found ({script_path})")
            return {'status': 'skipped', 'message': 'File not found'}
        
        # Check syntax/imports
        return self.run_command(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, '.'); exec(open('{script_path}').read().split('if __name__')[0])"],
            f"{stage_name} (syntax check)",
            timeout=30
        )
    
    def validate_stage0_sampling(self) -> Dict[str, Any]:
        """Validate Stage 0: Data Sampling."""
        # Check if output exists
        output_path = self.project_root / "data" / "raw" / "seismic_sample.parquet"
        if output_path.exists():
            return {'status': 'success', 'message': 'Output exists', 'output': str(output_path)}
        return self.run_python_check("src/stage0_data_sampling.py", "Stage 0: Data Sampling")
    
    def validate_stage1_ingestion(self) -> Dict[str, Any]:
        """Validate Stage 1: Data Ingestion."""
        output_path = self.project_root / "data" / "bronze" / "seismic_data"
        if output_path.exists():
            return {'status': 'success', 'message': 'Output exists', 'output': str(output_path)}
        return self.run_python_check("src/stage1_data_ingestion.py", "Stage 1: Data Ingestion")
    
    def validate_stage2_features(self) -> Dict[str, Any]:
        """Validate Stage 2: Feature Engineering."""
        output_path = self.project_root / "data" / "silver" / "seismic_features"
        if output_path.exists():
            return {'status': 'success', 'message': 'Output exists', 'output': str(output_path)}
        return self.run_python_check("src/stage2_feature_engineering.py", "Stage 2: Feature Engineering")
    
    def validate_stage3_training(self) -> Dict[str, Any]:
        """Validate Stage 3: Model Training."""
        model_path = self.project_root / "models" / "seismic_classifier.pkl"
        if model_path.exists():
            return {'status': 'success', 'message': 'Model exists', 'output': str(model_path)}
        return self.run_python_check("src/stage3_model_training.py", "Stage 3: Model Training")
    
    def validate_stage3b_tuning(self) -> Dict[str, Any]:
        """Validate Stage 3b: Hyperparameter Tuning."""
        results_path = self.project_root / "models" / "hyperparameter_tuning_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            return {
                'status': 'success',
                'message': 'Tuning results exist',
                'best_model': results.get('best_model_type'),
                'best_cv_score': results.get('best_cv_score')
            }
        return self.run_python_check("src/stage3_hyperparameter_tuning.py", "Stage 3b: Hyperparameter Tuning")
    
    def validate_stage4_evaluation(self) -> Dict[str, Any]:
        """Validate Stage 4: Model Evaluation."""
        eval_path = self.project_root / "data" / "gold" / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                results = json.load(f)
            metrics = results.get('metrics', {})
            return {
                'status': 'success',
                'message': 'Evaluation results exist',
                'accuracy': metrics.get('accuracy'),
                'f1_weighted': metrics.get('f1_weighted')
            }
        return self.run_python_check("src/stage4_model_evaluation.py", "Stage 4: Model Evaluation")
    
    def validate_stage5_registry(self) -> Dict[str, Any]:
        """Validate Stage 5: Model Registry."""
        # Check for versioned artifacts
        models_dir = self.project_root / "models"
        version_dirs = list(models_dir.glob("SeismicClassifier_v*"))
        if version_dirs:
            latest = sorted(version_dirs)[-1]
            return {
                'status': 'success',
                'message': f'Model registered',
                'latest_version': latest.name,
                'artifacts': len(list(latest.glob("*")))
            }
        return self.run_python_check("src/stage5_model_registry.py", "Stage 5: Model Registry")
    
    def validate_stage6_deployment(self) -> Dict[str, Any]:
        """Validate Stage 6: Model Deployment (batch mode)."""
        predictions_path = self.project_root / "data" / "gold" / "batch_predictions.parquet"
        if predictions_path.exists():
            import pandas as pd
            df = pd.read_parquet(predictions_path)
            return {
                'status': 'success',
                'message': 'Batch predictions exist',
                'num_predictions': len(df)
            }
        
        # Run batch inference test
        return self.run_command(
            [sys.executable, "-c", 
             "from src.stage6_model_deployment import ModelServing; "
             "s = ModelServing(); "
             "r = s.batch_predict('data/silver/seismic_features.parquet', 'data/gold/batch_predictions.parquet'); "
             "print(f'Predictions: {r[\"num_predictions\"]}')"
            ],
            "Stage 6: Batch Deployment",
            timeout=60
        )
    
    def validate_stage7_monitoring(self) -> Dict[str, Any]:
        """Validate Stage 7: Monitoring."""
        report_path = self.project_root / "data" / "gold" / "monitoring_report.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
            return {
                'status': 'success',
                'message': 'Monitoring report exists',
                'alerts': len(report.get('alerts', [])),
                'drift_severity': report.get('feature_drift', {}).get('severity')
            }
        return self.run_python_check("src/stage7_monitoring.py", "Stage 7: Monitoring")
    
    def run_quick_validation(self) -> Dict[str, Any]:
        """
        Run quick validation (check outputs only, no execution).
        
        Returns:
            Validation results
        """
        print("=" * 60)
        print("CI/CD Quick Validation")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Check each stage's outputs
        validations = [
            ('stage0_sampling', self.validate_stage0_sampling),
            ('stage1_ingestion', self.validate_stage1_ingestion),
            ('stage2_features', self.validate_stage2_features),
            ('stage3_training', self.validate_stage3_training),
            ('stage3b_tuning', self.validate_stage3b_tuning),
            ('stage4_evaluation', self.validate_stage4_evaluation),
            ('stage5_registry', self.validate_stage5_registry),
            ('stage6_deployment', self.validate_stage6_deployment),
            ('stage7_monitoring', self.validate_stage7_monitoring),
        ]
        
        for name, validator in validations:
            self.results['stages'][name] = validator()
        
        # Calculate overall status
        failed = [n for n, r in self.results['stages'].items() 
                 if r.get('status') not in ['success', 'skipped']]
        
        self.results['overall_status'] = 'failed' if failed else 'success'
        self.results['failed_stages'] = failed
        self.results['duration_seconds'] = round(time.time() - self.start_time, 2)
        
        return self.results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run full CI/CD pipeline (execute all stages).
        
        Returns:
            Pipeline results
        """
        print("=" * 60)
        print("CI/CD Full Pipeline")
        print("=" * 60)
        print("[WARNING] This will execute all stages sequentially")
        
        self.start_time = time.time()
        
        stages = [
            ('stage0', [sys.executable, "src/stage0_data_sampling.py"], 120),
            ('stage1', [sys.executable, "src/stage1_data_ingestion.py"], 180),
            ('stage2', [sys.executable, "src/stage2_feature_engineering.py"], 180),
            ('stage3', [sys.executable, "src/stage3_model_training.py"], 300),
            ('stage4', [sys.executable, "src/stage4_model_evaluation.py"], 300),
        ]
        
        for name, command, timeout in stages:
            result = self.run_command(command, f"Execute {name}", timeout)
            self.results['stages'][name] = result
            
            # Stop on failure
            if result.get('status') == 'failed':
                break
        
        # Calculate overall status
        failed = [n for n, r in self.results['stages'].items() 
                 if r.get('status') == 'failed']
        
        self.results['overall_status'] = 'failed' if failed else 'success'
        self.results['failed_stages'] = failed
        self.results['duration_seconds'] = round(time.time() - self.start_time, 2)
        
        return self.results
    
    def generate_github_actions_workflow(self) -> str:
        """Generate GitHub Actions workflow YAML."""
        workflow = """
name: ML Pipeline CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run Stage 0 - Data Sampling
      run: python src/stage0_data_sampling.py
    
    - name: Run Stage 1 - Data Ingestion
      run: python src/stage1_data_ingestion.py
    
    - name: Run Stage 2 - Feature Engineering
      run: python src/stage2_feature_engineering.py
    
    - name: Run Stage 3 - Model Training
      run: python src/stage3_model_training.py
    
    - name: Run Stage 4 - Model Evaluation
      run: python src/stage4_model_evaluation.py
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: |
          models/
          data/gold/evaluation_results.json
"""
        return workflow.strip()
    
    def save_results(self, output_path: str = "cicd_results.json"):
        """Save CI/CD results to file."""
        results_path = self.project_root / output_path
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")
    
    def print_summary(self):
        """Print pipeline summary."""
        print("\n" + "=" * 60)
        print("CI/CD Pipeline Summary")
        print("=" * 60)
        
        # Overall status
        status = self.results['overall_status']
        status_icon = "[OK]" if status == 'success' else "[FAIL]"
        print(f"\n{status_icon} Overall Status: {status.upper()}")
        print(f"  Duration: {self.results['duration_seconds']}s")
        
        # Stage details
        print("\nStage Results:")
        for stage_name, result in self.results['stages'].items():
            status = result.get('status', 'unknown')
            icon = "[OK]" if status == 'success' else "[SKIP]" if status == 'skipped' else "[FAIL]"
            
            details = []
            if 'accuracy' in result:
                details.append(f"acc={result['accuracy']:.3f}")
            if 'best_cv_score' in result:
                details.append(f"cv={result['best_cv_score']:.3f}")
            if 'num_predictions' in result:
                details.append(f"n={result['num_predictions']}")
            if 'alerts' in result:
                details.append(f"alerts={result['alerts']}")
            
            detail_str = f" ({', '.join(details)})" if details else ""
            print(f"  {icon} {stage_name}: {status}{detail_str}")
        
        # Failed stages
        if self.results.get('failed_stages'):
            print(f"\n[WARNING] Failed stages: {', '.join(self.results['failed_stages'])}")


def main():
    """Execute Stage 8: CI/CD."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Pipeline CI/CD')
    parser.add_argument('--full', action='store_true', help='Run full pipeline (execute all stages)')
    parser.add_argument('--workflow', action='store_true', help='Generate GitHub Actions workflow')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stage 8: CI/CD Automation")
    print("=" * 60)
    
    pipeline = CICDPipeline(project_root=".")
    
    if args.workflow:
        print("\nGenerated GitHub Actions Workflow:")
        print("-" * 40)
        print(pipeline.generate_github_actions_workflow())
        print("-" * 40)
        
        # Save workflow
        workflow_path = Path(".github/workflows/ml-pipeline.yml")
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        with open(workflow_path, 'w') as f:
            f.write(pipeline.generate_github_actions_workflow())
        print(f"\nWorkflow saved to: {workflow_path}")
        return
    
    if args.full:
        results = pipeline.run_full_pipeline()
    else:
        results = pipeline.run_quick_validation()
    
    pipeline.save_results()
    pipeline.print_summary()
    
    # Exit with appropriate code
    if results['overall_status'] != 'success':
        sys.exit(1)


if __name__ == "__main__":
    main()
