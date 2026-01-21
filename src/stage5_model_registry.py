"""
Stage 5: Model Registry & Artifact Management

Register model in MLflow, version artifacts, generate documentation.

Updated to support:
- Models trained with tuned hyperparameters (Stage 3b Optuna TPE)
- 40 features (8 handcrafted + 32 PCA embeddings)
- Model staging workflow (None -> Staging -> Production)
- Comprehensive artifact management
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import json
from typing import Dict, Any, Optional, List
import pickle
import shutil
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class ModelRegistryManager:
    """
    Manage model registry and artifacts with MLflow.
    
    Supports:
    - Model versioning and registration
    - Stage transitions (None -> Staging -> Production)
    - Artifact management with versioning
    - LLM-generated documentation
    - Hyperparameter tracking from Optuna TPE
    """
    
    def __init__(self, models_dir: str = "models",
                 mlflow_uri: str = "file:./mlruns"):
        """
        Initialize registry manager.
        
        Args:
            models_dir: Directory for model artifacts
            mlflow_uri: MLflow tracking URI
        """
        self.models_dir = Path(models_dir)
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Load tuning results if available
        self.tuning_results = self._load_tuning_results()
    
    def _load_tuning_results(self) -> Optional[Dict]:
        """Load hyperparameter tuning results from Stage 3b."""
        tuning_path = self.models_dir / "hyperparameter_tuning_results.json"
        if tuning_path.exists():
            with open(tuning_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_latest_model_run(self, experiment_name: str = "seismic_classification") -> Any:
        """
        Get the latest model run from MLflow.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Latest run info
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Try alternative experiment name
            experiment = mlflow.get_experiment_by_name("seismic_classification_test")
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Get all runs, ordered by start time
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("No runs found in experiment")
        
        latest_run = runs[0]
        return latest_run
    
    def get_best_model_run(self, experiment_name: str = "seismic_classification",
                          metric: str = "test_f1_weighted") -> Any:
        """
        Get the best model run based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize (default: test_f1_weighted)
            
        Returns:
            Best run info
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment = mlflow.get_experiment_by_name("seismic_classification_test")
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Get all runs, ordered by metric
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("No runs found in experiment")
        
        return runs[0]
    
    def register_model_version(self, model_name: str = "SeismicClassifier",
                              run_id: str = None,
                              use_best: bool = False) -> str:
        """
        Register model version in MLflow Model Registry.
        
        Args:
            model_name: Name of the model
            run_id: Run ID (if None, uses latest or best)
            use_best: If True, use best run by F1 score
            
        Returns:
            Model version number
        """
        if run_id is None:
            if use_best:
                run = self.get_best_model_run()
            else:
                run = self.get_latest_model_run()
            run_id = run.info.run_id
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            print(f"Model registered: {model_name} v{model_version.version}")
            return model_version.version
        except Exception as e:
            print(f"Model registration note: {e}")
            # Get latest version if already registered
            try:
                versions = self.client.get_latest_versions(model_name, stages=["None"])
                if versions:
                    return versions[0].version
            except:
                pass
            
            # Create new version
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            return model_version.version
    
    def transition_model_stage(self, model_name: str, version: str,
                               stage: str = "Staging") -> None:
        """
        Transition model to a new stage.
        
        Args:
            model_name: Model name
            version: Model version
            stage: Target stage (None, Staging, Production, Archived)
        """
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} v{version} transitioned to: {stage}")
    
    def get_production_model(self, model_name: str = "SeismicClassifier") -> Optional[Any]:
        """
        Get the current production model version.
        
        Args:
            model_name: Model name
            
        Returns:
            Production model version info or None
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return versions[0]
        except Exception as e:
            print(f"No production model found: {e}")
        return None
    
    def save_artifacts(self, model_name: str, version: str,
                      artifacts: Dict[str, str]) -> str:
        """
        Save model artifacts with versioning.
        
        Args:
            model_name: Model name
            version: Model version
            artifacts: Dictionary of artifact_name -> artifact_path
            
        Returns:
            Artifact directory path
        """
        artifact_dir = self.models_dir / f"{model_name}_v{version}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy artifacts
        for artifact_name, artifact_path in artifacts.items():
            src_path = Path(artifact_path)
            if src_path.exists():
                dst_path = artifact_dir / artifact_name
                if src_path.is_file():
                    shutil.copy(src_path, dst_path)
                else:
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"  Copied: {artifact_name}")
            else:
                print(f"  Warning: {artifact_path} not found")
        
        # Save metadata including tuning info
        metadata = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'artifacts': list(artifacts.keys()),
            'num_features': 40,
            'feature_types': {
                'handcrafted': 8,
                'embeddings': 32
            }
        }
        
        # Add tuning info if available
        if self.tuning_results:
            metadata['hyperparameter_tuning'] = {
                'method': 'Optuna TPE',
                'best_model_type': self.tuning_results.get('best_model_type'),
                'best_cv_score': self.tuning_results.get('best_cv_score'),
                'best_params': self.tuning_results.get('best_params')
            }
        
        metadata_path = artifact_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Artifacts saved to: {artifact_dir}")
        return str(artifact_dir)
    
    def llm_generate_documentation(self, model_info: Dict[str, Any],
                                   artifact_list: list) -> Optional[str]:
        """
        Generate model documentation using LLM.
        
        Args:
            model_info: Model information dictionary
            artifact_list: List of artifact names
            
        Returns:
            Documentation text or None
        """
        if not OLLAMA_AVAILABLE:
            return None
        
        try:
            # Prepare compact info for LLM
            compact_info = {
                'model_type': model_info.get('model_type', 'Unknown'),
                'num_features': 40,
                'feature_breakdown': '8 handcrafted + 32 PCA embeddings',
                'metrics': {k: round(v, 4) for k, v in model_info.get('metrics', {}).items() 
                           if isinstance(v, (int, float))},
                'artifacts': artifact_list[:10]
            }
            
            # Add tuning info
            if self.tuning_results:
                compact_info['hyperparameter_tuning'] = {
                    'method': 'Optuna TPE',
                    'best_cv_score': round(self.tuning_results.get('best_cv_score', 0), 4)
                }
            
            prompt = f"""
            Create model documentation for a seismic classification model. Return JSON with:
            1. Purpose (1 sentence about seismic trace classification)
            2. Architecture (model type, features, hyperparameter tuning method)
            3. Performance (key metrics)
            4. Artifacts (list of saved files)
            5. Usage (how to load and use the model)
            
            Model Info: {json.dumps(compact_info)}
            """
            
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=prompt,
                options={"temperature": 0.2}
            )
            
            return response['response']
        except Exception as e:
            print(f"LLM documentation generation failed: {e}")
            return None
    
    def update_model_description(self, model_name: str, version: str,
                                 description: str) -> None:
        """
        Update model version description.
        
        Args:
            model_name: Model name
            version: Model version
            description: Description text
        """
        self.client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        print(f"Updated description for {model_name} v{version}")
    
    def list_model_versions(self, model_name: str = "SeismicClassifier") -> List[Dict]:
        """
        List all versions of a model.
        
        Args:
            model_name: Model name
            
        Returns:
            List of version info dictionaries
        """
        try:
            versions = []
            for stage in ["None", "Staging", "Production", "Archived"]:
                stage_versions = self.client.get_latest_versions(model_name, stages=[stage])
                for v in stage_versions:
                    versions.append({
                        'version': v.version,
                        'stage': v.current_stage,
                        'status': v.status,
                        'creation_timestamp': v.creation_timestamp
                    })
            return versions
        except Exception as e:
            print(f"Error listing versions: {e}")
            return []


def main():
    """Execute Stage 5: Model Registry."""
    print("=" * 60)
    print("Stage 5: Model Registry & Artifact Management")
    print("=" * 60)
    
    # Initialize registry manager
    registry = ModelRegistryManager(models_dir="models")
    
    # Get latest run
    print("\nGetting latest model run...")
    try:
        run = registry.get_latest_model_run()
        run_id = run.info.run_id
        print(f"Latest run ID: {run_id}")
    except Exception as e:
        print(f"Error getting run: {e}")
        print("Using local model files instead...")
        run = None
        run_id = None
    
    # Get model metrics and params
    if run:
        metrics = run.data.metrics
        params = run.data.params
        
        model_info = {
            'model_type': params.get('model_type', 'Unknown'),
            'metrics': metrics,
            'parameters': params,
            'hyperparameters_source': params.get('hyperparameters_source', 'default')
        }
        
        print(f"\nModel Info:")
        print(f"  Type: {model_info['model_type']}")
        print(f"  Hyperparameters: {model_info['hyperparameters_source']}")
        if 'test_accuracy' in metrics:
            print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        if 'test_f1_weighted' in metrics:
            print(f"  Test F1 (weighted): {metrics['test_f1_weighted']:.4f}")
    else:
        model_info = {'model_type': 'Unknown', 'metrics': {}, 'parameters': {}}
    
    # Register model version
    print("\nRegistering model version...")
    try:
        version = registry.register_model_version(
            model_name="SeismicClassifier",
            run_id=run_id
        )
    except Exception as e:
        print(f"Registration error: {e}")
        version = "1"
    
    # Save artifacts
    print("\nSaving model artifacts...")
    artifacts = {
        'model.pkl': 'models/seismic_classifier.pkl',
        'model_features.json': 'models/seismic_classifier_features.json',
        'feature_scaler.pkl': 'data/silver/feature_scaler.pkl',
        'evaluation_results.json': 'data/gold/evaluation_results.json',
        'hyperparameter_tuning_results.json': 'models/hyperparameter_tuning_results.json'
    }
    
    artifact_dir = registry.save_artifacts(
        model_name="SeismicClassifier",
        version=version,
        artifacts=artifacts
    )
    
    # Generate documentation
    print("\nGenerating model documentation...")
    documentation = registry.llm_generate_documentation(
        model_info,
        list(artifacts.keys())
    )
    
    if documentation:
        print("\nLLM-Generated Documentation:")
        print("-" * 40)
        print(documentation[:500] + "..." if len(documentation) > 500 else documentation)
        print("-" * 40)
        
        # Update model description
        registry.update_model_description(
            model_name="SeismicClassifier",
            version=version,
            description=documentation[:1000]  # Limit description length
        )
    else:
        # Create basic description
        test_acc = model_info.get('metrics', {}).get('test_accuracy', 'N/A')
        test_f1 = model_info.get('metrics', {}).get('test_f1_weighted', 'N/A')
        
        if isinstance(test_acc, (int, float)):
            test_acc_str = f"{test_acc:.4f}"
        else:
            test_acc_str = str(test_acc)
        
        if isinstance(test_f1, (int, float)):
            test_f1_str = f"{test_f1:.4f}"
        else:
            test_f1_str = str(test_f1)
        
        hp_source = model_info.get('hyperparameters_source', 'default')
        
        basic_desc = f"""
Seismic Classification Model (v{version})

Model Type: {model_info['model_type']}
Features: 40 (8 handcrafted + 32 PCA embeddings)
Hyperparameters: {hp_source}
Test Accuracy: {test_acc_str}
Test F1-score: {test_f1_str}

Artifacts: {', '.join(artifacts.keys())}
        """
        
        registry.update_model_description(
            model_name="SeismicClassifier",
            version=version,
            description=basic_desc
        )
    
    # List all versions
    print("\nModel Versions:")
    versions = registry.list_model_versions()
    for v in versions:
        print(f"  v{v['version']}: {v['stage']}")
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Stage 5 complete!")
    print(f"{'='*60}")
    print(f"  Model: SeismicClassifier")
    print(f"  Version: {version}")
    print(f"  Artifacts: {artifact_dir}")
    print(f"  Features: 40 (8 handcrafted + 32 embeddings)")
    if registry.tuning_results:
        print(f"  Hyperparameters: Optuna TPE optimized")


if __name__ == "__main__":
    main()
