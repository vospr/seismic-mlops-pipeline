"""
Stage 3b: Hyperparameter Tuning with Optuna

Bayesian hyperparameter optimization using:
- Optuna with TPE (Tree of Parzen Estimators) sampler
- MLflow integration for experiment tracking
- Cross-validation for robust evaluation
- Early stopping (pruning) of unpromising trials

This module can be used standalone or integrated with Stage 3.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, List, Tuple, Callable
import pickle
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import MLflowCallback

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

import mlflow
import mlflow.sklearn

from deltalake import DeltaTable

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class HyperparameterTuner:
    """
    Hyperparameter optimization using Optuna with TPE sampler.
    
    Features:
    - TPE (Tree of Parzen Estimators) for efficient Bayesian optimization
    - Cross-validation for robust evaluation
    - MLflow integration for experiment tracking
    - Pruning of unpromising trials
    - Support for multiple model types
    """
    
    def __init__(self, 
                 input_dir: str = "data/silver",
                 output_dir: str = "models",
                 mlflow_uri: str = "file:./mlruns",
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            input_dir: Directory with Stage 2 features
            output_dir: Directory for model artifacts
            mlflow_uri: MLflow tracking URI
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("seismic_hyperparameter_tuning")
        
        self.feature_names: List[str] = []
        self.best_params: Dict[str, Any] = {}
        self.best_model = None
        self.study: Optional[optuna.Study] = None
    
    def load_features(self, table_name: str = "seismic_features",
                     use_embeddings: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels from Delta/Parquet.
        
        Args:
            table_name: Name of the features table
            use_embeddings: Whether to include embedding features
            
        Returns:
            Tuple of (X, y)
        """
        # Try Delta Lake first
        delta_path = self.input_dir / table_name
        if delta_path.exists():
            print(f"Loading features from Delta Lake: {delta_path}")
            dt = DeltaTable(str(delta_path))
            df = dt.to_pandas()
        else:
            parquet_path = self.input_dir / f"{table_name}.parquet"
            if parquet_path.exists():
                print(f"Loading features from Parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)
            else:
                raise FileNotFoundError(f"Features not found in {self.input_dir}")
        
        # Define feature columns
        handcrafted_cols = [
            'mean_amplitude', 'std_amplitude', 'min_amplitude', 'max_amplitude',
            'rms_amplitude', 'energy', 'zero_crossings', 'dominant_frequency'
        ]
        embedding_cols = [f'embedding_{i}' for i in range(32)]
        
        if use_embeddings:
            available_embedding_cols = [col for col in embedding_cols if col in df.columns]
            feature_cols = handcrafted_cols + available_embedding_cols
        else:
            feature_cols = handcrafted_cols
        
        # Use scaled features if available
        scaled_cols = [f"{col}_scaled" for col in feature_cols]
        if all(col in df.columns for col in scaled_cols):
            X = df[scaled_cols].values
            self.feature_names = scaled_cols
        else:
            X = df[feature_cols].values
            self.feature_names = feature_cols
        
        y = df['class_label'].values
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def _create_objective(self, X: np.ndarray, y: np.ndarray, 
                         model_type: str) -> Callable:
        """
        Create Optuna objective function for a specific model type.
        
        Args:
            X: Features
            y: Labels
            model_type: Type of model to optimize
            
        Returns:
            Objective function for Optuna
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        scorer = make_scorer(f1_score, average='weighted')
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            
            if model_type == "RandomForest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == "GradientBoosting":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': self.random_state
                }
                model = GradientBoostingClassifier(**params)
                
            elif model_type == "LogisticRegression":
                params = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': 'saga',  # Supports both l1 and l2
                    'max_iter': 2000,
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                    'random_state': self.random_state
                }
                model = LogisticRegression(**params)
                
            elif model_type == "DecisionTree":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                    'random_state': self.random_state
                }
                model = DecisionTreeClassifier(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cross-validation score
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                return scores.mean()
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        return objective
    
    def optimize(self, X: np.ndarray, y: np.ndarray,
                model_type: str = "RandomForest",
                direction: str = "maximize") -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Optuna with TPE.
        
        Args:
            X: Features
            y: Labels
            model_type: Type of model to optimize
            direction: Optimization direction ("maximize" for metrics like F1)
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*60}")
        print(f"Hyperparameter Optimization: {model_type}")
        print(f"{'='*60}")
        print(f"Sampler: TPE (Tree of Parzen Estimators)")
        print(f"Trials: {self.n_trials}")
        print(f"CV Folds: {self.cv_folds}")
        print(f"Metric: F1-score (weighted)")
        
        # Create TPE sampler
        sampler = TPESampler(
            seed=self.random_state,
            n_startup_trials=10,  # Random trials before TPE kicks in
            multivariate=True     # Consider parameter correlations
        )
        
        # Create pruner for early stopping
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0
        )
        
        # Create study
        study_name = f"{model_type}_optimization"
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Create objective function
        objective = self._create_objective(X, y, model_type)
        
        # MLflow callback for logging
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="cv_f1_weighted",
            create_experiment=False
        )
        
        # Run optimization
        print(f"\nStarting optimization...")
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
            n_jobs=1  # Sequential for stability
        )
        
        # Get best results
        self.best_params = self.study.best_params
        best_value = self.study.best_value
        
        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best F1-score (CV): {best_value:.4f}")
        print(f"\nBest Hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Create and store best model
        self.best_model = self._create_model_with_params(model_type, self.best_params)
        
        # Optimization summary
        results = {
            'model_type': model_type,
            'best_params': self.best_params,
            'best_cv_score': float(best_value),
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'sampler': 'TPE',
            'optimization_history': [
                {'trial': i, 'value': trial.value}
                for i, trial in enumerate(self.study.trials)
                if trial.value is not None
            ]
        }
        
        return results
    
    def _create_model_with_params(self, model_type: str, 
                                  params: Dict[str, Any]) -> Any:
        """Create model instance with optimized parameters."""
        params_copy = params.copy()
        params_copy['random_state'] = self.random_state
        
        if model_type == "RandomForest":
            params_copy['n_jobs'] = -1
            return RandomForestClassifier(**params_copy)
        elif model_type == "GradientBoosting":
            return GradientBoostingClassifier(**params_copy)
        elif model_type == "LogisticRegression":
            params_copy['solver'] = 'saga'
            params_copy['max_iter'] = 2000
            return LogisticRegression(**params_copy)
        elif model_type == "DecisionTree":
            return DecisionTreeClassifier(**params_copy)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def optimize_multiple_models(self, X: np.ndarray, y: np.ndarray,
                                model_types: List[str] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for multiple model types and select best.
        
        Args:
            X: Features
            y: Labels
            model_types: List of model types to optimize
            
        Returns:
            Dictionary with results for all models and best overall
        """
        if model_types is None:
            model_types = ["RandomForest", "GradientBoosting", "LogisticRegression"]
        
        all_results = {}
        best_overall_score = -1
        best_overall_model_type = None
        
        for model_type in model_types:
            print(f"\n{'#'*60}")
            print(f"Optimizing: {model_type}")
            print(f"{'#'*60}")
            
            results = self.optimize(X, y, model_type)
            all_results[model_type] = results
            
            if results['best_cv_score'] > best_overall_score:
                best_overall_score = results['best_cv_score']
                best_overall_model_type = model_type
        
        # Set best model
        self.best_params = all_results[best_overall_model_type]['best_params']
        self.best_model = self._create_model_with_params(
            best_overall_model_type, self.best_params
        )
        
        print(f"\n{'='*60}")
        print("Multi-Model Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best Model: {best_overall_model_type}")
        print(f"Best CV F1-score: {best_overall_score:.4f}")
        
        return {
            'all_results': all_results,
            'best_model_type': best_overall_model_type,
            'best_cv_score': best_overall_score,
            'best_params': self.best_params
        }
    
    def train_best_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Train the best model on full training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of metrics
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run optimize() first.")
        
        print("\nTraining best model on full training data...")
        self.best_model.fit(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score
        
        y_train_pred = self.best_model.predict(X_train)
        y_val_pred = self.best_model.predict(X_val)
        
        metrics = {
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'train_f1_weighted': float(f1_score(y_train, y_train_pred, average='weighted')),
            'val_accuracy': float(accuracy_score(y_val, y_val_pred)),
            'val_f1_weighted': float(f1_score(y_val, y_val_pred, average='weighted'))
        }
        
        print(f"  Training accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Training F1 (weighted): {metrics['train_f1_weighted']:.4f}")
        print(f"  Validation accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  Validation F1 (weighted): {metrics['val_f1_weighted']:.4f}")
        
        return metrics
    
    def save_results(self, results: Dict[str, Any], 
                    filename: str = "hyperparameter_tuning_results.json"):
        """Save optimization results to JSON."""
        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {results_path}")
        return results_path
    
    def save_best_model(self, model_name: str = "seismic_classifier_tuned"):
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model found. Run optimize() first.")
        
        model_path = self.output_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save feature names and params
        info_path = self.output_dir / f"{model_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'num_features': len(self.feature_names),
                'best_params': self.best_params,
                'model_type': type(self.best_model).__name__
            }, f, indent=2)
        
        print(f"Best model saved to: {model_path}")
        return model_path
    
    def get_optimization_insights(self) -> Optional[str]:
        """Use LLM to generate insights from optimization results."""
        if not OLLAMA_AVAILABLE or self.study is None:
            return None
        
        try:
            # Prepare summary
            summary = {
                'best_score': round(self.study.best_value, 4),
                'best_params': self.best_params,
                'n_trials': len(self.study.trials),
                'score_improvement': round(
                    self.study.best_value - self.study.trials[0].value 
                    if self.study.trials[0].value else 0, 4
                )
            }
            
            prompt = f"""
            Analyze these hyperparameter optimization results for a seismic classification model:
            
            {json.dumps(summary, indent=2)}
            
            Provide brief insights on:
            1. Are the optimized hyperparameters reasonable?
            2. What do the parameter values suggest about the data?
            3. Any recommendations for further improvement?
            
            Keep response concise (3-4 sentences per point).
            """
            
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=prompt,
                options={"temperature": 0.2}
            )
            
            return response['response']
        except Exception as e:
            print(f"LLM insights generation failed: {e}")
            return None


def main():
    """Execute Stage 3b: Hyperparameter Tuning."""
    print("=" * 60)
    print("Stage 3b: Hyperparameter Tuning with Optuna (TPE)")
    print("=" * 60)
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        input_dir="data/silver",
        output_dir="models",
        n_trials=30,  # Reduced for faster demo
        cv_folds=5,
        random_state=42
    )
    
    # Load features
    print("\nLoading features...")
    X, y = tuner.load_features(use_embeddings=True)
    
    # Split data (same as Stage 3)
    from sklearn.model_selection import train_test_split
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Option 1: Optimize single model type
    # results = tuner.optimize(X_train, y_train, model_type="RandomForest")
    
    # Option 2: Optimize multiple models and select best
    results = tuner.optimize_multiple_models(
        X_train, y_train,
        model_types=["RandomForest", "GradientBoosting", "LogisticRegression"]
    )
    
    # Train best model on full training data
    print("\n" + "=" * 60)
    print("Training Best Model")
    print("=" * 60)
    metrics = tuner.train_best_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    y_test_pred = tuner.best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 (weighted): {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred,
                               target_names=['Normal', 'Anomaly', 'Boundary']))
    
    # Get LLM insights
    print("\nGenerating LLM insights...")
    insights = tuner.get_optimization_insights()
    if insights:
        print("\nLLM Optimization Insights:")
        print("-" * 40)
        print(insights)
        print("-" * 40)
    
    # Save results
    results['final_metrics'] = {
        'train_accuracy': metrics['train_accuracy'],
        'train_f1_weighted': metrics['train_f1_weighted'],
        'val_accuracy': metrics['val_accuracy'],
        'val_f1_weighted': metrics['val_f1_weighted'],
        'test_accuracy': float(test_acc),
        'test_f1_weighted': float(test_f1)
    }
    results['llm_insights'] = insights
    
    tuner.save_results(results)
    tuner.save_best_model()
    
    # Log to MLflow
    with mlflow.start_run(run_name="best_tuned_model"):
        mlflow.log_params(tuner.best_params)
        mlflow.log_params({
            'model_type': results['best_model_type'],
            'n_trials': tuner.n_trials,
            'cv_folds': tuner.cv_folds,
            'sampler': 'TPE'
        })
        mlflow.log_metrics({
            'best_cv_f1': results['best_cv_score'],
            'test_accuracy': test_acc,
            'test_f1_weighted': test_f1
        })
        mlflow.sklearn.log_model(tuner.best_model, "model")
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Stage 3b complete!")
    print(f"{'='*60}")
    print(f"  Best model: {results['best_model_type']}")
    print(f"  Best CV F1: {results['best_cv_score']:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Tuned model saved to: models/seismic_classifier_tuned.pkl")


if __name__ == "__main__":
    main()
