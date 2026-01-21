"""
Stage 3: Model Training

Train ML model with MLflow tracking:
- Load features from Stage 2 (8 handcrafted + 32 embeddings = 40 features)
- Optional: Load from Feature Store (Feast)
- Load optimized hyperparameters from Stage 3b (Optuna TPE)
- Query LLM for model recommendation
- Train model with optimized hyperparameters
- Track experiment with MLflow
- Save model artifacts

Updated to use:
- All 40 features from Stage 2 (including PCA embeddings)
- Feature Store integration (optional)
- Optimized hyperparameters from Stage 3b (Optuna TPE)
- F3 Netherlands dataset characteristics
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple, List
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from deltalake import DeltaTable


class ModelTrainer:
    """
    Train ML model with MLflow tracking.
    
    Supports:
    - Loading features from Delta Lake/Parquet (Stage 2 output)
    - Loading features from Feature Store (Feast)
    - Multiple model types (RF, LR, DT, GBM)
    - MLflow experiment tracking
    """
    
    def __init__(self, input_dir: str = "data/silver",
                 output_dir: str = "models",
                 mlflow_uri: str = None,
                 use_feature_store: bool = False):
        """
        Initialize trainer.
        
        Args:
            input_dir: Directory with Stage 2 features
            output_dir: Directory for model artifacts
            mlflow_uri: MLflow tracking URI (None for local file-based)
            use_feature_store: Whether to load from Feature Store
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_feature_store = use_feature_store
        
        # Setup MLflow (use local file-based tracking if no server)
        if mlflow_uri is None:
            mlflow_uri = "file:./mlruns"  # Local file-based tracking
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("seismic_classification")
        
        self.model = None
        self.model_type = "RandomForestClassifier"  # Default
        self.feature_names = []
        self.num_features = 0
        self.tuned_params = None  # Optimized hyperparameters from Stage 3b
        self.use_tuned_params = False
    
    def load_tuned_hyperparameters(self, tuning_results_path: str = "models/hyperparameter_tuning_results.json") -> bool:
        """
        Load optimized hyperparameters from Stage 3b (Optuna TPE).
        
        Args:
            tuning_results_path: Path to hyperparameter tuning results JSON
            
        Returns:
            True if successfully loaded, False otherwise
        """
        results_path = Path(tuning_results_path)
        if not results_path.exists():
            print(f"[WARNING] Tuning results not found at {tuning_results_path}")
            print("  Run stage3_hyperparameter_tuning.py first, or using default parameters")
            return False
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            self.tuned_params = {
                'best_model_type': results.get('best_model_type'),
                'best_params': results.get('best_params'),
                'best_cv_score': results.get('best_cv_score'),
                'all_results': results.get('all_results', {})
            }
            self.use_tuned_params = True
            
            print(f"\n[INFO] Loaded optimized hyperparameters from Stage 3b:")
            print(f"  Best model type: {self.tuned_params['best_model_type']}")
            print(f"  Best CV F1-score: {self.tuned_params['best_cv_score']:.4f}")
            print(f"  Best parameters: {self.tuned_params['best_params']}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load tuning results: {e}")
            return False
    
    def query_llm_for_model(self, num_features: int = 40, num_classes: int = 3) -> Optional[str]:
        """
        Query LLM for model recommendation based on dataset characteristics.
        
        Args:
            num_features: Number of features (default 40: 8 handcrafted + 32 embeddings)
            num_classes: Number of classes (default 3: Normal, Anomaly, Boundary)
            
        Returns:
            Model name or None if LLM not available
        """
        if not OLLAMA_AVAILABLE:
            return None
        
        try:
            prompt = f"""
            Recommend the simplest ML model for this classification task:
            - {num_classes}-class classification (seismic trace classification)
            - {num_features} numerical features (8 statistical + 32 PCA embeddings)
            - 500 training samples
            - Classes: Normal (50%), Anomaly (30%), Boundary (20%) - imbalanced
            
            Consider: Logistic Regression, Random Forest, Decision Tree, or Gradient Boosting.
            Which model would work best for this small dataset with many features?
            
            Return ONLY the model name (e.g., "Random Forest"), nothing else.
            """
            
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            model_name = response['response'].strip()
            print(f"LLM recommended model: {model_name}")
            return model_name
        except Exception as e:
            print(f"LLM query failed: {e}")
            return None
    
    def load_features(self, table_name: str = "seismic_features",
                     use_embeddings: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load features and labels from Delta/Parquet or Feature Store.
        
        Args:
            table_name: Name of the features table
            use_embeddings: Whether to include embedding features
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Try Feature Store first if enabled
        if self.use_feature_store:
            return self._load_from_feature_store()
        
        # Try Delta Lake first
        delta_path = self.input_dir / table_name
        if delta_path.exists():
            print(f"Loading features from Delta Lake: {delta_path}")
            dt = DeltaTable(str(delta_path))
            df = dt.to_pandas()
        else:
            # Fall back to Parquet
            parquet_path = self.input_dir / f"{table_name}.parquet"
            if parquet_path.exists():
                print(f"Loading features from Parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)
            else:
                raise FileNotFoundError(f"Features not found in {self.input_dir}")
        
        # Define feature columns
        # Handcrafted features (8)
        handcrafted_cols = [
            'mean_amplitude', 'std_amplitude', 'min_amplitude', 'max_amplitude',
            'rms_amplitude', 'energy', 'zero_crossings', 'dominant_frequency'
        ]
        
        # Embedding features (32)
        embedding_cols = [f'embedding_{i}' for i in range(32)]
        
        # Determine which features to use
        if use_embeddings:
            # Check if embeddings exist
            available_embedding_cols = [col for col in embedding_cols if col in df.columns]
            feature_cols = handcrafted_cols + available_embedding_cols
        else:
            feature_cols = handcrafted_cols
        
        # Use scaled features if available
        scaled_cols = [f"{col}_scaled" for col in feature_cols]
        if all(col in df.columns for col in scaled_cols):
            X = df[scaled_cols].values
            self.feature_names = scaled_cols
            print(f"Using scaled features ({len(scaled_cols)} features)")
        else:
            X = df[feature_cols].values
            self.feature_names = feature_cols
            print(f"Using original features ({len(feature_cols)} features)")
        
        self.num_features = X.shape[1]
        
        # Extract labels
        if 'class_label' in df.columns:
            y = df['class_label'].values
        else:
            raise ValueError("class_label column not found in features")
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"  Handcrafted features: {len(handcrafted_cols)}")
        if use_embeddings:
            print(f"  Embedding features: {len(available_embedding_cols)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, self.feature_names
    
    def _load_from_feature_store(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load features from Feast Feature Store."""
        try:
            from feature_store import FeatureStoreIntegration
            
            integration = FeatureStoreIntegration(
                data_dir="data",
                repo_path="feature_store"
            )
            
            df = integration.get_training_dataset()
            print(f"Loaded {len(df)} samples from Feature Store")
            
            # Get feature columns (exclude metadata)
            exclude_cols = ['trace_id', 'file_id', 'event_timestamp', 'class_label']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols].values
            y = df['class_label'].values
            self.feature_names = feature_cols
            self.num_features = X.shape[1]
            
            return X, y, feature_cols
            
        except Exception as e:
            print(f"Feature Store loading failed: {e}")
            print("Falling back to file-based loading...")
            self.use_feature_store = False
            return self.load_features()
    
    def create_model(self, model_name: Optional[str] = None, 
                     use_tuned: bool = True) -> Any:
        """
        Create model instance based on recommendation or tuned parameters.
        
        Args:
            model_name: Model name from LLM or None for default/tuned
            use_tuned: Whether to use tuned hyperparameters from Stage 3b
            
        Returns:
            Model instance
        """
        # If tuned parameters available and use_tuned is True, use them
        if use_tuned and self.use_tuned_params and self.tuned_params:
            return self._create_model_with_tuned_params()
        
        if model_name is None:
            model_name = "RandomForestClassifier"
        
        model_name_lower = model_name.lower()
        
        if "random" in model_name_lower and "forest" in model_name_lower:
            self.model_type = "RandomForestClassifier"
            model = RandomForestClassifier(
                n_estimators=100,  # More trees for 40 features
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
        elif "logistic" in model_name_lower:
            self.model_type = "LogisticRegression"
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial',
                class_weight='balanced',
                solver='lbfgs'
            )
        elif "decision" in model_name_lower and "tree" in model_name_lower:
            self.model_type = "DecisionTreeClassifier"
            model = DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif "gradient" in model_name_lower or "boosting" in model_name_lower or "gbm" in model_name_lower:
            self.model_type = "GradientBoostingClassifier"
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            # Default to Random Forest
            print(f"Unknown model '{model_name}', using Random Forest")
            self.model_type = "RandomForestClassifier"
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        self.model = model
        print(f"Created model: {self.model_type} (default parameters)")
        return model
    
    def _create_model_with_tuned_params(self) -> Any:
        """
        Create model with optimized hyperparameters from Stage 3b.
        
        Returns:
            Model instance with tuned parameters
        """
        best_model_type = self.tuned_params['best_model_type']
        best_params = self.tuned_params['best_params'].copy()
        
        print(f"\n[INFO] Creating model with TUNED hyperparameters from Stage 3b (Optuna TPE)")
        print(f"  Model type: {best_model_type}")
        print(f"  Parameters: {best_params}")
        
        # Add random_state for reproducibility
        best_params['random_state'] = 42
        
        if best_model_type == "RandomForest":
            self.model_type = "RandomForestClassifier"
            best_params['n_jobs'] = -1
            model = RandomForestClassifier(**best_params)
            
        elif best_model_type == "GradientBoosting":
            self.model_type = "GradientBoostingClassifier"
            model = GradientBoostingClassifier(**best_params)
            
        elif best_model_type == "LogisticRegression":
            self.model_type = "LogisticRegression"
            best_params['solver'] = 'saga'  # Supports both l1 and l2
            best_params['max_iter'] = 2000
            model = LogisticRegression(**best_params)
            
        elif best_model_type == "DecisionTree":
            self.model_type = "DecisionTreeClassifier"
            model = DecisionTreeClassifier(**best_params)
            
        else:
            print(f"[WARNING] Unknown tuned model type: {best_model_type}, using RandomForest")
            self.model_type = "RandomForestClassifier"
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        self.model = model
        print(f"Created model: {self.model_type} (TUNED parameters from Optuna TPE)")
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Train model and return metrics.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of training metrics
        """
        print(f"\nTraining {self.model_type}...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # F1 scores (weighted for imbalanced classes)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        metrics = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'train_f1_weighted': float(train_f1),
            'val_f1_weighted': float(val_f1),
            'num_features': int(X_train.shape[1]),
            'num_train_samples': int(len(X_train)),
            'num_val_samples': int(len(X_val))
        }
        
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Training F1 (weighted): {train_f1:.4f}")
        print(f"  Validation F1 (weighted): {val_f1:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True))
            return importance_dict
        elif hasattr(self.model, 'coef_'):
            # For logistic regression, use absolute coefficient values
            importances = np.abs(self.model.coef_).mean(axis=0)
            importance_dict = dict(zip(self.feature_names, importances))
            importance_dict = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True))
            return importance_dict
        
        return None
    
    def save_model(self, model_name: str = "seismic_classifier"):
        """
        Save model to disk.
        
        Args:
            model_name: Name for the model file
        """
        model_path = self.output_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save feature names
        feature_names_path = self.output_dir / f"{model_name}_features.json"
        with open(feature_names_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'num_features': self.num_features,
                'model_type': self.model_type
            }, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Feature names saved to: {feature_names_path}")
        return model_path


def main(use_tuned_params: bool = True):
    """
    Execute Stage 3: Model Training.
    
    Args:
        use_tuned_params: Whether to use optimized hyperparameters from Stage 3b
    """
    print("=" * 60)
    print("Stage 3: Model Training")
    print("=" * 60)
    print("Using features from Stage 2 (8 handcrafted + 32 embeddings)")
    
    # Initialize trainer
    trainer = ModelTrainer(
        input_dir="data/silver",
        output_dir="models",
        use_feature_store=False  # Set to True to use Feast Feature Store
    )
    
    # Try to load tuned hyperparameters from Stage 3b
    if use_tuned_params:
        print("\n" + "-" * 40)
        print("Loading optimized hyperparameters from Stage 3b...")
        print("-" * 40)
        tuned_loaded = trainer.load_tuned_hyperparameters()
    else:
        tuned_loaded = False
        print("\n[INFO] Using default hyperparameters (use_tuned_params=False)")
    
    # Load features (including embeddings)
    print("\nLoading features...")
    X, y, feature_names = trainer.load_features(use_embeddings=True)
    
    # Create model - use tuned params if available, otherwise query LLM
    if tuned_loaded:
        print("\nUsing TUNED hyperparameters from Stage 3b (Optuna TPE)")
        model = trainer.create_model(use_tuned=True)
    else:
        # Query LLM for model recommendation
        print("\nQuerying LLM for model recommendation...")
        recommended_model = trainer.query_llm_for_model(
            num_features=X.shape[1],
            num_classes=len(np.unique(y))
        )
        
        if recommended_model:
            print(f"Using LLM-recommended model: {recommended_model}")
        else:
            print("Using default model: RandomForestClassifier")
        
        model = trainer.create_model(recommended_model, use_tuned=False)
    
    # Split data: 64% train, 16% validation, 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train model with MLflow tracking
    run_name = f"{trainer.model_type}_tuned" if trainer.use_tuned_params else f"{trainer.model_type}_default"
    print(f"\nTraining model with MLflow tracking (run: {run_name})...")
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            'model_type': trainer.model_type,
            'num_features': X.shape[1],
            'num_handcrafted_features': 8,
            'num_embedding_features': X.shape[1] - 8,
            'num_samples': len(X),
            'num_classes': len(np.unique(y)),
            'test_size': 0.2,
            'val_size': 0.16,
            'random_state': 42,
            'hyperparameters_source': 'optuna_tpe' if trainer.use_tuned_params else 'default'
        })
        
        # Log model-specific parameters (from tuned or default)
        if trainer.use_tuned_params and trainer.tuned_params:
            # Log tuned parameters
            mlflow.log_params({
                'tuned_cv_score': trainer.tuned_params['best_cv_score'],
                **{f'tuned_{k}': v for k, v in trainer.tuned_params['best_params'].items() if v is not None}
            })
        elif trainer.model_type == "RandomForestClassifier":
            mlflow.log_params({
                'n_estimators': trainer.model.n_estimators,
                'max_depth': trainer.model.max_depth,
                'min_samples_split': trainer.model.min_samples_split,
                'class_weight': 'balanced'
            })
        elif trainer.model_type == "LogisticRegression":
            mlflow.log_params({
                'max_iter': trainer.model.max_iter,
                'multi_class': getattr(trainer.model, 'multi_class', 'auto'),
                'C': getattr(trainer.model, 'C', 1.0),
                'penalty': getattr(trainer.model, 'penalty', 'l2')
            })
        
        # Train
        metrics = trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Log training metrics
        mlflow.log_metrics(metrics)
        
        # Evaluate on test set
        y_test_pred = trainer.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        mlflow.log_metrics({
            'test_accuracy': float(test_acc),
            'test_f1_weighted': float(test_f1),
            'test_precision_weighted': float(test_precision),
            'test_recall_weighted': float(test_recall)
        })
        
        # Log model
        signature = infer_signature(X_train, trainer.model.predict(X_train))
        mlflow.sklearn.log_model(
            trainer.model,
            "model",
            signature=signature,
            registered_model_name="SeismicClassifier"
        )
        
        # Classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
        
        # Feature importance
        importance = trainer.get_feature_importance()
        if importance:
            mlflow.log_dict(importance, "feature_importance.json")
            
            # Log top 10 features
            print("\nTop 10 Most Important Features:")
            for i, (name, imp) in enumerate(list(importance.items())[:10]):
                print(f"  {i+1}. {name}: {imp:.4f}")
        
        # Print results
        print(f"\n{'='*40}")
        print("Test Set Results")
        print(f"{'='*40}")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"F1 Score (weighted): {test_f1:.4f}")
        print(f"Precision (weighted): {test_precision:.4f}")
        print(f"Recall (weighted): {test_recall:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['Normal', 'Anomaly', 'Boundary']))
        
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save model locally
        trainer.save_model()
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print(f"Model registered as: SeismicClassifier")
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Stage 3 complete!")
    print(f"{'='*60}")
    print(f"  Model type: {trainer.model_type}")
    print(f"  Hyperparameters: {'TUNED (Optuna TPE)' if trainer.use_tuned_params else 'Default'}")
    if trainer.use_tuned_params and trainer.tuned_params:
        print(f"  Tuned CV F1-score: {trainer.tuned_params['best_cv_score']:.4f}")
    print(f"  Features used: {trainer.num_features} (8 handcrafted + {trainer.num_features - 8} embeddings)")
    print(f"  Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Validation accuracy: {metrics['val_accuracy']:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Model saved to: models/seismic_classifier.pkl")


if __name__ == "__main__":
    main()
