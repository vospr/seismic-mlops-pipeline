"""
Stage 4: Model Evaluation

Comprehensive evaluation with:
- Multiple metrics (accuracy, precision, recall, F1-score)
- Confusion matrix and ROC-AUC
- Data drift detection (KS test)
- LLM-generated evaluation report
- Feature importance analysis

Updated to use:
- All 40 features from Stage 2 (8 handcrafted + 32 embeddings)
- Feature Store integration (optional)
- Consistent with Stage 3 model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, List, Tuple
import pickle

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from scipy import stats

import mlflow
import mlflow.sklearn

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from deltalake import DeltaTable, write_deltalake


class ModelEvaluator:
    """
    Comprehensive model evaluation with drift detection.
    
    Supports:
    - Loading features from Delta Lake/Parquet (Stage 2 output)
    - Loading features from Feature Store (Feast)
    - All 40 features (8 handcrafted + 32 embeddings)
    - Multiple evaluation metrics
    - Data drift detection using KS test
    - LLM-generated evaluation reports
    """
    
    def __init__(self, features_dir: str = "data/silver",
                 models_dir: str = "models",
                 output_dir: str = "data/gold",
                 mlflow_uri: str = "file:./mlruns",
                 use_feature_store: bool = False):
        """
        Initialize evaluator.
        
        Args:
            features_dir: Directory with Stage 2 features
            models_dir: Directory with trained models
            output_dir: Directory for evaluation outputs (gold layer)
            mlflow_uri: MLflow tracking URI
            use_feature_store: Whether to load from Feature Store
        """
        self.features_dir = Path(features_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_feature_store = use_feature_store
        
        mlflow.set_tracking_uri(mlflow_uri)
        
        self.feature_names: List[str] = []
        self.num_features = 0
    
    def load_model(self, model_name: str = "seismic_classifier.pkl") -> Any:
        """
        Load trained model and feature names.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Loaded model
        """
        model_path = self.models_dir / model_name
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load feature names if available
        feature_names_path = self.models_dir / f"{model_name.replace('.pkl', '')}_features.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_info = json.load(f)
                self.feature_names = feature_info.get('feature_names', [])
                self.num_features = feature_info.get('num_features', 0)
            print(f"Loaded feature names: {self.num_features} features")
        
        print(f"Loaded model from: {model_path}")
        return model
    
    def load_features(self, table_name: str = "seismic_features",
                     use_embeddings: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load features and labels from Delta/Parquet or Feature Store.
        
        Args:
            table_name: Name of the features table
            use_embeddings: Whether to include embedding features
            
        Returns:
            Tuple of (X, y, metadata_df)
        """
        # Try Feature Store first if enabled
        if self.use_feature_store:
            return self._load_from_feature_store()
        
        # Try Delta Lake first
        delta_path = self.features_dir / table_name
        if delta_path.exists():
            print(f"Loading features from Delta Lake: {delta_path}")
            dt = DeltaTable(str(delta_path))
            df = dt.to_pandas()
        else:
            parquet_path = self.features_dir / f"{table_name}.parquet"
            if parquet_path.exists():
                print(f"Loading features from Parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)
            else:
                raise FileNotFoundError(f"Features not found in {self.features_dir}")
        
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
        y = df['class_label'].values if 'class_label' in df.columns else None
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        if use_embeddings:
            print(f"  Handcrafted features: {len(handcrafted_cols)}")
            print(f"  Embedding features: {len(available_embedding_cols)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, df
    
    def _load_from_feature_store(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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
            
            return X, y, df
            
        except Exception as e:
            print(f"Feature Store loading failed: {e}")
            print("Falling back to file-based loading...")
            self.use_feature_store = False
            return self.load_features()
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model with comprehensive metrics.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision_macro': float(precision_score(y, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(y, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
            'num_features': self.num_features,
            'num_samples': len(y)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        class_names = ['normal', 'anomaly', 'boundary']
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = float(precision_per_class[i])
                metrics[f'recall_{class_name}'] = float(recall_per_class[i])
                metrics[f'f1_{class_name}'] = float(f1_per_class[i])
        
        # ROC AUC (if probabilities available)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc_macro'] = float(roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro'))
                metrics['roc_auc_weighted'] = float(roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted'))
            except Exception as e:
                print(f"ROC AUC calculation failed: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        return metrics
    
    def detect_drift(self, X_train: np.ndarray, X_test: np.ndarray,
                    significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            X_train: Training features
            X_test: Test features
            significance_level: P-value threshold for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        print("Detecting data drift...")
        
        drift_results = {
            'features_drifted': [],
            'features_drifted_names': [],
            'ks_statistics': {},
            'ks_pvalues': {},
            'drift_detected': False,
            'drift_severity': 'none',
            'num_features_tested': X_train.shape[1],
            'significance_level': significance_level
        }
        
        # Kolmogorov-Smirnov test for each feature
        for feature_idx in range(X_train.shape[1]):
            train_feature = X_train[:, feature_idx]
            test_feature = X_test[:, feature_idx]
            
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(train_feature, test_feature)
            
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f'feature_{feature_idx}'
            
            drift_results['ks_statistics'][feature_name] = float(ks_stat)
            drift_results['ks_pvalues'][feature_name] = float(ks_pvalue)
            
            # Drift detected if p-value < significance level
            if ks_pvalue < significance_level:
                drift_results['features_drifted'].append(feature_idx)
                drift_results['features_drifted_names'].append(feature_name)
                drift_results['drift_detected'] = True
        
        # Calculate drift severity
        num_drifted = len(drift_results['features_drifted'])
        drift_ratio = num_drifted / X_train.shape[1]
        
        if drift_ratio == 0:
            drift_results['drift_severity'] = 'none'
        elif drift_ratio < 0.1:
            drift_results['drift_severity'] = 'low'
        elif drift_ratio < 0.3:
            drift_results['drift_severity'] = 'medium'
        else:
            drift_results['drift_severity'] = 'high'
        
        drift_results['drift_ratio'] = float(drift_ratio)
        
        if drift_results['drift_detected']:
            print(f"  Drift detected in {num_drifted}/{X_train.shape[1]} features ({drift_ratio*100:.1f}%)")
            print(f"  Drift severity: {drift_results['drift_severity']}")
            if num_drifted <= 5:
                print(f"  Drifted features: {drift_results['features_drifted_names']}")
        else:
            print("  No significant drift detected")
        
        return drift_results
    
    def analyze_feature_importance_drift(self, model, X_train: np.ndarray, 
                                         X_test: np.ndarray) -> Dict[str, Any]:
        """
        Analyze if important features have drifted.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            
        Returns:
            Dictionary with feature importance drift analysis
        """
        analysis = {
            'important_features_drifted': [],
            'top_features': [],
            'drift_in_top_features': False
        }
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            return analysis
        
        # Get top 10 important features
        top_indices = np.argsort(importances)[-10:][::-1]
        
        for idx in top_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
            analysis['top_features'].append({
                'name': feature_name,
                'importance': float(importances[idx]),
                'index': int(idx)
            })
            
            # Check if this feature has drifted
            ks_stat, ks_pvalue = stats.ks_2samp(X_train[:, idx], X_test[:, idx])
            if ks_pvalue < 0.05:
                analysis['important_features_drifted'].append({
                    'name': feature_name,
                    'importance': float(importances[idx]),
                    'ks_pvalue': float(ks_pvalue)
                })
                analysis['drift_in_top_features'] = True
        
        return analysis
    
    def llm_generate_report(self, metrics: Dict[str, Any], 
                           drift_results: Dict[str, Any],
                           importance_analysis: Dict[str, Any] = None) -> Optional[str]:
        """
        Use LLM to generate comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics
            drift_results: Drift detection results
            importance_analysis: Feature importance drift analysis
            
        Returns:
            LLM-generated report or None
        """
        if not OLLAMA_AVAILABLE:
            print("Ollama not available, skipping LLM report generation")
            return None
        
        try:
            # Prepare compact data for LLM
            compact_data = {
                'accuracy': round(metrics['accuracy'], 4),
                'f1_macro': round(metrics['f1_macro'], 4),
                'f1_weighted': round(metrics['f1_weighted'], 4),
                'precision_macro': round(metrics['precision_macro'], 4),
                'recall_macro': round(metrics['recall_macro'], 4),
                'num_features': metrics.get('num_features', 40),
                'drift_detected': drift_results['drift_detected'],
                'drift_severity': drift_results['drift_severity'],
                'features_drifted_count': len(drift_results['features_drifted']),
                'drift_ratio': round(drift_results.get('drift_ratio', 0), 4)
            }
            
            if importance_analysis:
                compact_data['important_features_drifted'] = len(importance_analysis.get('important_features_drifted', []))
            
            prompt = f"""
            Analyze these ML model evaluation results for a seismic trace classification model.
            The model classifies traces into 3 classes: Normal, Anomaly, Boundary.
            
            Evaluation Metrics:
            {json.dumps(compact_data, indent=2)}
            
            Provide a brief evaluation report with:
            1. Overall Performance Assessment (excellent/good/fair/poor) with reasoning
            2. Drift Analysis - severity and impact on model reliability
            3. Top 3 Recommendations for improvement
            4. Production Readiness Assessment (ready/needs-work/not-ready)
            
            Keep the response concise and actionable.
            """
            
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=prompt,
                options={"temperature": 0.2}
            )
            
            return response['response']
        except Exception as e:
            print(f"LLM report generation failed: {e}")
            return None
    
    def save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        metadata_df: pd.DataFrame, y_pred_proba: np.ndarray = None):
        """
        Save predictions to Delta/Parquet (gold layer).
        
        Args:
            y_true: True labels
            y_pred: Predictions
            metadata_df: Metadata DataFrame
            y_pred_proba: Prediction probabilities (optional)
        """
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'file_id': metadata_df['file_id'].values[:len(y_true)],
            'trace_id': metadata_df['trace_id'].values[:len(y_true)],
            'true_label': y_true,
            'predicted_label': y_pred,
            'correct': (y_true == y_pred)
        })
        
        # Add class names
        class_names = {0: 'Normal', 1: 'Anomaly', 2: 'Boundary'}
        predictions_df['true_class'] = predictions_df['true_label'].map(class_names)
        predictions_df['predicted_class'] = predictions_df['predicted_label'].map(class_names)
        
        # Add probabilities if available
        if y_pred_proba is not None:
            for i, class_name in enumerate(['prob_normal', 'prob_anomaly', 'prob_boundary']):
                if i < y_pred_proba.shape[1]:
                    predictions_df[class_name] = y_pred_proba[:, i]
        
        # Save to Delta Lake
        output_path = self.output_dir / "predictions"
        
        # Remove existing if present (handle schema changes)
        import shutil
        if output_path.exists():
            shutil.rmtree(output_path)
        
        write_deltalake(
            str(output_path),
            predictions_df,
            mode="overwrite"
        )
        
        # Also save as Parquet
        parquet_path = self.output_dir / "predictions.parquet"
        predictions_df.to_parquet(parquet_path, engine='pyarrow', index=False)
        
        print(f"Predictions saved to: {output_path}")
        print(f"  Total predictions: {len(predictions_df)}")
        print(f"  Correct: {predictions_df['correct'].sum()} ({predictions_df['correct'].mean()*100:.1f}%)")
        
        return output_path


def main():
    """Execute Stage 4: Model Evaluation."""
    print("=" * 60)
    print("Stage 4: Model Evaluation & Drift Detection")
    print("=" * 60)
    print("Using features from Stage 2 (8 handcrafted + 32 embeddings)")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        features_dir="data/silver",
        models_dir="models",
        output_dir="data/gold",
        use_feature_store=False  # Set to True to use Feast Feature Store
    )
    
    # Load model
    print("\nLoading model...")
    model = evaluator.load_model()
    
    # Load features (including embeddings)
    print("\nLoading features...")
    X, y, metadata_df = evaluator.load_features(use_embeddings=True)
    
    # Split into train/test (same split as Stage 3 training)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\nData split (same as Stage 3):")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating model on test set...")
    test_metrics = evaluator.evaluate_model(model, X_test, y_test)
    
    print("\nTest Set Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted): {test_metrics['precision_weighted']:.4f}")
    print(f"  Recall (macro): {test_metrics['recall_macro']:.4f}")
    print(f"  Recall (weighted): {test_metrics['recall_weighted']:.4f}")
    print(f"  F1-score (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1-score (weighted): {test_metrics['f1_weighted']:.4f}")
    
    if 'roc_auc_macro' in test_metrics:
        print(f"  ROC-AUC (macro): {test_metrics['roc_auc_macro']:.4f}")
    
    print("\nPer-Class Metrics:")
    for class_name in ['normal', 'anomaly', 'boundary']:
        if f'precision_{class_name}' in test_metrics:
            print(f"  {class_name.capitalize()}:")
            print(f"    Precision: {test_metrics[f'precision_{class_name}']:.4f}")
            print(f"    Recall: {test_metrics[f'recall_{class_name}']:.4f}")
            print(f"    F1-score: {test_metrics[f'f1_{class_name}']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(test_metrics['confusion_matrix'])
    print(f"  {'':>10} {'Normal':>8} {'Anomaly':>8} {'Boundary':>8}")
    for i, class_name in enumerate(['Normal', 'Anomaly', 'Boundary']):
        print(f"  {class_name:>10} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8}")
    
    # Drift detection
    print("\n" + "=" * 60)
    print("Data Drift Detection")
    print("=" * 60)
    drift_results = evaluator.detect_drift(X_train, X_test)
    
    # Feature importance drift analysis
    print("\nAnalyzing feature importance drift...")
    importance_analysis = evaluator.analyze_feature_importance_drift(model, X_train, X_test)
    
    if importance_analysis['top_features']:
        print("\nTop 5 Most Important Features:")
        for i, feat in enumerate(importance_analysis['top_features'][:5]):
            print(f"  {i+1}. {feat['name']}: {feat['importance']:.4f}")
    
    if importance_analysis['drift_in_top_features']:
        print("\n⚠️  WARNING: Drift detected in important features!")
        for feat in importance_analysis['important_features_drifted']:
            print(f"    - {feat['name']} (importance: {feat['importance']:.4f}, p-value: {feat['ks_pvalue']:.4f})")
    
    # LLM-generated report
    print("\n" + "=" * 60)
    print("Generating LLM evaluation report...")
    llm_report = evaluator.llm_generate_report(test_metrics, drift_results, importance_analysis)
    if llm_report:
        print("\nLLM Evaluation Report:")
        print("-" * 40)
        print(llm_report)
        print("-" * 40)
    
    # Save predictions
    print("\nSaving predictions to gold layer...")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Get metadata for test samples
    test_indices = list(range(len(X_train_full), len(X_train_full) + len(X_test)))
    test_metadata = metadata_df.iloc[test_indices].reset_index(drop=True)
    
    evaluator.save_predictions(y_test, y_test_pred, test_metadata, y_test_proba)
    
    # Save comprehensive evaluation results
    results = {
        'metrics': test_metrics,
        'drift_detection': drift_results,
        'feature_importance_analysis': importance_analysis,
        'llm_report': llm_report,
        'data_split': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'total_samples': len(X)
        },
        'model_info': {
            'num_features': evaluator.num_features,
            'feature_names': evaluator.feature_names[:10] + ['...'] if len(evaluator.feature_names) > 10 else evaluator.feature_names
        }
    }
    
    results_path = evaluator.output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Log to MLflow
    try:
        mlflow.set_experiment("seismic_classification")
        with mlflow.start_run(run_name="stage4_evaluation"):
            mlflow.log_metrics({
                'test_accuracy': test_metrics['accuracy'],
                'test_f1_macro': test_metrics['f1_macro'],
                'test_f1_weighted': test_metrics['f1_weighted'],
                'test_precision_macro': test_metrics['precision_macro'],
                'test_recall_macro': test_metrics['recall_macro'],
                'drift_ratio': drift_results.get('drift_ratio', 0),
                'num_features_drifted': len(drift_results['features_drifted'])
            })
            mlflow.log_params({
                'num_features': evaluator.num_features,
                'drift_severity': drift_results['drift_severity'],
                'test_samples': len(X_test)
            })
            mlflow.log_dict(results, "evaluation_results.json")
    except Exception as e:
        print(f"MLflow logging failed: {e}")
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Stage 4 complete!")
    print(f"{'='*60}")
    print(f"  Features evaluated: {evaluator.num_features} (8 handcrafted + {evaluator.num_features - 8} embeddings)")
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  Drift detected: {drift_results['drift_detected']} ({drift_results['drift_severity']})")
    print(f"  Results saved to: {results_path}")
    print(f"  Predictions saved to: {evaluator.output_dir / 'predictions'}")


if __name__ == "__main__":
    main()
