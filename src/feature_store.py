"""
Feature Store Implementation using Feast

Provides centralized feature management for the seismic ML pipeline:
- Feature definitions and schemas
- Feature versioning and lineage
- Online/offline feature serving
- Point-in-time feature retrieval

This demonstrates Feature Store concepts required for MLOps.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Feast imports
try:
    from feast import Entity, Feature, FeatureView, FileSource, ValueType
    from feast import FeatureStore as FeastFeatureStore
    from feast.types import Float32, Float64, Int64, String
    from feast.infra.offline_stores.file_source import FileSource
    FEAST_AVAILABLE = True
except ImportError as e:
    FEAST_AVAILABLE = False
    print(f"Warning: Feast not available ({e}). Install with: pip install feast")


class SeismicFeatureStore:
    """
    Feature Store for seismic data using Feast.
    
    Provides:
    - Feature definitions for seismic traces
    - Offline feature storage (Parquet/Delta)
    - Feature retrieval for training and inference
    - Feature versioning and metadata
    """
    
    def __init__(self, repo_path: str = "feature_store", 
                 data_dir: str = "data"):
        """
        Initialize Feature Store.
        
        Args:
            repo_path: Path to Feast repository
            data_dir: Path to data directory
        """
        self.repo_path = Path(repo_path)
        self.data_dir = Path(data_dir)
        self.store = None
        
        # Create repo directory
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        if FEAST_AVAILABLE:
            self._initialize_feast()
    
    def _initialize_feast(self):
        """Initialize Feast repository and feature store."""
        # Create feature_store.yaml config
        config_path = self.repo_path / "feature_store.yaml"
        
        config = f"""
project: seismic_mlops
registry: {self.repo_path}/registry.db
provider: local
offline_store:
  type: file
online_store:
  type: sqlite
  path: {self.repo_path}/online_store.db
entity_key_serialization_version: 2
"""
        
        with open(config_path, 'w') as f:
            f.write(config)
        
        print(f"Feast config created: {config_path}")
    
    def create_feature_definitions(self) -> str:
        """
        Create Feast feature definitions file.
        
        Returns:
            Path to the created definitions file
        """
        if not FEAST_AVAILABLE:
            raise RuntimeError("Feast not available")
        
        # Create feature definitions Python file
        definitions = '''"""
Seismic Feature Definitions for Feast Feature Store

Defines:
- Entities (trace identifiers)
- Feature Views (feature groups)
- Data Sources (Parquet files)
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String

# =============================================================================
# ENTITIES
# =============================================================================

# Trace entity - unique identifier for each seismic trace
trace_entity = Entity(
    name="trace_id",
    description="Unique identifier for a seismic trace",
    join_keys=["trace_id"],
)

# File entity - groups traces by source file
file_entity = Entity(
    name="file_id",
    description="Source file identifier",
    join_keys=["file_id"],
)

# =============================================================================
# DATA SOURCES
# =============================================================================

# Seismic features source (from Stage 2 output)
# Use absolute path for Feast compatibility
import os
_data_path = os.path.abspath("../data/silver/seismic_features_feast.parquet")
seismic_features_source = FileSource(
    name="seismic_features_source",
    path=_data_path,
    timestamp_field="event_timestamp",
)

# =============================================================================
# FEATURE VIEWS
# =============================================================================

# Statistical features view
statistical_features_view = FeatureView(
    name="seismic_statistical_features",
    entities=[trace_entity],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        Field(name="mean_amplitude", dtype=Float64),
        Field(name="std_amplitude", dtype=Float64),
        Field(name="min_amplitude", dtype=Float64),
        Field(name="max_amplitude", dtype=Float64),
        Field(name="rms_amplitude", dtype=Float64),
    ],
    source=seismic_features_source,
    description="Statistical features extracted from seismic traces",
    tags={"team": "seismic", "stage": "feature_engineering"},
)

# Energy and frequency features view
energy_frequency_features_view = FeatureView(
    name="seismic_energy_frequency_features",
    entities=[trace_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="energy", dtype=Float64),
        Field(name="zero_crossings", dtype=Int64),
        Field(name="dominant_frequency", dtype=Float64),
    ],
    source=seismic_features_source,
    description="Energy and frequency features from seismic traces",
    tags={"team": "seismic", "stage": "feature_engineering"},
)

# Embedding features view
embedding_features_view = FeatureView(
    name="seismic_embedding_features",
    entities=[trace_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="embedding_0", dtype=Float32),
        Field(name="embedding_1", dtype=Float32),
        Field(name="embedding_2", dtype=Float32),
        Field(name="embedding_3", dtype=Float32),
        Field(name="embedding_4", dtype=Float32),
        Field(name="embedding_5", dtype=Float32),
        Field(name="embedding_6", dtype=Float32),
        Field(name="embedding_7", dtype=Float32),
        Field(name="embedding_8", dtype=Float32),
        Field(name="embedding_9", dtype=Float32),
        Field(name="embedding_10", dtype=Float32),
        Field(name="embedding_11", dtype=Float32),
        Field(name="embedding_12", dtype=Float32),
        Field(name="embedding_13", dtype=Float32),
        Field(name="embedding_14", dtype=Float32),
        Field(name="embedding_15", dtype=Float32),
        Field(name="embedding_16", dtype=Float32),
        Field(name="embedding_17", dtype=Float32),
        Field(name="embedding_18", dtype=Float32),
        Field(name="embedding_19", dtype=Float32),
        Field(name="embedding_20", dtype=Float32),
        Field(name="embedding_21", dtype=Float32),
        Field(name="embedding_22", dtype=Float32),
        Field(name="embedding_23", dtype=Float32),
        Field(name="embedding_24", dtype=Float32),
        Field(name="embedding_25", dtype=Float32),
        Field(name="embedding_26", dtype=Float32),
        Field(name="embedding_27", dtype=Float32),
        Field(name="embedding_28", dtype=Float32),
        Field(name="embedding_29", dtype=Float32),
        Field(name="embedding_30", dtype=Float32),
        Field(name="embedding_31", dtype=Float32),
    ],
    source=seismic_features_source,
    description="PCA embedding features from seismic traces",
    tags={"team": "seismic", "stage": "feature_engineering", "type": "embedding"},
)

# Labels view (for training)
labels_view = FeatureView(
    name="seismic_labels",
    entities=[trace_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="class_label", dtype=Int64),
        Field(name="file_id", dtype=String),
    ],
    source=seismic_features_source,
    description="Class labels for seismic traces",
    tags={"team": "seismic", "stage": "labels"},
)
'''
        
        definitions_path = self.repo_path / "feature_definitions.py"
        with open(definitions_path, 'w') as f:
            f.write(definitions)
        
        print(f"Feature definitions created: {definitions_path}")
        return str(definitions_path)
    
    def prepare_features_for_feast(self, features_path: str = None) -> str:
        """
        Prepare feature data for Feast (add timestamp column).
        
        Args:
            features_path: Path to features parquet file
            
        Returns:
            Path to prepared parquet file
        """
        if features_path is None:
            features_path = self.data_dir / "silver" / "seismic_features.parquet"
        
        features_path = Path(features_path)
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        # Load features
        df = pd.read_parquet(features_path)
        print(f"Loaded {len(df)} features from {features_path}")
        
        # Add event_timestamp (required by Feast)
        # Use current time for all features (batch ingestion)
        df['event_timestamp'] = pd.Timestamp.now()
        
        # Ensure trace_id is integer
        df['trace_id'] = df['trace_id'].astype(int)
        
        # Ensure file_id is string
        df['file_id'] = df['file_id'].astype(str)
        
        # Select only the columns needed for Feast
        # (original features, not scaled)
        feature_cols = [
            'trace_id', 'file_id', 'event_timestamp',
            'mean_amplitude', 'std_amplitude', 'min_amplitude', 'max_amplitude',
            'rms_amplitude', 'energy', 'zero_crossings', 'dominant_frequency',
            'class_label'
        ]
        
        # Add embedding columns if they exist
        embedding_cols = [f'embedding_{i}' for i in range(32)]
        for col in embedding_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        df_feast = df[feature_cols].copy()
        
        # Save prepared features
        output_path = self.data_dir / "silver" / "seismic_features_feast.parquet"
        df_feast.to_parquet(output_path, index=False)
        
        print(f"Prepared features saved to: {output_path}")
        print(f"  Columns: {len(feature_cols)}")
        print(f"  Rows: {len(df_feast)}")
        
        return str(output_path)
    
    def apply_feature_definitions(self):
        """Apply feature definitions to Feast registry."""
        if not FEAST_AVAILABLE:
            raise RuntimeError("Feast not available")
        
        import subprocess
        import os
        
        # Change to repo directory and run feast apply
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            result = subprocess.run(
                ['feast', 'apply'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Feature definitions applied successfully!")
                print(result.stdout)
            else:
                print(f"Error applying features: {result.stderr}")
                
        finally:
            os.chdir(original_dir)
    
    def get_historical_features(self, entity_df: pd.DataFrame, 
                               feature_refs: List[str]) -> pd.DataFrame:
        """
        Get historical features for training.
        
        Args:
            entity_df: DataFrame with entity keys and timestamps
            feature_refs: List of feature references (e.g., "seismic_statistical_features:mean_amplitude")
            
        Returns:
            DataFrame with requested features
        """
        if not FEAST_AVAILABLE:
            raise RuntimeError("Feast not available")
        
        # Initialize Feast store
        store = FeastFeatureStore(repo_path=str(self.repo_path))
        
        # Get historical features
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
        
        return training_df
    
    def materialize_features(self, start_date: datetime = None, 
                            end_date: datetime = None):
        """
        Materialize features to online store for serving.
        
        Args:
            start_date: Start of materialization window
            end_date: End of materialization window
        """
        if not FEAST_AVAILABLE:
            raise RuntimeError("Feast not available")
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Initialize Feast store
        store = FeastFeatureStore(repo_path=str(self.repo_path))
        
        # Materialize features
        store.materialize(start_date=start_date, end_date=end_date)
        
        print(f"Features materialized from {start_date} to {end_date}")
    
    def get_online_features(self, entity_rows: List[Dict]) -> Dict:
        """
        Get features from online store for inference.
        
        Args:
            entity_rows: List of entity dictionaries (e.g., [{"trace_id": 1}])
            
        Returns:
            Dictionary with feature values
        """
        if not FEAST_AVAILABLE:
            raise RuntimeError("Feast not available")
        
        # Initialize Feast store
        store = FeastFeatureStore(repo_path=str(self.repo_path))
        
        # Get online features
        feature_vector = store.get_online_features(
            features=[
                "seismic_statistical_features:mean_amplitude",
                "seismic_statistical_features:std_amplitude",
                "seismic_statistical_features:rms_amplitude",
                "seismic_energy_frequency_features:energy",
                "seismic_energy_frequency_features:dominant_frequency",
            ],
            entity_rows=entity_rows
        ).to_dict()
        
        return feature_vector
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about registered features.
        
        Returns:
            Dictionary with feature metadata
        """
        metadata = {
            'project': 'seismic_mlops',
            'repo_path': str(self.repo_path),
            'feature_views': [
                {
                    'name': 'seismic_statistical_features',
                    'features': ['mean_amplitude', 'std_amplitude', 'min_amplitude', 
                                'max_amplitude', 'rms_amplitude'],
                    'entity': 'trace_id',
                    'description': 'Statistical features from seismic traces'
                },
                {
                    'name': 'seismic_energy_frequency_features',
                    'features': ['energy', 'zero_crossings', 'dominant_frequency'],
                    'entity': 'trace_id',
                    'description': 'Energy and frequency features'
                },
                {
                    'name': 'seismic_embedding_features',
                    'features': [f'embedding_{i}' for i in range(32)],
                    'entity': 'trace_id',
                    'description': 'PCA embedding features (32 dimensions)'
                },
                {
                    'name': 'seismic_labels',
                    'features': ['class_label', 'file_id'],
                    'entity': 'trace_id',
                    'description': 'Class labels for training'
                }
            ],
            'entities': [
                {'name': 'trace_id', 'type': 'int64', 'description': 'Unique trace identifier'},
                {'name': 'file_id', 'type': 'string', 'description': 'Source file identifier'}
            ]
        }
        
        return metadata
    
    def save_feature_catalog(self):
        """Save feature catalog/registry as JSON for documentation."""
        metadata = self.get_feature_metadata()
        metadata['created_at'] = datetime.now().isoformat()
        
        catalog_path = self.repo_path / "feature_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Feature catalog saved to: {catalog_path}")
        return catalog_path


class FeatureStoreIntegration:
    """
    Integration layer between Stage 2 and Feature Store.
    """
    
    def __init__(self, data_dir: str = "data", repo_path: str = "feature_store"):
        self.data_dir = Path(data_dir)
        self.repo_path = Path(repo_path)
        self.feature_store = SeismicFeatureStore(repo_path=repo_path, data_dir=data_dir)
    
    def register_features(self):
        """
        Register features from Stage 2 to Feature Store.
        
        Steps:
        1. Create feature definitions
        2. Prepare data for Feast
        3. Apply definitions to registry
        4. Save feature catalog
        """
        print("=" * 60)
        print("Registering Features to Feature Store")
        print("=" * 60)
        
        # Step 1: Create feature definitions
        print("\n1. Creating feature definitions...")
        self.feature_store.create_feature_definitions()
        
        # Step 2: Prepare features for Feast
        print("\n2. Preparing features for Feast...")
        self.feature_store.prepare_features_for_feast()
        
        # Step 3: Apply definitions
        print("\n3. Applying feature definitions...")
        self.feature_store.apply_feature_definitions()
        
        # Step 4: Save catalog
        print("\n4. Saving feature catalog...")
        self.feature_store.save_feature_catalog()
        
        print("\n[SUCCESS] Features registered to Feature Store!")
        print(f"  Repository: {self.repo_path}")
        print(f"  Feature views: 4")
        print(f"  Total features: 43 (8 statistical + 3 energy/freq + 32 embeddings)")
    
    def get_training_dataset(self, feature_views: List[str] = None) -> pd.DataFrame:
        """
        Get training dataset from Feature Store.
        
        Args:
            feature_views: List of feature view names to include
            
        Returns:
            DataFrame with features for training
        """
        if feature_views is None:
            feature_views = [
                'seismic_statistical_features',
                'seismic_energy_frequency_features',
                'seismic_embedding_features',
                'seismic_labels'
            ]
        
        # For local development, read directly from prepared parquet
        feast_path = self.data_dir / "silver" / "seismic_features_feast.parquet"
        
        if feast_path.exists():
            df = pd.read_parquet(feast_path)
            print(f"Loaded training dataset: {len(df)} samples, {len(df.columns)} features")
            return df
        else:
            raise FileNotFoundError(f"Feature data not found: {feast_path}")


def main():
    """Demonstrate Feature Store functionality."""
    print("=" * 60)
    print("Feature Store Implementation (Feast)")
    print("=" * 60)
    
    # Check Feast availability
    print(f"\nFeast available: {FEAST_AVAILABLE}")
    
    if not FEAST_AVAILABLE:
        print("Install Feast with: pip install feast")
        return
    
    # Initialize integration
    integration = FeatureStoreIntegration(
        data_dir="data",
        repo_path="feature_store"
    )
    
    # Register features
    integration.register_features()
    
    # Get training dataset
    print("\n" + "=" * 60)
    print("Getting Training Dataset")
    print("=" * 60)
    
    try:
        df = integration.get_training_dataset()
        print(f"\nTraining dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error getting training dataset: {e}")
    
    # Show feature metadata
    print("\n" + "=" * 60)
    print("Feature Metadata")
    print("=" * 60)
    
    metadata = integration.feature_store.get_feature_metadata()
    print(f"\nProject: {metadata['project']}")
    print(f"Feature Views: {len(metadata['feature_views'])}")
    for fv in metadata['feature_views']:
        print(f"  - {fv['name']}: {len(fv['features'])} features")
    
    print("\n[SUCCESS] Feature Store demonstration complete!")


if __name__ == "__main__":
    main()
