"""
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
seismic_features_source = FileSource(
    name="seismic_features_source",
    path="data/silver/seismic_features_feast.parquet",
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
