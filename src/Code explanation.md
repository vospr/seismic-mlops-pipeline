# Code Explanation - MLOps Pipeline Stages

This document provides detailed explanations for each stage of the MLOps pipeline implementation.

---

## Table of Contents

1. [Stage 0: Data Sampling & Preprocessing](#stage-0-data-sampling--preprocessing)
2. [Stage 1: Data Ingestion & Quality Assurance](#stage-1-data-ingestion--quality-assurance)
3. [AI Quality Agents](#ai-quality-agents)
4. [Stage 2: Feature Engineering with Embeddings](#stage-2-feature-engineering-with-embeddings)
5. [RAG Pipeline (LLMOps Component)](#rag-pipeline-llmops-component)
6. [Feature Store (Feast)](#feature-store-feast)
7. [Stage 3: Model Training](#stage-3-model-training)
8. [Stage 3b: Hyperparameter Tuning (Optuna TPE)](#stage-3b-hyperparameter-tuning-optuna-tpe)
9. [Stage 4: Model Evaluation](#stage-4-model-evaluation)
10. [Stage 5: Model Registry](#stage-5-model-registry)
11. [Stage 6: Model Deployment](#stage-6-model-deployment)
12. [Stage 7: Monitoring & Observability](#stage-7-monitoring--observability)
13. [Stage 8: CI/CD Automation](#stage-8-cicd-automation)
14. [Common Patterns Across Stages](#common-patterns-across-stages)

---

## Stage 0: Data Sampling & Preprocessing

**File:** `stage0_data_sampling.py`

### Overview

This script implements **Stage 0** (pre-processing stage) of the MLOps pipeline: reading the F3 dataset from the Temp folder, performing random sampling to create a manageable dataset size, and imputing missing values using KNN (k=2). This stage prepares the data for Stage 1 ingestion.

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **segyio** | Read SGY/SEGY files | Industry-standard Python library for seismic data; supports F3 dataset format | ObsPy (heavier, more for seismology), segpy (less maintained) |
| **pandas** | DataFrame operations | Standard for tabular data manipulation; easy sampling and filtering | Polars (faster but less ecosystem support), PySpark (overkill for local) |
| **sklearn.impute.KNNImputer** | Missing value imputation | KNN preserves data relationships better than simple imputation | SimpleImputer (mean/median loses patterns), MICE (slower, complex) |
| **numpy** | Numerical operations | Fast array operations; required by segyio | - |
| **random seed (42)** | Reproducibility | Fixed seed ensures same samples across runs | UUID-based (non-reproducible) |

### Code Structure

#### 1. Main Class: `DataSampler`

##### Initialization

```python
def __init__(self, source_file: str = "Temp/f3_dataset.sgy",
             output_dir: str = "data/raw",
             target_traces: int = 500,
             random_seed: int = 42):
```

- Sets source F3 dataset file path
- Sets output directory for sampled files
- Sets target number of traces to sample (default: 500)
- Sets random seed for reproducibility

##### Method 1: `read_f3_dataset()`

**Purpose**: Read the full F3 dataset and convert to DataFrame.

**Process**:
1. Opens F3 SEGY file with `ignore_geometry=True` (handles non-standard geometry)
2. Reads file metadata (size, number of traces, samples, sample rate)
3. Reads traces in batches (10,000 at a time) to manage memory
4. Extracts trace data and header metadata for each trace
5. Returns DataFrame with all traces

##### Method 2: `random_sample()`

**Purpose**: Randomly sample a subset of traces from the full dataset.

**Process**:
1. Checks if dataset is smaller than target (uses all if smaller)
2. Uses `df.sample()` with fixed random seed for reproducibility
3. Resets index and updates trace_id to be sequential
4. Returns sampled DataFrame

##### Method 3: `detect_missing_values()`

**Purpose**: Detect and report missing values in the DataFrame.

##### Method 4: `impute_missing_values_knn()`

**Purpose**: Impute missing values using KNN (k=2) method.

**Process**:
1. Separates `trace_data` (list column) from numeric columns
2. Identifies numeric columns with missing values
3. Applies `KNNImputer` with k=2 neighbors
4. Updates DataFrame with imputed values
5. Falls back to median imputation if KNN fails

##### Method 5: `split_into_files()`

**Purpose**: Split sampled DataFrame into multiple SEGY files.

**Output**: Creates files like `synthetic_seismic_000.sgy`, `synthetic_seismic_001.sgy`, etc.

##### Method 6: `run_pipeline()`

**Purpose**: Execute complete Stage 0 pipeline.

### Data Flow

```
F3 Dataset (Temp/f3_dataset.sgy)
    ↓
read_f3_dataset() → Full DataFrame (600,515 traces)
    ↓
random_sample() → Sampled DataFrame (500 traces)
    ↓
detect_missing_values() → Missing statistics
    ↓
impute_missing_values_knn(k=2) → Imputed DataFrame
    ↓
split_into_files() → Multiple SEGY files (data/raw/)
    ↓
save_metadata() → dataset_metadata.json
```

### Output Structure

After running:
- `data/raw/synthetic_seismic_000.sgy` to `synthetic_seismic_004.sgy` - Sampled SEGY files
- `data/raw/dataset_metadata.json` - Dataset metadata

---

## Stage 1: Data Ingestion & Quality Assurance

**File:** `stage1_data_ingestion.py`

### Overview

This script implements **Stage 1** of the MLOps pipeline: ingesting SGY/SEGY seismic files, validating quality, and storing the data in Delta Lake/Parquet format. It includes optional LLM integration for intelligent schema analysis.

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **segyio** | Read SGY/SEGY files | Native support for seismic formats; handles F3 geometry issues with `ignore_geometry=True` | ObsPy (seismology-focused), custom parser (time-consuming) |
| **deltalake** | Delta Lake storage | ACID transactions, schema evolution, time-travel WITHOUT Databricks/Spark | PySpark Delta (requires Spark cluster), plain Parquet (no ACID) |
| **pandas** | DataFrame operations | Lightweight local processing; sufficient for 500-trace dataset | PySpark (overkill), Dask (unnecessary complexity) |
| **pyarrow** | Parquet I/O | Fast columnar storage; required by deltalake | fastparquet (slower), pickle (not columnar) |
| **Ollama (llama3.1:8b)** | LLM schema analysis | Local LLM, no API costs, privacy-preserving | OpenAI API (costs, data leaves system), no LLM (less intelligent) |

### Code Structure

#### 1. Imports and Setup

```python
import segyio          # For reading SEG-Y format files
import pandas as pd    # For DataFrame operations
import numpy as np     # For numerical operations
from deltalake import write_deltalake  # For Delta Lake storage
```

- **Optional LLM import**: Tries to import `ollama`; sets `OLLAMA_AVAILABLE` flag if available.

#### 2. Main Class: `SeismicDataIngestion`

##### Initialization

```python
def __init__(self, raw_data_dir: str = "data/raw", 
             output_dir: str = "data/bronze",
             use_llm: bool = False):
```

##### Method 1: `read_segy_file()`

**Purpose**: Read a single SEG-Y file and convert it to a Pandas DataFrame.

**Flow**:
1. Opens SEG-Y file with `segyio.open()`
2. Extracts file ID from filename
3. Reads binary header (sample interval, number of samples)
4. Iterates through traces extracting data and headers
5. Generates class labels deterministically (50% normal, 30% anomaly, 20% boundary)
6. Returns DataFrame with all traces

**Example output row**:
```python
{
    'file_id': '000',
    'trace_id': 0,
    'trace_data': [1.2, 3.4, 5.6, ...],  # amplitude values
    'num_samples': 462,
    'sample_rate': 0.004,
    'inline': 0,
    'crossline': 0,
    'cdp_x': 0,
    'cdp_y': 0,
    'class_label': 0  # Normal
}
```

##### Method 2: `validate_data_quality()`

**Purpose**: Validate the ingested data for quality issues.

**Checks**:
- Total traces count
- Number of unique files
- Sample rate consistency
- Number of samples consistency
- Missing data detection

##### Method 3: `llm_analyze_schema()`

**Purpose**: Use LLM (Ollama) to analyze data schema and provide insights.

**Output**: LLM analysis is saved to `data/bronze/llm_schema_analysis.txt`

##### Method 4: `ingest_all_files()`

**Purpose**: Process all SEG-Y files in the raw data directory.

##### Method 5: `save_to_delta()`

**Purpose**: Save DataFrame to Delta Lake format (without Databricks).

### Data Flow

```
SGY Files (data/raw/*.sgy)
    ↓
read_segy_file() → DataFrame per file
    ↓
ingest_all_files() → Combined DataFrame (all files)
    ↓
validate_data_quality() → Validation results
    ↓
llm_analyze_schema() → LLM analysis (optional)
    ↓
save_to_delta() → Delta Lake + Parquet (data/bronze/)
```

### Output Structure

After running:
- `data/bronze/seismic_data/` - Delta Lake table (partitioned by file_id)
- `data/bronze/seismic_data.parquet` - Parquet backup
- `data/bronze/validation_results.json` - Quality validation report
- `data/bronze/llm_schema_analysis.txt` - LLM schema analysis (if enabled)

---

## AI Quality Agents

**Files:** `ai_quality_agents.py`, `run_quality_agents.py`

### Overview

The AI Quality Agents system provides comprehensive data quality analysis using LLM-powered agents. It analyzes Stage 1 ingestion results and generates detailed quality reports.

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **Ollama (llama3.1:8b)** | LLM-powered analysis | Local inference, no API costs, domain expertise via prompting | GPT-4 API (expensive), rule-based only (less intelligent) |
| **threading** | Timeout handling | Windows-compatible timeout mechanism | signal.alarm (Unix only), asyncio (more complex) |
| **pandas** | Data statistics | Easy statistical aggregation | numpy only (less convenient), scipy (overkill) |
| **json** | Report serialization | Standard format, human-readable | YAML (less common), pickle (not human-readable) |

### Core Components

#### 1. `call_ollama_with_timeout()`

**Purpose**: Windows-compatible timeout wrapper for Ollama LLM calls.

**Features**:
- Sequential Execution: Each call completes before returning
- Timeout Protection: 120-second timeout per call
- Progress Tracking: Shows progress every 30 seconds
- Error Handling: Gracefully handles timeouts and connection errors

#### 2. `LLMStatisticalAnalyzer`

**Purpose**: Performs deep statistical analysis of ingested data using LLM.

**Output**: Statistical analysis with LLM-powered insights on:
- Amplitude range reasonableness
- Coordinate consistency
- Class distribution balance
- Statistical anomalies

#### 3. `LLMDomainValidator`

**Purpose**: Domain-specific seismic data validation using LLM expertise.

**Checks**:
- Trace Amplitude Ranges
- Sample Rate Validation (industry standards: 0.5-4ms)
- Spatial Consistency
- Coordinate Consistency
- Data Completeness

#### 4. `LLMDriftDetector`

**Purpose**: Compares current data with historical patterns to detect drift.

**Drift Indicators**:
- Class Distribution Change
- Amplitude Shift
- Volume Change

#### 5. `DataQualityAgent`

**Purpose**: Main orchestrator that runs all analysis agents.

**Workflow**:
- Step 1/4: Statistical Analysis
- Step 2/4: Domain Validation
- Step 3/4: Drift Detection
- Step 4/4: Action Recommendation

#### 6. `QualityReportGenerator`

**Purpose**: Generates comprehensive quality reports in TXT and MD formats.

### Data Flow

```
Stage 1 Outputs (data/bronze/)
    ↓
Load: seismic_data (Delta/Parquet) + validation_results.json
    ↓
DataQualityAgent.evaluate()
    ↓
Sequential LLM Analyses:
  Step 1: Statistical Analysis → Complete
  Step 2: Domain Validation → Complete
  Step 3: Drift Detection → Complete
  Step 4: Action Recommendation → Complete
    ↓
QualityReportGenerator.generate_reports()
    ↓
Save: quality_report_*.txt + quality_report_*.md (data/bronze/)
```

### Output Structure

After running:
- `data/bronze/quality_report_YYYYMMDD_HHMMSS.txt` - Plain text quality report
- `data/bronze/quality_report_YYYYMMDD_HHMMSS.md` - Markdown quality report
- `quality_registry/historical_stats.json` - Statistics for future drift detection

---

## Stage 2: Feature Engineering with Embeddings

**File:** `stage2_feature_engineering.py`

### Overview

This script implements **Stage 2** of the MLOps pipeline: extracting features from seismic trace data, generating **learned embeddings**, and normalizing them for machine learning.

**Features Generated**:
- **8 handcrafted features**: Statistical and signal characteristics
- **32 embedding features**: PCA-based compressed representations
- **Total: 40 features**

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **sklearn.decomposition.PCA** | Trace embeddings | Lightweight, no GPU required, interpretable (explained variance) | TensorFlow Autoencoder (DLL errors on Windows), PyTorch (same DLL issues), VAE (complex) |
| **sklearn.preprocessing.StandardScaler** | Feature normalization | Standard approach, saves scaler for inference | MinMaxScaler (sensitive to outliers), manual normalization (error-prone) |
| **numpy.fft** | Frequency analysis | Built-in FFT for dominant frequency extraction | scipy.fft (similar), pywt wavelets (overkill) |
| **deltalake** | Feature storage | Consistent with Stage 1; schema evolution support | Parquet only (no versioning), HDF5 (less common in ML) |
| **pickle** | Model serialization | Save scaler/embedder for inference | joblib (similar), ONNX (overkill for scaler) |

**Note on Embeddings**: Initially planned to use TensorFlow/PyTorch autoencoders, but encountered DLL loading errors on Windows. Switched to sklearn PCA which provides similar dimensionality reduction without deep learning dependencies.

### Code Structure

#### 1. Class: `SeismicTraceEmbedder`

**Purpose**: Generate compressed embeddings from seismic traces using PCA or MLP autoencoder.

##### Initialization

```python
def __init__(self, input_dim: int = 462, embedding_dim: int = 32, method: str = 'pca'):
```

- `input_dim`: Number of samples per trace (462 for F3 dataset)
- `embedding_dim`: Dimension of embedding vectors (default: 32)
- `method`: 'pca' or 'mlp_autoencoder'

##### Key Methods:

1. **`fit()`**: Train the embedder on trace data
2. **`encode()`**: Generate embeddings for new traces
3. **`save()`/`load()`**: Persist embedder models

#### 2. Class: `SeismicFeatureExtractor`

**Purpose**: Extract both handcrafted features and learned embeddings from seismic traces.

##### Handcrafted Features (8):
```python
['mean_amplitude', 'std_amplitude', 'min_amplitude', 'max_amplitude',
 'rms_amplitude', 'energy', 'zero_crossings', 'dominant_frequency']
```

##### Embedding Features (32):
```python
['embedding_0', 'embedding_1', ..., 'embedding_31']
```

##### Method: `extract_trace_features()`

**Features Extracted**:

1. **Statistical Features**: mean, std, min, max, rms amplitude
2. **Energy Features**: Total energy (sum of squared amplitudes)
3. **Signal Characteristics**: Zero crossings count
4. **Frequency Features**: Dominant frequency from FFT analysis

#### 3. Class: `FeatureEngineeringPipeline`

**Purpose**: Complete feature engineering pipeline with normalization and storage.

##### Method: `normalize_features()`

**Purpose**: Normalize features using StandardScaler (zero mean, unit variance).

##### Method: `save_features()`

**Output Files**:
- `data/silver/seismic_features/` - Delta Lake table
- `data/silver/seismic_features.parquet` - Parquet backup
- `data/silver/feature_scaler.pkl` - Saved scaler for inference
- `data/silver/embedder/` - Saved embedder model

### Data Flow

```
Bronze Layer (data/bronze/seismic_data)
    ↓
load_data() → DataFrame with trace_data
    ↓
Train embedder on traces → PCA model
    ↓
extract_features_from_dataframe() → 8 handcrafted + 32 embeddings
    ↓
normalize_features() → Normalized Features DataFrame
    ↓
save_features() → Silver Layer (data/silver/)
```

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| mean_amplitude | Average signal strength |
| std_amplitude | Signal variability (noise level indicator) |
| min_amplitude | Minimum signal value |
| max_amplitude | Maximum signal value |
| rms_amplitude | Root Mean Square (energy measure) |
| energy | Total signal energy (sum of squares) |
| zero_crossings | Number of zero crossings (frequency indicator) |
| dominant_frequency | Main frequency component from FFT |
| embedding_0..31 | PCA-compressed trace representations |

### Output Structure

After running:
- `data/silver/seismic_features/` - Delta Lake table with 40 features
- `data/silver/seismic_features.parquet` - Parquet backup
- `data/silver/feature_scaler.pkl` - StandardScaler for inference
- `data/silver/feature_summary.json` - Feature statistics and metadata
- `data/silver/embedder/` - Trained embedder model

---

## RAG Pipeline (LLMOps Component)

**File:** `rag_pipeline.py`

### Overview

This script implements a **Retrieval-Augmented Generation (RAG)** pipeline for semantic search over seismic documentation. It demonstrates LLMOps capabilities.

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **sklearn.TfidfVectorizer** | Text embeddings | Lightweight, no GPU, works offline | sentence-transformers (Keras/TF conflicts), OpenAI embeddings (API costs) |
| **FAISS** | Vector store | Industry-standard, efficient similarity search | Weaviate/Milvus (require server), Chroma (heavier), sklearn cosine (fallback) |
| **Ollama (llama3.1:8b)** | Answer generation | Local LLM, consistent with other stages | OpenAI API (costs), no generation (search only) |
| **json** | Document storage | Simple persistence for document metadata | SQLite (overkill), pickle (not human-readable) |

**Note on Embeddings**: Initially planned to use sentence-transformers for semantic embeddings, but encountered Keras 3 / tf-keras compatibility issues. TF-IDF provides adequate retrieval for domain-specific documentation with simpler dependencies.

### Key Components

#### 1. `SeismicDocumentLoader`

**Purpose**: Load and chunk documents for indexing.

**Chunking Strategy**:
- Default chunk size: 500 characters
- Overlap: 50 characters
- Breaks at sentence boundaries when possible

#### 2. `TfidfEmbeddingModel`

**Purpose**: Generate text embeddings using TF-IDF.

```python
self.vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)
```

#### 3. `FAISSVectorStore` / `SimpleVectorStore`

**Purpose**: Store and search document embeddings.

**FAISS Version**: Efficient for large document collections
**Simple Version**: Fallback using sklearn cosine similarity

#### 4. `SeismicRAGPipeline`

**Purpose**: Complete RAG pipeline orchestration.

**Methods**:
1. **`build_knowledge_base()`**: Index all seismic documents
2. **`query()`**: Answer questions using RAG

### Documents Indexed

1. `dataset_metadata.json` - F3 dataset information
2. `validation_results.json` - Data quality checks
3. `llm_schema_analysis.txt` - LLM analysis from Stage 1
4. `quality_report_*.txt` - AI quality agent reports
5. `feature_summary.json` - Feature engineering summary
6. Domain knowledge about seismic data (built-in)

### Output Structure

After running:
- `data/rag/vector_store/faiss.index` - FAISS index file
- `data/rag/vector_store/documents.json` - Document metadata
- `data/rag/tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer

---

## Feature Store (Feast)

**File:** `feature_store.py`

### Overview

This script implements a **Feature Store** using Feast for centralized feature management. It provides feature versioning, cataloging, and serving capabilities.

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **Feast** | Feature store | Open-source, vendor-agnostic, supports offline/online serving | Tecton (commercial), Hopsworks (heavier), custom solution (time-consuming) |
| **Parquet (FileSource)** | Offline storage | Simple file-based storage, no database required | BigQuery/Redshift (cloud-only), PostgreSQL (requires server) |
| **SQLite** | Online store | Lightweight, no server, good for local development | Redis (requires server), DynamoDB (cloud-only) |
| **pandas** | Feature preparation | Add timestamps, transform data for Feast | PySpark (overkill for local) |

**Note on Feast**: Chosen over alternatives because it's open-source, works locally without cloud infrastructure, and aligns with job requirements for "feature store concepts".

### Key Components

#### 1. `SeismicFeatureStore`

**Capabilities**:
- Feature definitions and schemas
- Feature versioning and lineage
- Offline feature storage (Parquet)
- Online feature serving (SQLite)
- Feature catalog/registry

#### 2. Feature Views

| Feature View | Features | Description |
|--------------|----------|-------------|
| `seismic_statistical_features` | 5 | mean, std, min, max, rms amplitude |
| `seismic_energy_frequency_features` | 3 | energy, zero_crossings, dominant_frequency |
| `seismic_embedding_features` | 32 | PCA embedding dimensions |
| `seismic_labels` | 2 | class_label, file_id |

**Total**: 42 features across 4 feature views

#### 3. Entities

| Entity | Type | Description |
|--------|------|-------------|
| `trace_id` | int64 | Unique trace identifier |
| `file_id` | string | Source file identifier |

### Key Methods

1. **`create_feature_definitions()`**: Generate Feast feature definition file
2. **`prepare_features_for_feast()`**: Add timestamp column for Feast compatibility
3. **`apply_feature_definitions()`**: Register features with Feast registry
4. **`get_historical_features()`**: Retrieve features for training
5. **`materialize_features()`**: Push features to online store
6. **`get_online_features()`**: Retrieve features for inference

### Feature Store vs File Storage

| Capability | File Storage | Feature Store |
|------------|--------------|---------------|
| Feature versioning | ❌ Overwrites | ✅ Time-travel |
| Point-in-time lookups | ❌ No | ✅ Yes |
| Online/offline serving | ❌ Offline only | ✅ Both |
| Feature registry | ❌ JSON summary | ✅ Metadata catalog |
| Feature sharing | ❌ No | ✅ Cross-team |
| Feature lineage | ❌ No | ✅ Data provenance |

### Output Structure

After running:
- `feature_store/feature_store.yaml` - Feast configuration
- `feature_store/feature_definitions.py` - Feature definitions
- `feature_store/feature_catalog.json` - Feature metadata
- `feature_store/registry.db` - Feast registry database
- `data/silver/seismic_features_feast.parquet` - Feast-compatible features

---

## Stage 3: Model Training

**File:** `stage3_model_training.py`

### Overview

This script implements **Stage 3** of the MLOps pipeline: training ML models using features from Stage 2 with MLflow experiment tracking.

**Input**: 40 features (8 handcrafted + 32 PCA embeddings)

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **scikit-learn** | ML models | Lightweight, no GPU required, sufficient for tabular data | PyTorch/TensorFlow (overkill for 500 samples), XGBoost (similar performance) |
| **MLflow** | Experiment tracking | Industry-standard, local file-based tracking, model registry | Weights & Biases (requires account), Neptune (cloud-based), TensorBoard (less features) |
| **Ollama (llama3.1:8b)** | Model recommendation | LLM suggests best model based on dataset characteristics | Manual selection (less intelligent), AutoML (heavier) |
| **pickle** | Model serialization | Simple, works with sklearn | joblib (similar), ONNX (for deployment) |

**Note on MLflow**: Uses local file-based tracking (`file:./mlruns`) instead of MLflow server. This allows full experiment tracking without infrastructure setup, suitable for local development and demonstration.

### Code Structure

#### 1. `ModelTrainer` Class

##### Initialization

```python
def __init__(self, input_dir="data/silver", output_dir="models",
             mlflow_uri=None, use_feature_store=False):
```

##### Key Methods:

1. **`query_llm_for_model()`**: Query Ollama for model recommendation
   - Prompt includes: 40 features, 3 classes, class imbalance info

2. **`load_features()`**: Load features from Stage 2
   - Loads from Delta Lake or Parquet
   - Includes all 40 features (handcrafted + embeddings)
   - Optional: Load from Feature Store

3. **`create_model()`**: Create sklearn model instance
   - Supports: RandomForest, LogisticRegression, DecisionTree, GradientBoosting
   - Uses `class_weight='balanced'` for imbalanced data

4. **`train_model()`**: Train and evaluate model
   - Computes accuracy and F1 scores

5. **`get_feature_importance()`**: Extract feature importance

### Data Split

```
Total: 500 samples
├── Training: 320 (64%)
├── Validation: 80 (16%)
└── Test: 100 (20%)
```

### MLflow Tracking

**Logged Parameters**:
- model_type, num_features, num_handcrafted_features, num_embedding_features
- Model-specific hyperparameters

**Logged Metrics**:
- train_accuracy, val_accuracy, test_accuracy
- train_f1_weighted, val_f1_weighted, test_f1_weighted
- test_precision_weighted, test_recall_weighted

**Logged Artifacts**:
- classification_report.json
- confusion_matrix.json
- feature_importance.json
- Trained model (registered as "SeismicClassifier")

### Output Structure

After running:
- `models/seismic_classifier.pkl` - Trained model
- `models/seismic_classifier_features.json` - Feature names and metadata
- `mlruns/` - MLflow experiment tracking data

---

## Stage 3b: Hyperparameter Tuning (Optuna TPE)

**File:** `stage3_hyperparameter_tuning.py`

### Overview

This script implements **Stage 3b** of the MLOps pipeline: Bayesian hyperparameter optimization using Optuna with TPE (Tree of Parzen Estimators) sampler. It automatically searches for optimal hyperparameters across multiple model types.

**Capabilities**:
- TPE (Tree of Parzen Estimators) Bayesian optimization
- Cross-validation for robust evaluation
- Multi-model optimization (compare RF, GBM, LR)
- MLflow integration for experiment tracking
- LLM-generated optimization insights

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **Optuna** | Hyperparameter optimization | State-of-the-art Bayesian optimization, TPE sampler, pruning support | Hyperopt (similar), GridSearchCV (exhaustive, slow), Ray Tune (heavier) |
| **TPESampler** | Bayesian sampling | Tree of Parzen Estimators - efficient for complex search spaces | RandomSampler (less efficient), CmaEsSampler (continuous only) |
| **MedianPruner** | Early stopping | Prunes unpromising trials based on intermediate values | No pruning (slower), SuccessiveHalvingPruner (more aggressive) |
| **StratifiedKFold** | Cross-validation | Maintains class distribution in folds | KFold (ignores class imbalance), LeaveOneOut (slow) |
| **MLflowCallback** | Experiment logging | Automatic logging of all trials to MLflow | Manual logging (error-prone) |

### Code Structure

#### 1. `HyperparameterTuner` Class

##### Initialization

```python
def __init__(self, input_dir="data/silver", output_dir="models",
             mlflow_uri="file:./mlruns", n_trials=50, cv_folds=5,
             random_state=42):
```

##### Key Methods:

1. **`optimize()`**: Run optimization for single model type
   - Creates TPE sampler with multivariate=True
   - Uses MedianPruner for early stopping
   - Returns best parameters and CV score

2. **`optimize_multiple_models()`**: Compare multiple model types
   - Optimizes RandomForest, GradientBoosting, LogisticRegression
   - Selects overall best model

3. **`_create_objective()`**: Create Optuna objective function
   - Defines hyperparameter search space per model
   - Uses cross-validation for evaluation

4. **`train_best_model()`**: Train with optimized parameters
   - Trains on full training data
   - Evaluates on validation set

5. **`get_optimization_insights()`**: LLM analysis of results

### TPE Algorithm

**Tree of Parzen Estimators** is a Bayesian optimization algorithm:

1. **Startup Phase**: Random sampling for first N trials (default: 10)
2. **TPE Phase**: Models P(x|y) instead of P(y|x)
   - Divides observations into "good" and "bad" based on threshold
   - Samples parameters more likely to produce good results
3. **Multivariate**: Considers correlations between parameters

### Hyperparameter Search Spaces

**RandomForest**:
| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | 50-300 | int |
| max_depth | 3-20 | int |
| min_samples_split | 2-20 | int |
| min_samples_leaf | 1-10 | int |
| max_features | sqrt, log2, None | categorical |
| class_weight | balanced, balanced_subsample, None | categorical |

**GradientBoosting**:
| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | 50-300 | int |
| max_depth | 2-10 | int |
| learning_rate | 0.01-0.3 | float (log) |
| min_samples_split | 2-20 | int |
| subsample | 0.6-1.0 | float |

**LogisticRegression**:
| Parameter | Range | Type |
|-----------|-------|------|
| C | 0.001-100 | float (log) |
| penalty | l1, l2 | categorical |
| class_weight | balanced, None | categorical |

### Data Flow

```
Stage 2 Features (data/silver/)
    ↓
load_features() → 40 features
    ↓
train_test_split() → Training data
    ↓
For each model type:
  ├── Create TPE sampler
  ├── Create objective function
  ├── Run N trials with CV
  └── Log to MLflow
    ↓
Select best model overall
    ↓
Train on full data
    ↓
Save tuned model + results
```

### Example Output

```
Multi-Model Optimization Complete!
Best Model: LogisticRegression
Best CV F1-score: 0.4218

Best Hyperparameters:
  C: 0.369
  penalty: l1
  class_weight: None

Test Results:
  Accuracy: 0.4900
  F1 (weighted): 0.4354
```

### Output Structure

After running:
- `models/seismic_classifier_tuned.pkl` - Best tuned model
- `models/seismic_classifier_tuned_info.json` - Model info and best params
- `models/hyperparameter_tuning_results.json` - Full optimization results
- `mlruns/` - MLflow experiment tracking (all trials)

### Alignment with Job Requirements

Stage 3b demonstrates:
- **Reproducible experiments**: Fixed seeds, logged parameters
- **Experiment tracking (MLflow)**: All trials logged automatically
- **Hyperparameter optimization**: Bayesian optimization with TPE
- **Model selection**: Automated comparison across model types

---

## Stage 4: Model Evaluation

**File:** `stage4_model_evaluation.py`

### Overview

This script implements **Stage 4** of the MLOps pipeline: comprehensive model evaluation with drift detection and LLM-generated reports. It evaluates the trained model from Stage 3 using all 40 features.

**Capabilities**:
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Per-class performance analysis
- Data drift detection using Kolmogorov-Smirnov test
- Feature importance drift analysis
- LLM-generated evaluation reports
- Predictions saved to gold layer

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **scikit-learn metrics** | Evaluation metrics | Comprehensive metric suite, industry standard | Custom metrics (error-prone), TensorFlow metrics (overkill) |
| **scipy.stats.ks_2samp** | Drift detection | Kolmogorov-Smirnov test for distribution comparison | PSI (Population Stability Index), chi-square (categorical only) |
| **MLflow** | Experiment logging | Consistent with Stage 3, tracks evaluation runs | Custom logging (less features), W&B (requires account) |
| **Ollama (llama3.1:8b)** | Report generation | LLM-powered insights, actionable recommendations | Rule-based reports (less intelligent), GPT-4 (API costs) |
| **deltalake** | Predictions storage | Gold layer storage, consistent with pipeline | Parquet only (no versioning), database (overkill) |

### Code Structure

#### 1. `ModelEvaluator` Class

##### Initialization

```python
def __init__(self, features_dir="data/silver", models_dir="models",
             output_dir="data/gold", mlflow_uri="file:./mlruns",
             use_feature_store=False):
```

##### Key Methods:

1. **`load_model()`**: Load trained model and feature names
   - Loads model from pickle
   - Loads feature names from JSON (saved by Stage 3)

2. **`load_features()`**: Load features from Stage 2
   - Uses all 40 features (8 handcrafted + 32 embeddings)
   - Optional: Load from Feature Store

3. **`evaluate_model()`**: Comprehensive evaluation
   - Accuracy, precision, recall, F1 (macro and weighted)
   - Per-class metrics (Normal, Anomaly, Boundary)
   - ROC-AUC (if probabilities available)
   - Confusion matrix

4. **`detect_drift()`**: Data drift detection
   - Kolmogorov-Smirnov test per feature
   - Drift severity classification (none/low/medium/high)
   - Reports drifted feature names

5. **`analyze_feature_importance_drift()`**: Check if important features drifted
   - Identifies top 10 important features
   - Checks if any have significant drift
   - Critical for model reliability assessment

6. **`llm_generate_report()`**: LLM-powered evaluation report
   - Performance assessment (excellent/good/fair/poor)
   - Drift analysis and impact
   - Top 3 recommendations
   - Production readiness assessment

7. **`save_predictions()`**: Save to gold layer
   - Predictions with true labels
   - Class probabilities (if available)
   - Correct/incorrect flags

### Data Flow

```
Stage 3 Outputs (models/, data/silver/)
    ↓
load_model() → Trained model + feature names
    ↓
load_features() → 40 features (same as Stage 3)
    ↓
train_test_split() → Same split as Stage 3 (64/16/20)
    ↓
evaluate_model() → Comprehensive metrics
    ↓
detect_drift() → KS test per feature
    ↓
analyze_feature_importance_drift() → Important feature drift
    ↓
llm_generate_report() → LLM evaluation report
    ↓
save_predictions() → Gold layer (data/gold/)
```

### Drift Detection

**Kolmogorov-Smirnov Test**:
- Compares distribution of each feature between train and test sets
- P-value < 0.05 indicates significant drift
- Severity based on percentage of drifted features:
  - `none`: 0% drifted
  - `low`: < 10% drifted
  - `medium`: 10-30% drifted
  - `high`: > 30% drifted

**Feature Importance Drift**:
- Identifies if top important features have drifted
- Critical warning if important features show drift
- Suggests model may be unreliable

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| accuracy | Overall correct predictions |
| precision_macro | Unweighted mean precision across classes |
| precision_weighted | Weighted by class support |
| recall_macro | Unweighted mean recall across classes |
| recall_weighted | Weighted by class support |
| f1_macro | Unweighted mean F1 across classes |
| f1_weighted | Weighted by class support |
| roc_auc_macro | One-vs-rest ROC-AUC |

### Output Structure

After running:
- `data/gold/predictions/` - Delta Lake table with predictions
- `data/gold/predictions.parquet` - Parquet backup
- `data/gold/evaluation_results.json` - Comprehensive results including:
  - All metrics
  - Drift detection results
  - Feature importance analysis
  - LLM report
  - Data split information

### Example Output

```
Test Set Metrics:
  Accuracy: 0.4100
  F1-score (weighted): 0.4215
  ROC-AUC (macro): 0.5277

Drift Detection:
  Drift detected in 1/40 features (2.5%)
  Drift severity: low
  Drifted features: ['embedding_13_scaled']

Top 5 Most Important Features:
  1. embedding_10_scaled: 0.1009
  2. embedding_15_scaled: 0.0775
  3. embedding_29_scaled: 0.0722

LLM Assessment: Fair performance, needs work before production
```

---

## Stage 5: Model Registry

**File:** `stage5_model_registry.py`

### Overview

This script implements **Stage 5** of the MLOps pipeline: Model Registry and Artifact Management using MLflow. It handles model versioning, artifact storage, and documentation generation.

**Capabilities**:
- MLflow Model Registry integration
- Model versioning with stage transitions
- Artifact management with versioning
- LLM-generated model documentation
- Hyperparameter tracking from Optuna TPE

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **MLflow Model Registry** | Model versioning | Industry standard, integrates with tracking | DVC (more complex), custom registry |
| **MlflowClient** | Registry API access | Programmatic control over registry | REST API (less convenient) |
| **Pickle** | Artifact serialization | Standard Python serialization | Joblib (similar), ONNX (different purpose) |
| **Ollama** | Documentation generation | Local LLM for automated docs | Manual documentation (time-consuming) |

### Code Structure

#### 1. `ModelRegistryManager` Class

##### Key Methods:

1. **`get_latest_model_run()`**: Get most recent MLflow run
2. **`get_best_model_run()`**: Get best run by metric (F1 score)
3. **`register_model_version()`**: Register model in MLflow registry
4. **`transition_model_stage()`**: Move model between stages
5. **`save_artifacts()`**: Save versioned artifacts with metadata
6. **`llm_generate_documentation()`**: Auto-generate model docs

### Model Staging Workflow

```
None → Staging → Production → Archived
  │        │          │
  │        │          └── Old production versions
  │        └── Testing/validation
  └── Newly registered models
```

### Artifact Structure

```
models/SeismicClassifier_v7/
├── model.pkl                           # Trained model
├── model_features.json                 # Feature names and metadata
├── feature_scaler.pkl                  # StandardScaler
├── evaluation_results.json             # Stage 4 metrics
├── hyperparameter_tuning_results.json  # Optuna TPE results
└── metadata.json                       # Version metadata
```

### Output Example

```json
{
  "model_name": "SeismicClassifier",
  "version": 7,
  "num_features": 40,
  "feature_types": {"handcrafted": 8, "embeddings": 32},
  "hyperparameter_tuning": {
    "method": "Optuna TPE",
    "best_model_type": "LogisticRegression",
    "best_cv_score": 0.4218
  }
}
```

---

## Stage 6: Model Deployment

**File:** `stage6_model_deployment.py`

### Overview

This script implements **Stage 6** of the MLOps pipeline: Model Deployment with multiple serving modes. It provides both real-time REST API inference and batch processing capabilities.

**Deployment Modes**:
1. **Real-time API**: FastAPI REST endpoints for online inference
2. **Batch Inference**: CLI for large-scale offline processing

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **FastAPI** | REST API framework | Modern, async, auto-docs | Flask (older), Django (heavier) |
| **Pydantic** | Request/response validation | Type safety, auto-validation | Manual validation |
| **Uvicorn** | ASGI server | High performance, async | Gunicorn (WSGI only) |
| **MLflow** | Model loading | Version control, registry integration | Direct pickle loading |

### Code Structure

#### 1. `ModelServing` Class

##### Key Methods:

1. **`_load_model()`**: Load from file or MLflow registry
2. **`_load_preprocessors()`**: Load scaler and PCA model
3. **`extract_handcrafted_features()`**: Extract 8 statistical features
4. **`extract_embeddings()`**: Generate 32 PCA embeddings
5. **`extract_all_features()`**: Combine all 40 features
6. **`predict()`**: Real-time inference
7. **`batch_predict()`**: Batch inference for large datasets

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check with model status |
| `/model_info` | GET | Model type, version, features |
| `/predict` | POST | Real-time predictions |
| `/batch_predict` | POST | Batch predictions |

### Deployment Modes

#### Mode 1: Real-time API (REST)

```bash
# Start API server
python src/stage6_model_deployment.py

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Request Example:**
```json
POST /predict
{
  "traces": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "include_embeddings": true
}
```

**Response Example:**
```json
{
  "predictions": [
    {
      "trace_id": 0,
      "predicted_class": 0,
      "predicted_class_name": "normal",
      "probabilities": {"normal": 0.75, "anomaly": 0.20, "boundary": 0.05},
      "confidence": 0.75
    }
  ],
  "timestamp": "2026-01-21T10:30:00",
  "model_version": "local-LogisticRegression",
  "num_features": 40
}
```

#### Mode 2: Batch Inference

```bash
# Run batch inference
python src/stage6_model_deployment.py --batch data/silver/seismic_features.parquet predictions.parquet
```

**Output:**
- Predictions saved to Parquet/CSV
- Class probabilities included
- Class distribution summary

### Data Flow

```
Real-time Mode:
  Raw Trace → Extract Features (40) → Scale → Model → Prediction

Batch Mode:
  Parquet (pre-extracted features) → Model → Predictions Parquet
```

### MLflow Integration

The deployment supports loading models from MLflow registry:

```python
# Load production model
serving = ModelServing(model_stage="Production")

# Load staging model for testing
serving = ModelServing(model_stage="Staging")

# Load latest version
serving = ModelServing(model_stage=None)  # Uses local file
```

### Model Deployment Principle

```
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Model Registry                     │
│  ┌─────────┐    ┌─────────┐    ┌────────────┐              │
│  │  None   │ →  │ Staging │ →  │ Production │              │
│  │ (new)   │    │ (test)  │    │  (live)    │              │
│  └─────────┘    └─────────┘    └────────────┘              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Deployment                           │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │   Real-time API  │    │  Batch Inference │              │
│  │    (FastAPI)     │    │     (CLI)        │              │
│  │                  │    │                  │              │
│  │  /predict        │    │  --batch input   │              │
│  │  /health         │    │         output   │              │
│  │  /model_info     │    │                  │              │
│  └──────────────────┘    └──────────────────┘              │
│           │                       │                         │
│           ▼                       ▼                         │
│    Single trace             Large datasets                  │
│    predictions              (Parquet files)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 7: Monitoring & Observability

**File:** `stage7_monitoring.py`

### Overview

This script implements **Stage 7** of the MLOps pipeline: Model Monitoring and Observability. It tracks model performance, detects data drift, and generates alerts.

**Capabilities**:
- Feature-level drift detection (KS test)
- Prediction distribution drift detection
- Prometheus metrics (optional)
- LLM-generated monitoring insights
- Automated alerting

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **scipy.stats** | Statistical tests | KS test for drift detection | Evidently (heavier), custom tests |
| **Prometheus** | Metrics collection | Industry standard, Grafana integration | Custom metrics, StatsD |
| **Ollama** | Alert analysis | LLM-powered insights | Rule-based alerts only |
| **pandas** | Data processing | DataFrame operations | numpy only (less convenient) |

### Drift Detection Methods

| Method | Purpose | Threshold |
|--------|---------|-----------|
| **KS Test (per feature)** | Feature distribution drift | p-value < 0.05 |
| **Chi-square Test** | Prediction distribution drift | p-value < 0.05 |

### Drift Severity Levels

| Severity | Drift Ratio | Action |
|----------|-------------|--------|
| **none** | 0% | No action |
| **low** | 1-10% | Monitor |
| **medium** | 10-30% | Investigate |
| **high** | >30% | Retrain model |

### Output Structure

```json
{
  "timestamp": "2026-01-21T11:21:36",
  "model_info": {
    "model_type": "LogisticRegression",
    "num_features": 40,
    "hyperparameters_source": "optuna_tpe"
  },
  "feature_drift": {
    "features_tested": 40,
    "features_drifted": [],
    "drift_ratio": 0.0,
    "severity": "none"
  },
  "prediction_drift": {
    "drift_detected": false,
    "chi2_pvalue": 0.609
  },
  "alerts": [],
  "llm_analysis": "Overall Health Status: Warning..."
}
```

---

## Stage 8: CI/CD Automation

**File:** `stage8_cicd.py`

### Overview

This script implements **Stage 8** of the MLOps pipeline: CI/CD Automation for ML models. It provides automated validation and can generate GitHub Actions workflows.

**Capabilities**:
- Quick validation (check outputs only)
- Full pipeline execution
- GitHub Actions workflow generation
- Stage-by-stage validation
- Results reporting

### Tools Used

| Tool | Purpose | Why Chosen | Alternative Considered |
|------|---------|------------|------------------------|
| **subprocess** | Command execution | Built-in, cross-platform | os.system (less control) |
| **argparse** | CLI arguments | Built-in, standard | click (external dependency) |
| **GitHub Actions** | CI/CD platform | Industry standard, free | Jenkins (self-hosted), GitLab CI |

### Validation Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Quick** | `python stage8_cicd.py` | Check outputs exist |
| **Full** | `python stage8_cicd.py --full` | Execute all stages |
| **Workflow** | `python stage8_cicd.py --workflow` | Generate GitHub Actions YAML |

### Validated Stages

| Stage | Validation Method |
|-------|-------------------|
| Stage 0 | Check `data/raw/seismic_sample.parquet` |
| Stage 1 | Check `data/bronze/seismic_data/` |
| Stage 2 | Check `data/silver/seismic_features/` |
| Stage 3 | Check `models/seismic_classifier.pkl` |
| Stage 3b | Check `models/hyperparameter_tuning_results.json` |
| Stage 4 | Check `data/gold/evaluation_results.json` |
| Stage 5 | Check `models/SeismicClassifier_v*/` |
| Stage 6 | Check `data/gold/batch_predictions.parquet` |
| Stage 7 | Check `data/gold/monitoring_report.json` |

### Output Example

```
CI/CD Pipeline Summary
============================================================

[OK] Overall Status: SUCCESS
  Duration: 3.58s

Stage Results:
  [OK] stage0_sampling: success
  [OK] stage1_ingestion: success
  [OK] stage2_features: success
  [OK] stage3_training: success
  [OK] stage3b_tuning: success (cv=0.422)
  [OK] stage4_evaluation: success (acc=0.490)
  [OK] stage5_registry: success
  [OK] stage6_deployment: success (n=500)
  [OK] stage7_monitoring: success (alerts=0)
```

---

## Common Patterns Across Stages

### LLM Integration Pattern

All stages that use LLM follow this pattern:
1. Check if LLM is enabled and available
2. Build summary/statistics for LLM
3. Create prompt with context
4. Call LLM with timeout protection
5. Handle errors gracefully (continue without LLM if fails)
6. Save LLM analysis to file (if applicable)

### Data Storage Pattern

All stages follow this storage pattern:
1. Try Delta Lake first (better for large datasets)
2. Fall back to Parquet (better compatibility)
3. Save both formats when possible
4. Partition by `file_id` for performance
5. Save metadata/artifacts alongside data

### Error Handling Pattern

All stages include:
- Try-except blocks for optional features (LLM)
- Graceful degradation (works without optional features)
- Clear error messages
- Progress indicators for long operations

---

## Architecture Summary

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 0: Data Sampling          Stage 1: Data Ingestion                │
│  ┌─────────────────────┐         ┌─────────────────────┐                │
│  │ F3 Dataset (SGY)    │────────▶│ SGY Files           │                │
│  │ Random Sampling     │         │ Quality Validation  │                │
│  │ KNN Imputation      │         │ LLM Schema Analysis │                │
│  └─────────────────────┘         └─────────────────────┘                │
│           │                               │                              │
│           ▼                               ▼                              │
│  ┌─────────────────────┐         ┌─────────────────────┐                │
│  │ data/raw/*.sgy      │         │ data/bronze/        │                │
│  │ dataset_metadata    │         │ Delta Lake + Parquet│                │
│  └─────────────────────┘         └─────────────────────┘                │
│                                           │                              │
│                    ┌──────────────────────┴──────────────────────┐      │
│                    ▼                                             ▼      │
│  ┌─────────────────────┐                        ┌─────────────────────┐ │
│  │ AI Quality Agents   │                        │ Stage 2: Features   │ │
│  │ Statistical Analysis│                        │ 8 Handcrafted       │ │
│  │ Domain Validation   │                        │ 32 PCA Embeddings   │ │
│  │ Drift Detection     │                        │ Normalization       │ │
│  └─────────────────────┘                        └─────────────────────┘ │
│           │                                              │              │
│           ▼                                              ▼              │
│  ┌─────────────────────┐                        ┌─────────────────────┐ │
│  │ Quality Reports     │                        │ data/silver/        │ │
│  │ TXT + MD formats    │                        │ 40 features         │ │
│  └─────────────────────┘                        └─────────────────────┘ │
│                                                          │              │
│                    ┌─────────────────────────────────────┤              │
│                    ▼                                     ▼              │
│  ┌─────────────────────┐                        ┌─────────────────────┐ │
│  │ Feature Store       │                        │ Stage 3: Training   │ │
│  │ Feast Integration   │◀───────────────────────│ MLflow Tracking     │ │
│  │ 4 Feature Views     │                        │ Model Registry      │ │
│  └─────────────────────┘                        └─────────────────────┘ │
│                                                          │              │
│                                                          ▼              │
│                                                 ┌─────────────────────┐ │
│                                                 │ models/             │ │
│                                                 │ seismic_classifier  │ │
│                                                 │ mlruns/             │ │
│                                                 └─────────────────────┘ │
│                                                          │              │
│                                                          ▼              │
│  ┌─────────────────────┐                        ┌─────────────────────┐ │
│  │ RAG Pipeline        │                        │ Stage 4: Evaluation │ │
│  │ TF-IDF Embeddings   │                        │ Drift Detection     │ │
│  │ FAISS Vector Store  │                        │ LLM Reports         │ │
│  └─────────────────────┘                        └─────────────────────┘ │
│                                                          │              │
│                                                          ▼              │
│                                                 ┌─────────────────────┐ │
│                                                 │ data/gold/          │ │
│                                                 │ predictions         │ │
│                                                 │ evaluation_results  │ │
│                                                 └─────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| Data Format | SGY/SEGY (segyio) |
| Data Storage | Delta Lake, Parquet |
| Feature Store | Feast |
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow |
| Hyperparameter Optimization | Optuna (TPE sampler) |
| LLM Integration | Ollama (llama3.1:8b) |
| Vector Store | FAISS |
| Text Embeddings | TF-IDF |
| Drift Detection | scipy.stats (KS test) |
| Model Serving | FastAPI |
| Monitoring | Prometheus |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

### Docker Deployment

The pipeline is fully containerized with Docker Compose:

```yaml
services:
  mlops:      # Main pipeline (FastAPI on :8000, Prometheus on :8001)
  mlflow:     # MLflow UI on :5000
  ollama:     # Optional LLM service on :11434 (--profile llm)
```

**Quick Start:**
```bash
docker-compose up -d
docker-compose exec mlops python src/stage8_cicd.py
```

### Alignment with MLOps Job Requirements

| Requirement | Implementation |
|-------------|----------------|
| Data pipelines for SGY/SEGY | Stage 0, Stage 1 |
| Feature engineering | Stage 2 (40 features) |
| Embeddings | PCA embeddings (32-dim) |
| Experiment tracking | MLflow |
| Model registry | Stage 5 (MLflow Model Registry) |
| Model versioning | Stage 5 (artifact versioning) |
| Feature store | Feast integration |
| Hyperparameter tuning | Stage 3b (Optuna TPE) |
| Model evaluation | Stage 4 (comprehensive metrics) |
| Drift detection | Stage 4, Stage 7 (KS test) |
| Model serving (API) | Stage 6 (FastAPI REST) |
| Batch inference | Stage 6 (CLI batch mode) |
| RAG pipelines | TF-IDF + FAISS |
| LLM integration | Ollama for analysis |
| Data quality | AI Quality Agents |
| Containerization | Docker, Docker Compose |
| CI/CD automation | Stage 8, GitHub Actions |
| Monitoring/Observability | Stage 7 (Prometheus metrics) |
