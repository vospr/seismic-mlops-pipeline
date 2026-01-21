"""
Stage 2: Feature Engineering with Embeddings

Extract features from seismic traces for ML model:
- Statistical features: mean, std, min, max, RMS amplitude
- Frequency features: dominant frequency (FFT)
- Energy features: total energy, zero crossings
- **NEW: PCA-based embeddings** - compressed representations using sklearn
- Total: 8 handcrafted + 32 embedding features = 40 features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import pickle
from deltalake import DeltaTable
import warnings
warnings.filterwarnings('ignore')


class SeismicTraceEmbedder:
    """
    Generate embeddings from seismic traces using sklearn.
    
    Uses a combination of:
    1. PCA for dimensionality reduction
    2. Optional MLP autoencoder for learned representations
    """
    
    def __init__(self, input_dim: int = 462, embedding_dim: int = 32, method: str = 'pca'):
        """
        Initialize embedder.
        
        Args:
            input_dim: Number of samples per trace
            embedding_dim: Dimension of embedding vectors
            method: 'pca' or 'mlp_autoencoder'
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.method = method
        self.is_trained = False
        
        # Normalization
        self.scaler = StandardScaler()
        
        # PCA model
        self.pca = PCA(n_components=embedding_dim)
        
        # MLP autoencoder (encoder + decoder)
        self.encoder = None
        self.decoder = None
        
        print(f"Embedder initialized: {input_dim} -> {embedding_dim} (method: {method})")
    
    def fit(self, traces: np.ndarray, epochs: int = 100) -> Dict[str, Any]:
        """
        Train embedder on trace data.
        
        Args:
            traces: Array of shape (n_samples, input_dim)
            epochs: Number of training epochs (for MLP)
            
        Returns:
            Training summary
        """
        print(f"Training embedder on {len(traces)} traces...")
        
        # Normalize traces
        traces_normalized = self.scaler.fit_transform(traces)
        
        if self.method == 'pca':
            # Fit PCA
            self.pca.fit(traces_normalized)
            
            # Calculate explained variance
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            
            # Calculate reconstruction error
            embeddings = self.pca.transform(traces_normalized)
            reconstructed = self.pca.inverse_transform(embeddings)
            mse = np.mean((traces_normalized - reconstructed) ** 2)
            
            training_summary = {
                'method': 'pca',
                'explained_variance': float(explained_var),
                'reconstruction_mse': float(mse),
                'embedding_dim': self.embedding_dim,
                'n_components': self.pca.n_components_
            }
            
            print(f"PCA training complete.")
            print(f"  Explained variance: {explained_var:.4f}")
            print(f"  Reconstruction MSE: {mse:.6f}")
            
        else:  # mlp_autoencoder
            # Create encoder (input -> embedding)
            hidden_layer_sizes = (256, 128, 64, self.embedding_dim)
            self.encoder = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                max_iter=epochs,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42,
                verbose=False
            )
            
            # Train encoder to predict embeddings (using PCA as target)
            # First get PCA embeddings as targets
            self.pca.fit(traces_normalized)
            pca_embeddings = self.pca.transform(traces_normalized)
            
            # Train encoder
            self.encoder.fit(traces_normalized, pca_embeddings)
            
            # Get encoder embeddings
            embeddings = self.encoder.predict(traces_normalized)
            
            # Train decoder (embedding -> reconstruction)
            self.decoder = MLPRegressor(
                hidden_layer_sizes=(64, 128, 256),
                activation='relu',
                solver='adam',
                max_iter=epochs,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42,
                verbose=False
            )
            self.decoder.fit(embeddings, traces_normalized)
            
            # Calculate reconstruction error
            reconstructed = self.decoder.predict(embeddings)
            mse = np.mean((traces_normalized - reconstructed) ** 2)
            
            training_summary = {
                'method': 'mlp_autoencoder',
                'reconstruction_mse': float(mse),
                'embedding_dim': self.embedding_dim,
                'encoder_loss': float(self.encoder.loss_),
                'decoder_loss': float(self.decoder.loss_)
            }
            
            print(f"MLP autoencoder training complete.")
            print(f"  Reconstruction MSE: {mse:.6f}")
        
        self.is_trained = True
        return training_summary
    
    def encode(self, traces: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for traces.
        
        Args:
            traces: Array of shape (n_samples, input_dim)
            
        Returns:
            Embeddings of shape (n_samples, embedding_dim)
        """
        if not self.is_trained:
            raise RuntimeError("Embedder must be trained before encoding")
        
        # Normalize
        traces_normalized = self.scaler.transform(traces)
        
        if self.method == 'pca':
            embeddings = self.pca.transform(traces_normalized)
        else:
            embeddings = self.encoder.predict(traces_normalized)
        
        return embeddings
    
    def save(self, path: Path):
        """Save embedder models."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        with open(path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save PCA
        with open(path / "pca.pkl", 'wb') as f:
            pickle.dump(self.pca, f)
        
        # Save MLP models if used
        if self.encoder is not None:
            with open(path / "encoder.pkl", 'wb') as f:
                pickle.dump(self.encoder, f)
        if self.decoder is not None:
            with open(path / "decoder.pkl", 'wb') as f:
                pickle.dump(self.decoder, f)
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'method': self.method,
            'is_trained': self.is_trained
        }
        with open(path / "embedder_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Embedder saved to: {path}")
    
    def load(self, path: Path):
        """Load embedder models."""
        path = Path(path)
        
        # Load metadata
        with open(path / "embedder_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.input_dim = metadata['input_dim']
        self.embedding_dim = metadata['embedding_dim']
        self.method = metadata['method']
        self.is_trained = metadata['is_trained']
        
        # Load scaler
        with open(path / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load PCA
        with open(path / "pca.pkl", 'rb') as f:
            self.pca = pickle.load(f)
        
        # Load MLP models if they exist
        encoder_path = path / "encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
        
        decoder_path = path / "decoder.pkl"
        if decoder_path.exists():
            with open(decoder_path, 'rb') as f:
                self.decoder = pickle.load(f)
        
        print(f"Embedder loaded from: {path}")


class SeismicFeatureExtractor:
    """
    Extract features from seismic trace data.
    Includes both handcrafted features and learned embeddings.
    """
    
    def __init__(self, use_embeddings: bool = True, embedding_dim: int = 32,
                 embedding_method: str = 'pca'):
        """
        Initialize feature extractor.
        
        Args:
            use_embeddings: Whether to use embeddings
            embedding_dim: Dimension of embedding vectors
            embedding_method: 'pca' or 'mlp_autoencoder'
        """
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        
        # Handcrafted feature names
        self.handcrafted_features = [
            'mean_amplitude',
            'std_amplitude',
            'min_amplitude',
            'max_amplitude',
            'rms_amplitude',
            'energy',
            'zero_crossings',
            'dominant_frequency'
        ]
        
        # Embedding feature names
        self.embedding_features = [f'embedding_{i}' for i in range(embedding_dim)] if use_embeddings else []
        
        # All feature names
        self.feature_names = self.handcrafted_features + self.embedding_features
        
        # Embedder (initialized later)
        self.embedder = None
    
    def extract_trace_features(self, trace_data: List[float]) -> Dict[str, float]:
        """
        Extract statistical and frequency features from a single trace.
        
        Args:
            trace_data: List of amplitude values for a trace
            
        Returns:
            Dictionary of feature names and values
        """
        trace_array = np.array(trace_data, dtype=np.float32)
        
        # Statistical features
        mean_amp = float(np.mean(trace_array))
        std_amp = float(np.std(trace_array))
        min_amp = float(np.min(trace_array))
        max_amp = float(np.max(trace_array))
        rms_amp = float(np.sqrt(np.mean(trace_array**2)))
        
        # Energy features
        energy = float(np.sum(trace_array**2))
        
        # Zero crossings
        zero_crossings = int(np.sum(np.diff(np.signbit(trace_array))))
        
        # Dominant frequency using FFT
        dominant_freq = self._calculate_dominant_frequency(trace_array)
        
        features = {
            'mean_amplitude': mean_amp,
            'std_amplitude': std_amp,
            'min_amplitude': min_amp,
            'max_amplitude': max_amp,
            'rms_amplitude': rms_amp,
            'energy': energy,
            'zero_crossings': zero_crossings,
            'dominant_frequency': dominant_freq
        }
        
        return features
    
    def _calculate_dominant_frequency(self, trace: np.ndarray) -> float:
        """
        Calculate dominant frequency using FFT.
        
        Args:
            trace: Trace data as numpy array
            
        Returns:
            Dominant frequency in Hz
        """
        # Compute FFT
        fft = np.fft.fft(trace)
        freqs = np.fft.fftfreq(len(trace))
        
        # Get power spectrum
        power = np.abs(fft)**2
        
        # Find dominant frequency (skip DC component)
        # Only look at positive frequencies
        positive_freqs = freqs[1:len(freqs)//2]
        positive_power = power[1:len(power)//2]
        
        if len(positive_power) > 0:
            dominant_idx = np.argmax(positive_power)
            dominant_freq = abs(positive_freqs[dominant_idx])
        else:
            dominant_freq = 0.0
        
        return float(dominant_freq)
    
    def train_embedder(self, df: pd.DataFrame, epochs: int = 100) -> Dict[str, Any]:
        """
        Train embedder on trace data.
        
        Args:
            df: DataFrame with 'trace_data' column
            epochs: Number of training epochs
            
        Returns:
            Training summary
        """
        if not self.use_embeddings:
            return {'status': 'skipped', 'reason': 'embeddings disabled'}
        
        # Extract trace arrays
        traces = np.array([np.array(t, dtype=np.float32) for t in df['trace_data'].values])
        input_dim = traces.shape[1]
        
        print(f"Training embedder on {len(traces)} traces with {input_dim} samples each...")
        
        # Initialize and train embedder
        self.embedder = SeismicTraceEmbedder(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim,
            method=self.embedding_method
        )
        
        training_summary = self.embedder.fit(traces, epochs=epochs)
        
        return training_summary
    
    def extract_features_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from all traces in DataFrame.
        
        Args:
            df: DataFrame with 'trace_data' column
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting handcrafted features from traces...")
        
        # Extract handcrafted features
        features_list = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"  Processing trace {idx}/{len(df)}...")
            
            trace_features = self.extract_trace_features(row['trace_data'])
            features_list.append(trace_features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add embeddings if available
        if self.use_embeddings and self.embedder is not None and self.embedder.is_trained:
            print("Generating embeddings...")
            traces = np.array([np.array(t, dtype=np.float32) for t in df['trace_data'].values])
            embeddings = self.embedder.encode(traces)
            
            # Add embedding columns
            for i in range(self.embedding_dim):
                features_df[f'embedding_{i}'] = embeddings[:, i]
            
            print(f"  Added {self.embedding_dim} embedding features")
        
        # Add metadata columns
        metadata_cols = ['file_id', 'trace_id', 'class_label']
        for col in metadata_cols:
            if col in df.columns:
                features_df[col] = df[col].values
        
        total_features = len(self.handcrafted_features)
        if self.use_embeddings and self.embedder is not None:
            total_features += self.embedding_dim
        
        print(f"Extracted {total_features} features from {len(df)} traces")
        print(f"  - Handcrafted features: {len(self.handcrafted_features)}")
        if self.use_embeddings:
            print(f"  - Embedding features: {self.embedding_dim}")
        
        return features_df


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline with embeddings.
    """
    
    def __init__(self, input_dir: str = "data/bronze",
                 output_dir: str = "data/silver",
                 use_embeddings: bool = True,
                 embedding_dim: int = 32,
                 embedding_method: str = 'pca'):
        """
        Initialize pipeline.
        
        Args:
            input_dir: Input data directory
            output_dir: Output data directory
            use_embeddings: Whether to use embeddings
            embedding_dim: Dimension of embedding vectors
            embedding_method: 'pca' or 'mlp_autoencoder'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        
        self.extractor = SeismicFeatureExtractor(
            use_embeddings=use_embeddings,
            embedding_dim=embedding_dim,
            embedding_method=embedding_method
        )
        self.scaler = StandardScaler()
    
    def load_data(self, table_name: str = "seismic_data") -> pd.DataFrame:
        """
        Load data from Delta Lake or Parquet.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame with trace data
        """
        # Try Delta Lake first
        delta_path = self.input_dir / table_name
        if delta_path.exists():
            print(f"Loading from Delta Lake: {delta_path}")
            dt = DeltaTable(str(delta_path))
            df = dt.to_pandas()
        else:
            # Fall back to Parquet
            parquet_path = self.input_dir / f"{table_name}.parquet"
            if parquet_path.exists():
                print(f"Loading from Parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)
            else:
                raise FileNotFoundError(f"Data not found in {self.input_dir}")
        
        print(f"Loaded {len(df)} traces")
        return df
    
    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            DataFrame with normalized features
        """
        print("Normalizing features...")
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['file_id', 'trace_id', 'class_label']]
        metadata_cols = ['file_id', 'trace_id', 'class_label']
        
        # Separate features and metadata
        X = features_df[feature_cols].values
        metadata = features_df[metadata_cols].copy() if all(col in features_df.columns for col in metadata_cols) else None
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Create normalized DataFrame
        normalized_df = pd.DataFrame(X_scaled, columns=[f"{col}_scaled" for col in feature_cols])
        
        # Add original features
        for col in feature_cols:
            normalized_df[col] = features_df[col].values
        
        # Add metadata back
        if metadata is not None:
            for col in metadata_cols:
                normalized_df[col] = metadata[col].values
        
        print(f"Normalized {len(feature_cols)} features")
        
        return normalized_df
    
    def save_features(self, features_df: pd.DataFrame, table_name: str = "seismic_features"):
        """
        Save features to Delta Lake and Parquet.
        
        Args:
            features_df: DataFrame with features
            table_name: Name of the output table
        """
        from deltalake import write_deltalake
        
        output_path = self.output_dir / table_name
        
        # Save to Delta Lake
        # Delete existing table if schema changed
        import shutil
        if output_path.exists():
            shutil.rmtree(output_path)
        
        write_deltalake(
            str(output_path),
            features_df,
            mode="overwrite",
            partition_by=["file_id"]
        )
        
        print(f"Features saved to Delta Lake: {output_path}")
        
        # Also save as Parquet
        parquet_path = self.output_dir / f"{table_name}.parquet"
        features_df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"Features also saved as Parquet: {parquet_path}")
        
        # Save scaler for later use
        scaler_path = self.output_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save embedder if used
        if self.use_embeddings and self.extractor.embedder is not None:
            embedder_path = self.output_dir / "embedder"
            self.extractor.embedder.save(embedder_path)
        
        return output_path
    
    def run_pipeline(self, embedder_epochs: int = 100):
        """
        Execute complete feature engineering pipeline.
        
        Args:
            embedder_epochs: Number of epochs for embedder training
            
        Returns:
            DataFrame with processed features
        """
        # Load data
        df = self.load_data()
        
        # Train embedder if enabled
        embedder_summary = None
        if self.use_embeddings:
            print("\n" + "=" * 40)
            print(f"Training Embedder ({self.embedding_method.upper()})")
            print("=" * 40)
            embedder_summary = self.extractor.train_embedder(df, epochs=embedder_epochs)
        
        # Extract features (handcrafted + embeddings)
        print("\n" + "=" * 40)
        print("Extracting Features")
        print("=" * 40)
        features_df = self.extractor.extract_features_from_dataframe(df)
        
        # Normalize features
        normalized_df = self.normalize_features(features_df)
        
        # Save features
        output_path = self.save_features(normalized_df)
        
        # Build feature summary
        feature_cols = [col for col in normalized_df.columns 
                       if col not in ['file_id', 'trace_id', 'class_label'] 
                       and not col.endswith('_scaled')]
        
        summary = {
            'num_features': len(feature_cols),
            'handcrafted_features': len(self.extractor.handcrafted_features),
            'embedding_features': self.embedding_dim if self.use_embeddings else 0,
            'embedding_method': self.embedding_method if self.use_embeddings else None,
            'feature_names': feature_cols,
            'num_samples': len(normalized_df),
            'embeddings_enabled': self.use_embeddings,
            'embedder_summary': embedder_summary,
            'feature_statistics': {
                col: {
                    'mean': float(normalized_df[col].mean()),
                    'std': float(normalized_df[col].std()),
                    'min': float(normalized_df[col].min()),
                    'max': float(normalized_df[col].max())
                }
                for col in feature_cols
            }
        }
        
        summary_path = self.output_dir / "feature_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nFeature summary saved to: {summary_path}")
        
        return normalized_df


def main():
    """Execute Stage 2: Feature Engineering with Embeddings."""
    print("=" * 60)
    print("Stage 2: Feature Engineering with Embeddings")
    print("=" * 60)
    
    print("Using sklearn-based embeddings (PCA + optional MLP)")
    print("Embeddings: ENABLED")
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(
        input_dir="data/bronze",
        output_dir="data/silver",
        use_embeddings=True,
        embedding_dim=32,
        embedding_method='pca'  # Use 'mlp_autoencoder' for neural network version
    )
    
    # Run pipeline
    features_df = pipeline.run_pipeline(embedder_epochs=100)
    
    print(f"\n[SUCCESS] Stage 2 complete!")
    print(f"  Handcrafted features: {len(pipeline.extractor.handcrafted_features)}")
    if pipeline.use_embeddings:
        print(f"  Embedding features: {pipeline.embedding_dim}")
        print(f"  Embedding method: {pipeline.embedding_method}")
    print(f"  Total features: {len(pipeline.extractor.feature_names)}")
    print(f"  Samples processed: {len(features_df)}")
    print(f"  Output directory: {pipeline.output_dir}")


if __name__ == "__main__":
    main()
