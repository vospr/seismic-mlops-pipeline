"""
Stage 0: Data Sampling & Preprocessing

Reads F3 dataset from Temp folder, performs random sampling to create
a manageable dataset size, and imputes missing values using KNN (k=2).

This stage prepares the data for Stage 1 ingestion.
"""
import segyio
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataSampler:
    """
    Sample and preprocess F3 dataset for pipeline testing.
    """
    
    def __init__(self, source_file: str = "Temp/f3_dataset.sgy",
                 output_dir: str = "data/raw",
                 target_traces: int = 500,
                 random_seed: int = 42):
        """
        Initialize data sampler.
        
        Args:
            source_file: Path to F3 dataset SEGY file
            output_dir: Output directory for sampled files
            target_traces: Target number of traces to sample
            random_seed: Random seed for reproducibility
        """
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_traces = target_traces
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def read_f3_dataset(self) -> pd.DataFrame:
        """
        Read F3 dataset and convert to DataFrame.
        
        Returns:
            DataFrame with all traces from F3 dataset
        """
        print("=" * 60)
        print("Reading F3 Dataset")
        print("=" * 60)
        print(f"Source file: {self.source_file}")
        
        if not self.source_file.exists():
            raise FileNotFoundError(f"F3 dataset not found: {self.source_file}")
        
        file_size_mb = self.source_file.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        rows = []
        
        try:
            with segyio.open(str(self.source_file), ignore_geometry=True) as segy:
                num_traces = len(segy.trace)
                num_samples = len(segy.samples)
                sample_interval_us = segy.bin[segyio.BinField.Interval]
                sample_rate_ms = sample_interval_us / 1000.0
                
                print(f"Total traces: {num_traces:,}")
                print(f"Samples per trace: {num_samples}")
                print(f"Sample rate: {sample_rate_ms:.2f} ms")
                print(f"\nReading traces (this may take a few minutes)...")
                
                # Read traces in batches to manage memory
                batch_size = 10000
                for batch_start in range(0, num_traces, batch_size):
                    batch_end = min(batch_start + batch_size, num_traces)
                    
                    if batch_start % 50000 == 0:
                        print(f"  Progress: {batch_start:,}/{num_traces:,} traces ({batch_start/num_traces*100:.1f}%)")
                    
                    for trace_idx in range(batch_start, batch_end):
                        try:
                            trace_data = segy.trace[trace_idx]
                            header = segy.header[trace_idx]
                            
                            # Extract metadata
                            row = {
                                'file_id': 'f3',  # Single file
                                'trace_id': trace_idx,
                                'trace_data': trace_data.tolist(),
                                'num_samples': num_samples,
                                'sample_rate': sample_rate_ms,
                                'inline': header.get(segyio.TraceField.INLINE_3D, None),
                                'crossline': header.get(segyio.TraceField.CROSSLINE_3D, None),
                                'cdp_x': header.get(segyio.TraceField.CDP_X, None),
                                'cdp_y': header.get(segyio.TraceField.CDP_Y, None),
                                'field_record': header.get(segyio.TraceField.FieldRecord, None),
                                'trace_number': header.get(segyio.TraceField.TraceNumber, None),
                            }
                            
                            rows.append(row)
                        except Exception as e:
                            print(f"  Warning: Error reading trace {trace_idx}: {e}")
                            continue
                
                print(f"  Completed: {len(rows):,} traces read")
        
        except Exception as e:
            print(f"ERROR reading F3 dataset: {e}")
            raise
        
        df = pd.DataFrame(rows)
        print(f"\nDataFrame created: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def random_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly sample traces from DataFrame.
        
        Args:
            df: Full DataFrame
            
        Returns:
            Sampled DataFrame
        """
        print("\n" + "=" * 60)
        print("Random Sampling")
        print("=" * 60)
        print(f"Total traces: {len(df):,}")
        print(f"Target traces: {self.target_traces}")
        
        if len(df) <= self.target_traces:
            print("  Dataset is smaller than target, using all traces")
            return df
        
        # Random sampling
        sampled_df = df.sample(n=self.target_traces, random_state=self.random_seed)
        sampled_df = sampled_df.reset_index(drop=True)
        
        # Update trace_id to be sequential
        sampled_df['trace_id'] = range(len(sampled_df))
        
        print(f"  Sampled {len(sampled_df)} traces")
        print(f"  Sampling ratio: {len(sampled_df)/len(df)*100:.2f}%")
        
        return sampled_df
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect missing values in DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with missing value statistics
        """
        print("\n" + "=" * 60)
        print("Missing Value Detection")
        print("=" * 60)
        
        missing_stats = {}
        
        for col in df.columns:
            if col == 'trace_data':
                # Check for None or empty lists in trace_data
                missing_count = df[col].isna().sum() + df[col].apply(
                    lambda x: x is None or (isinstance(x, list) and len(x) == 0)
                ).sum()
            else:
                missing_count = df[col].isna().sum()
            
            if missing_count > 0:
                missing_stats[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
                print(f"  {col}: {missing_count} missing ({missing_count/len(df)*100:.2f}%)")
        
        if not missing_stats:
            print("  No missing values detected")
        
        return missing_stats
    
    def impute_missing_values_knn(self, df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
        """
        Impute missing values using KNN (k=2).
        
        Args:
            df: DataFrame with missing values
            k: Number of neighbors for KNN
            
        Returns:
            DataFrame with imputed values
        """
        print("\n" + "=" * 60)
        print(f"KNN Imputation (k={k})")
        print("=" * 60)
        
        df_imputed = df.copy()
        
        # Separate trace_data (list column) from numeric columns
        trace_data_col = df_imputed['trace_data'].copy()
        numeric_cols = ['num_samples', 'sample_rate', 'inline', 'crossline', 
                       'cdp_x', 'cdp_y', 'field_record', 'trace_number']
        
        # Get numeric columns that exist in DataFrame
        available_numeric_cols = [col for col in numeric_cols if col in df_imputed.columns]
        
        if not available_numeric_cols:
            print("  No numeric columns to impute")
            return df_imputed
        
        # Check for missing values in numeric columns
        numeric_df = df_imputed[available_numeric_cols].copy()
        missing_before = numeric_df.isna().sum().sum()
        
        if missing_before == 0:
            print("  No missing values in numeric columns")
            return df_imputed
        
        print(f"  Missing values before imputation: {missing_before}")
        
        # Apply KNN imputation
        try:
            imputer = KNNImputer(n_neighbors=k, weights='uniform')
            numeric_imputed = imputer.fit_transform(numeric_df)
            
            # Update DataFrame with imputed values
            for i, col in enumerate(available_numeric_cols):
                df_imputed[col] = numeric_imputed[:, i]
            
            missing_after = pd.DataFrame(numeric_imputed, columns=available_numeric_cols).isna().sum().sum()
            print(f"  Missing values after imputation: {missing_after}")
            print(f"  Imputed: {missing_before - missing_after} values")
            
        except Exception as e:
            print(f"  Warning: KNN imputation failed: {e}")
            print(f"  Falling back to median imputation")
            # Fallback to median imputation
            for col in available_numeric_cols:
                if df_imputed[col].isna().any():
                    median_val = df_imputed[col].median()
                    df_imputed[col].fillna(median_val, inplace=True)
                    print(f"    {col}: filled {df_imputed[col].isna().sum()} missing with median {median_val}")
        
        # Handle missing trace_data (if any)
        if trace_data_col.isna().any() or trace_data_col.apply(
            lambda x: x is None or (isinstance(x, list) and len(x) == 0)
        ).any():
            print("  Warning: Some trace_data is missing or empty")
            print("  Removing rows with invalid trace_data")
            valid_mask = trace_data_col.apply(
                lambda x: x is not None and isinstance(x, list) and len(x) > 0
            )
            df_imputed = df_imputed[valid_mask].reset_index(drop=True)
            print(f"  Remaining traces: {len(df_imputed)}")
        
        return df_imputed
    
    def split_into_files(self, df: pd.DataFrame, traces_per_file: int = 100) -> List[Path]:
        """
        Split sampled DataFrame into multiple SEGY files.
        
        Args:
            df: Sampled DataFrame
            traces_per_file: Number of traces per output file
            
        Returns:
            List of output file paths
        """
        print("\n" + "=" * 60)
        print("Splitting into Files")
        print("=" * 60)
        
        num_files = (len(df) + traces_per_file - 1) // traces_per_file
        print(f"Total traces: {len(df)}")
        print(f"Traces per file: {traces_per_file}")
        print(f"Number of files: {num_files}")
        
        output_files = []
        
        for file_idx in range(num_files):
            start_idx = file_idx * traces_per_file
            end_idx = min(start_idx + traces_per_file, len(df))
            file_df = df.iloc[start_idx:end_idx].copy()
            
            # Class labels already added to df, just use them
            # (class_label column should already exist)
            
            # Save as SEGY file
            output_file = self.output_dir / f"synthetic_seismic_{file_idx:03d}.sgy"
            self._save_as_segy(file_df, output_file, file_idx)
            output_files.append(output_file)
            
            print(f"  Created {output_file.name}: {len(file_df)} traces")
        
        return output_files
    
    def _generate_class_label(self, trace_id: int) -> int:
        """
        Generate deterministic class label (same logic as Stage 1).
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Class label (0=normal, 1=anomaly, 2=boundary)
        """
        np.random.seed(trace_id)
        rand = np.random.random()
        if rand < 0.5:
            return 0  # Normal
        elif rand < 0.8:
            return 1  # Anomaly
        else:
            return 2  # Boundary
    
    def _save_as_segy(self, df: pd.DataFrame, output_file: Path, file_id: int):
        """
        Save DataFrame as SEGY file.
        
        Args:
            df: DataFrame with trace data
            output_file: Output file path
            file_id: File ID
        """
        if len(df) == 0:
            return
        
        # Get sample info from first row
        num_samples = int(df['num_samples'].iloc[0])
        sample_rate_ms = float(df['sample_rate'].iloc[0])
        
        # Create SEGY file
        spec = segyio.spec()
        spec.sorting = 1  # CDP sorting
        spec.format = 5  # 4-byte IEEE float
        spec.samples = list(range(num_samples))
        spec.tracecount = len(df)
        
        with segyio.create(str(output_file), spec) as f:
            # Write binary header
            f.bin[segyio.BinField.Interval] = int(sample_rate_ms * 1000)  # microseconds
            f.bin[segyio.BinField.Samples] = num_samples
            f.bin[segyio.BinField.Format] = 5  # IEEE float
            
            # Write text header
            text_header = segyio.tools.create_text_header({
                'client': 'MLOps Pipeline - Stage 0',
                'survey': 'F3 Dataset Sampled',
                'file_id': str(file_id),
                'num_traces': str(len(df)),
                'num_samples': str(num_samples),
                'sample_rate': f'{sample_rate_ms}ms',
            })
            f.text[0] = text_header
            
            # Write traces
            df_reset = df.reset_index(drop=True)
            for i in range(len(df_reset)):
                row = df_reset.iloc[i]
                trace_data = np.array(row['trace_data'], dtype=np.float32)
                f.trace[i] = trace_data
                
                # Write header
                f.header[i] = {
                    segyio.TraceField.INLINE_3D: int(row['inline']) if pd.notna(row['inline']) else 0,
                    segyio.TraceField.CROSSLINE_3D: int(row['crossline']) if pd.notna(row['crossline']) else 0,
                    segyio.TraceField.CDP_X: int(row['cdp_x']) if pd.notna(row['cdp_x']) else 0,
                    segyio.TraceField.CDP_Y: int(row['cdp_y']) if pd.notna(row['cdp_y']) else 0,
                    segyio.TraceField.FieldRecord: int(row['field_record']) if pd.notna(row['field_record']) else 0,
                    segyio.TraceField.TraceNumber: int(row['trace_number']) if pd.notna(row['trace_number']) else i + 1,
                }
    
    def save_metadata(self, df: pd.DataFrame, output_files: List[Path]):
        """
        Save dataset metadata.
        
        Args:
            df: Final sampled DataFrame
            output_files: List of output file paths
        """
        metadata = {
            'num_files': len(output_files),
            'num_traces_per_file': len(df) // len(output_files) if output_files else 0,
            'total_traces': len(df),
            'num_samples': int(df['num_samples'].iloc[0]) if len(df) > 0 else 0,
            'sample_rate_ms': float(df['sample_rate'].iloc[0]) if len(df) > 0 else 0.0,
            'classes': ['normal', 'anomaly', 'boundary'],
            'source': str(self.source_file),
            'sampling_ratio': f"{len(df)/600515*100:.2f}%" if len(df) < 600515 else "100%",
            'files': {}
        }
        
        # Add file-specific metadata
        for file_path in output_files:
            file_id = file_path.stem.split('_')[-1]
            file_df = df[df['file_id'] == 'f3']  # All from same source
            # This is simplified - in practice, you'd track which traces went to which file
            metadata['files'][str(file_path)] = {
                'num_traces': len(file_df) if len(output_files) == 1 else len(df) // len(output_files),
                'class_distribution': {
                    'normal': int((df['class_label'] == 0).sum() / len(output_files)),
                    'anomaly': int((df['class_label'] == 1).sum() / len(output_files)),
                    'boundary': int((df['class_label'] == 2).sum() / len(output_files))
                }
            }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\nMetadata saved to: {metadata_path}")
        return metadata_path
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete Stage 0 pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        print("=" * 60)
        print("Stage 0: Data Sampling & Preprocessing")
        print("=" * 60)
        
        # Step 1: Read F3 dataset
        df_full = self.read_f3_dataset()
        
        # Step 2: Random sampling
        df_sampled = self.random_sample(df_full)
        
        # Step 3: Detect missing values
        missing_stats = self.detect_missing_values(df_sampled)
        
        # Step 4: KNN imputation (k=2)
        df_imputed = self.impute_missing_values_knn(df_sampled, k=2)
        
        # Step 5: Add class labels to main DataFrame before splitting
        df_imputed['class_label'] = df_imputed.apply(
            lambda row: self._generate_class_label(row['trace_id']), axis=1
        )
        
        # Step 6: Split into files
        output_files = self.split_into_files(df_imputed, traces_per_file=100)
        
        # Step 7: Save metadata
        metadata_path = self.save_metadata(df_imputed, output_files)
        
        print("\n" + "=" * 60)
        print("Stage 0 Complete!")
        print("=" * 60)
        print(f"  Source: {self.source_file}")
        print(f"  Sampled traces: {len(df_imputed)}")
        print(f"  Output files: {len(output_files)}")
        print(f"  Output directory: {self.output_dir.absolute()}")
        print(f"  Missing values imputed: {len(missing_stats)} columns")
        
        return {
            'source_file': str(self.source_file),
            'sampled_traces': len(df_imputed),
            'output_files': [str(f) for f in output_files],
            'metadata_path': str(metadata_path),
            'missing_values_imputed': missing_stats
        }


def main():
    """Execute Stage 0: Data Sampling & Preprocessing."""
    sampler = DataSampler(
        source_file="Temp/f3_dataset.sgy",
        output_dir="data/raw",
        target_traces=500,  # Similar to current test dataset
        random_seed=42
    )
    
    results = sampler.run_pipeline()
    
    print("\nNext step: Run Stage 1 (Data Ingestion)")
    print("  python src/stage1_data_ingestion.py")


if __name__ == "__main__":
    main()
