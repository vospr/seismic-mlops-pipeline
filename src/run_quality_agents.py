"""
Run AI Quality Agents on Stage 1 Results

This script:
1. Loads data from Stage 1 (bronze layer)
2. Loads validation results from Stage 1
3. Runs all AI quality agents
4. Generates comprehensive reports in seismic_data/ folder
"""
import pandas as pd
import json
from pathlib import Path
from typing import Optional
from ai_quality_agents import DataQualityAgent, QualityReportGenerator
from deltalake import DeltaTable


def load_stage1_data(bronze_dir: str = "data/bronze") -> pd.DataFrame:
    """
    Load data from Stage 1 bronze layer.
    
    Args:
        bronze_dir: Directory containing Stage 1 output
        
    Returns:
        DataFrame with ingested data
    """
    bronze_path = Path(bronze_dir)
    
    # Try to load from Delta Lake first
    delta_path = bronze_path / "seismic_data"
    if delta_path.exists():
        try:
            print(f"Loading from Delta Lake: {delta_path}")
            delta_table = DeltaTable(str(delta_path))
            df = delta_table.to_pandas()
            print(f"  Loaded {len(df)} traces from Delta Lake")
            return df
        except Exception as e:
            print(f"  Failed to load from Delta Lake: {e}")
            print("  Trying Parquet...")
    
    # Fallback to Parquet
    parquet_path = bronze_path / "seismic_data.parquet"
    if parquet_path.exists():
        print(f"Loading from Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df)} traces from Parquet")
        return df
    
    raise FileNotFoundError(
        f"Could not find Stage 1 data in {bronze_dir}. "
        "Please run Stage 1 first."
    )


def load_historical_stats(registry_path: str = "quality_registry") -> Optional[dict]:
    """
    Load historical statistics for drift detection.
    
    Args:
        registry_path: Path to quality registry
        
    Returns:
        Historical statistics dictionary or None
    """
    registry_dir = Path(registry_path)
    if not registry_dir.exists():
        return None
    
    # Look for most recent historical stats
    # In a real implementation, this would query a time-series database
    # For now, we'll check for a simple JSON file
    stats_file = registry_dir / "historical_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            return json.load(f)
    
    return None


def save_current_stats(df: pd.DataFrame, registry_path: str = "quality_registry"):
    """
    Save current statistics for future drift detection.
    
    Args:
        df: Current DataFrame
        registry_path: Path to quality registry
    """
    registry_dir = Path(registry_path)
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    stats = {
        'total_traces': len(df),
        'total_files': int(df['file_id'].nunique()) if 'file_id' in df.columns else 0,
    }
    
    if 'class_label' in df.columns:
        class_counts = df['class_label'].value_counts()
        stats['class_distribution'] = (class_counts / len(df)).to_dict()
    
    # Sample trace statistics
    if 'trace_data' in df.columns:
        import numpy as np
        sample_traces = df['trace_data'].head(50)
        amplitudes = []
        for trace in sample_traces:
            if isinstance(trace, list):
                amplitudes.extend(trace)
        
        if amplitudes:
            stats['mean_amplitude'] = float(np.mean(amplitudes))
            stats['std_amplitude'] = float(np.std(amplitudes))
    
    # Save to file
    stats_file = registry_dir / "historical_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Saved current statistics to {stats_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("AI Quality Agents - Data Quality Evaluation")
    print("=" * 80)
    print()
    
    # Configuration
    bronze_dir = "data/bronze"
    validation_results_path = "data/bronze/validation_results.json"
    output_dir = "data/bronze"  # Save reports to bronze folder
    registry_path = "quality_registry"
    
    # Load Stage 1 data
    print("Step 1: Loading Stage 1 data...")
    try:
        df = load_stage1_data(bronze_dir)
        print(f"  [OK] Loaded {len(df)} traces\n")
    except Exception as e:
        print(f"  [ERROR] Failed to load data: {e}\n")
        return
    
    # Load historical stats (if available)
    print("Step 2: Loading historical statistics...")
    historical_stats = load_historical_stats(registry_path)
    if historical_stats:
        print(f"  [OK] Loaded historical statistics\n")
    else:
        print(f"  [INFO] No historical data available (first run)\n")
    
    # Initialize quality agent
    print("Step 3: Initializing AI Quality Agent...")
    quality_agent = DataQualityAgent(
        validation_results_path=validation_results_path,
        output_dir=output_dir,
        use_llm=True  # Set to False if Ollama not available
    )
    print("  [OK] Agent initialized\n")
    
    # Run evaluation
    print("Step 4: Running comprehensive quality evaluation...")
    print("  NOTE: This may take 5-10 minutes (4 sequential LLM calls)")
    print("  Each call completes before the next one starts\n")
    quality_report = quality_agent.evaluate(df, historical_stats)
    print("  [OK] Evaluation complete\n")
    
    # Generate reports
    print("Step 5: Generating reports...")
    report_generator = QualityReportGenerator(output_dir=output_dir)
    reports = report_generator.generate_reports(quality_report, format="both")
    
    print(f"  [OK] Reports generated:")
    for fmt, path in reports.items():
        print(f"    - {fmt.upper()}: {path}")
    print()
    
    # Save current stats for future drift detection
    print("Step 6: Saving current statistics...")
    save_current_stats(df, registry_path)
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Priority: {quality_report['priority']}")
    print(f"Recommended Action: {quality_report['recommended_action']['action']}")
    print(f"Reasoning: {quality_report['recommended_action'].get('reasoning', 'N/A')[:100]}...")
    print()
    print(f"Reports saved to: {output_dir}/")
    print("=" * 80)
    print("\n[SUCCESS] Quality agents execution completed!")
    print("All reports have been generated successfully.\n")


if __name__ == "__main__":
    main()
