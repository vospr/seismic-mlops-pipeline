"""
Generate synthetic SGY/SEGY test data files for MLOps pipeline testing.

This script creates realistic SEG-Y format files with:
- Multiple traces per file
- Realistic amplitude patterns
- Binary and text headers
- Trace headers with metadata
- Labeled data for classification (3 classes)
"""
import segyio
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple


class SyntheticSeismicGenerator:
    """
    Generate synthetic seismic data in SEG-Y format.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # SEG-Y parameters
        self.num_samples = 2000  # Samples per trace
        self.sample_rate = 0.002  # 2ms sample interval
        self.num_traces_per_file = 100  # Traces per file
        
        # Class labels: 0=normal, 1=anomaly, 2=boundary
        self.classes = ["normal", "anomaly", "boundary"]
    
    def generate_trace(self, trace_id: int, class_label: int, 
                      file_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Generate a single seismic trace with realistic patterns.
        
        Args:
            trace_id: Unique trace identifier
            class_label: Class label (0=normal, 1=anomaly, 2=boundary)
            file_id: File identifier
            
        Returns:
            Tuple of (trace_data, trace_header_dict)
        """
        # Base signal: sine wave with noise
        t = np.linspace(0, self.num_samples * self.sample_rate, self.num_samples)
        
        # Generate different patterns based on class
        if class_label == 0:  # Normal
            # Regular sine wave with low noise
            frequency = 20 + np.random.uniform(-5, 5)  # 15-25 Hz
            amplitude = 1000 + np.random.uniform(-200, 200)
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
            noise = np.random.normal(0, 50, self.num_samples)
            
        elif class_label == 1:  # Anomaly
            # Irregular pattern with high amplitude spikes
            frequency = 30 + np.random.uniform(-10, 10)  # 20-40 Hz
            amplitude = 1500 + np.random.uniform(-300, 300)
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
            # Add spikes
            spike_positions = np.random.choice(self.num_samples, size=5, replace=False)
            signal[spike_positions] += np.random.uniform(500, 1000, 5)
            noise = np.random.normal(0, 100, self.num_samples)
            
        else:  # Boundary (class_label == 2)
            # Transition pattern: frequency changes mid-trace
            mid_point = self.num_samples // 2
            freq1 = 15 + np.random.uniform(-3, 3)
            freq2 = 35 + np.random.uniform(-5, 5)
            amplitude = 1200 + np.random.uniform(-200, 200)
            
            signal = np.zeros(self.num_samples)
            signal[:mid_point] = amplitude * np.sin(2 * np.pi * freq1 * t[:mid_point])
            signal[mid_point:] = amplitude * np.sin(2 * np.pi * freq2 * t[mid_point:])
            noise = np.random.normal(0, 75, self.num_samples)
        
        # Combine signal and noise
        trace_data = signal + noise
        
        # Convert to 32-bit float (SEG-Y standard)
        trace_data = trace_data.astype(np.float32)
        
        # Create trace header (minimal required fields)
        trace_header = {
            segyio.TraceField.FieldRecord: file_id,
            segyio.TraceField.TraceNumber: trace_id + 1,
            segyio.TraceField.EnergySourcePoint: file_id * 1000 + trace_id,
            segyio.TraceField.CDP: trace_id + 1,
            segyio.TraceField.CDP_TRACE: trace_id + 1,
            segyio.TraceField.DataUse: 1,  # Production
            segyio.TraceField.ElevationScalar: -100,
            segyio.TraceField.CDP_X: trace_id * 25,  # X coordinate
            segyio.TraceField.CDP_Y: file_id * 1000,  # Y coordinate
            segyio.TraceField.CoordinateUnits: 1,  # Meters
            segyio.TraceField.DelayRecordingTime: 0,
            segyio.TraceField.Correlated: 1,  # Yes
            # 3D seismic fields
            segyio.TraceField.INLINE_3D: file_id,
            segyio.TraceField.CROSSLINE_3D: trace_id,
            # Store class label in UnassignedInt1 field (we'll extract this later)
            # Note: We'll store labels separately in metadata JSON
        }
        
        return trace_data, trace_header
    
    def create_segy_file(self, file_id: int, num_traces: int = None) -> Tuple[str, List[int]]:
        """
        Create a SEG-Y file with synthetic traces.
        
        Args:
            file_id: File identifier
            num_traces: Number of traces (default: self.num_traces_per_file)
            
        Returns:
            Tuple of (file_path, list_of_class_labels)
        """
        if num_traces is None:
            num_traces = self.num_traces_per_file
        
        file_path = self.output_dir / f"synthetic_seismic_{file_id:03d}.sgy"
        
        # Generate class labels (mix of all classes)
        class_labels = []
        for i in range(num_traces):
            # Mix: 50% normal, 30% anomaly, 20% boundary
            rand = np.random.random()
            if rand < 0.5:
                class_labels.append(0)  # Normal
            elif rand < 0.8:
                class_labels.append(1)  # Anomaly
            else:
                class_labels.append(2)  # Boundary
        
        # Create SEG-Y file using segyio
        spec = segyio.spec()
        spec.format = 5  # IEEE floating point
        spec.sorting = 1  # CDP sorting
        spec.samples = list(range(self.num_samples))  # Sample indices
        spec.tracecount = num_traces
        
        with segyio.create(str(file_path), spec) as f:
            # Write text header (3200 bytes)
            text_header = segyio.tools.create_text_header({
                'client': 'MLOps Test Pipeline',
                'survey': 'Synthetic Seismic Data',
                'file_id': str(file_id),
                'num_traces': str(num_traces),
                'num_samples': str(self.num_samples),
                'sample_rate': f'{self.sample_rate * 1000}ms',
            })
            f.text[0] = text_header
            
            # Write binary header
            f.bin[segyio.BinField.JobID] = file_id
            f.bin[segyio.BinField.LineNumber] = file_id
            f.bin[segyio.BinField.ReelNumber] = 1
            f.bin[segyio.BinField.Interval] = int(self.sample_rate * 1e6)  # Microseconds
            f.bin[segyio.BinField.Samples] = self.num_samples
            f.bin[segyio.BinField.Format] = 5  # IEEE floating point
            f.bin[segyio.BinField.MeasurementSystem] = 1  # Meters
            
            # Write traces
            for trace_idx in range(num_traces):
                trace_data, trace_header = self.generate_trace(
                    trace_idx, class_labels[trace_idx], file_id
                )
                f.trace[trace_idx] = trace_data
                f.header[trace_idx] = trace_header
        
        print(f"Created SEG-Y file: {file_path} ({num_traces} traces)")
        return str(file_path), class_labels
    
    def generate_test_dataset(self, num_files: int = 5) -> Dict[str, List[int]]:
        """
        Generate multiple SEG-Y files for testing.
        
        Args:
            num_files: Number of files to generate
            
        Returns:
            Dictionary mapping file paths to class labels
        """
        dataset_info = {}
        
        for file_id in range(num_files):
            file_path, class_labels = self.create_segy_file(file_id)
            dataset_info[file_path] = class_labels
        
        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'num_files': num_files,
                'num_traces_per_file': self.num_traces_per_file,
                'total_traces': num_files * self.num_traces_per_file,
                'num_samples': self.num_samples,
                'sample_rate_ms': self.sample_rate * 1000,
                'classes': self.classes,
                'files': {path: {'num_traces': len(labels), 'class_distribution': {
                    self.classes[i]: labels.count(i) for i in range(3)
                }} for path, labels in dataset_info.items()}
            }, f, indent=2)
        
        print(f"\nGenerated {num_files} SEG-Y files")
        print(f"Total traces: {num_files * self.num_traces_per_file}")
        print(f"Metadata saved to: {metadata_path}")
        
        return dataset_info


def main():
    """Generate synthetic seismic test dataset."""
    generator = SyntheticSeismicGenerator(output_dir="data/raw")
    dataset_info = generator.generate_test_dataset(num_files=5)
    
    print("\nDataset generation complete!")
    print(f"Files created in: {generator.output_dir.absolute()}")


if __name__ == "__main__":
    main()
