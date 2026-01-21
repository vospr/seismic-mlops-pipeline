"""
Statistical Analysis: Original F3 Dataset vs Testing Dataset

Performs comprehensive statistical analysis:
- Normal distribution tests
- Class distribution analysis
- Balance score calculation
- Comparison between original F3 and sampled testing dataset
"""
import segyio
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera
import warnings
warnings.filterwarnings('ignore')


class F3StatisticalAnalyzer:
    """
    Statistical analysis of F3 dataset.
    """
    
    def __init__(self, f3_file: str = "Temp/f3_dataset.sgy",
                 test_data_path: str = "data/bronze/seismic_data.parquet"):
        self.f3_file = Path(f3_file)
        self.test_data_path = Path(test_data_path)
    
    def analyze_f3_dataset(self, sample_size: int = 10000) -> Dict[str, Any]:
        """
        Analyze original F3 dataset with statistical tests.
        
        Args:
            sample_size: Number of traces to sample for analysis (for performance)
            
        Returns:
            Dictionary with statistical analysis results
        """
        print("=" * 60)
        print("Analyzing Original F3 Dataset")
        print("=" * 60)
        print(f"File: {self.f3_file}")
        print(f"Sample size: {sample_size:,} traces (for performance)")
        
        if not self.f3_file.exists():
            raise FileNotFoundError(f"F3 dataset not found: {self.f3_file}")
        
        results = {
            'file': str(self.f3_file),
            'total_traces': 0,
            'samples_per_trace': 0,
            'sample_rate_ms': 0.0,
            'trace_statistics': {},
            'normal_distribution_tests': {},
            'class_distribution': {},
            'balance_score': 0.0
        }
        
        # Read F3 dataset
        print("\nReading F3 dataset...")
        with segyio.open(str(self.f3_file), ignore_geometry=True) as segy:
            num_traces = len(segy.trace)
            num_samples = len(segy.samples)
            sample_interval_us = segy.bin[segyio.BinField.Interval]
            sample_rate_ms = sample_interval_us / 1000.0
            
            results['total_traces'] = num_traces
            results['samples_per_trace'] = num_samples
            results['sample_rate_ms'] = sample_rate_ms
            
            print(f"  Total traces: {num_traces:,}")
            print(f"  Samples per trace: {num_samples}")
            print(f"  Sample rate: {sample_rate_ms:.2f} ms")
            
            # Sample traces for analysis
            sample_indices = np.random.choice(num_traces, 
                                            size=min(sample_size, num_traces), 
                                            replace=False)
            sample_indices = np.sort(sample_indices)
            
            print(f"\nSampling {len(sample_indices):,} traces for analysis...")
            
            # Collect trace statistics
            trace_stats = []
            trace_means = []
            trace_stds = []
            trace_data_list = []
            
            for i, trace_idx in enumerate(sample_indices):
                if i % 1000 == 0:
                    print(f"  Progress: {i:,}/{len(sample_indices):,}")
                
                trace_data = segy.trace[trace_idx]
                trace_array = np.array(trace_data, dtype=np.float64)
                
                trace_stats.append({
                    'mean': float(np.mean(trace_array)),
                    'std': float(np.std(trace_array)),
                    'min': float(np.min(trace_array)),
                    'max': float(np.max(trace_array)),
                    'variance': float(np.var(trace_array)),
                })
                
                trace_means.append(float(np.mean(trace_array)))
                trace_stds.append(float(np.std(trace_array)))
                trace_data_list.append(trace_array)
            
            # Aggregate statistics
            results['trace_statistics'] = {
                'mean_of_means': float(np.mean(trace_means)),
                'std_of_means': float(np.std(trace_means)),
                'mean_of_stds': float(np.mean(trace_stds)),
                'std_of_stds': float(np.std(trace_stds)),
                'overall_min': float(min([s['min'] for s in trace_stats])),
                'overall_max': float(max([s['max'] for s in trace_stats])),
                'sample_size': len(sample_indices)
            }
            
            print(f"\nTrace Statistics:")
            print(f"  Mean of means: {results['trace_statistics']['mean_of_means']:.4f}")
            print(f"  Mean of stds: {results['trace_statistics']['mean_of_stds']:.4f}")
            print(f"  Overall range: [{results['trace_statistics']['overall_min']:.2f}, {results['trace_statistics']['overall_max']:.2f}]")
            
            # Normal distribution tests on trace means
            print(f"\nTesting normal distribution of trace means...")
            trace_means_array = np.array(trace_means)
            
            # Shapiro-Wilk test (for smaller samples)
            if len(trace_means) <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(trace_means_array)
                    results['normal_distribution_tests']['shapiro_wilk'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
                    print(f"  Shapiro-Wilk: statistic={shapiro_stat:.4f}, p={shapiro_p:.4f}, normal={shapiro_p > 0.05}")
                except Exception as e:
                    print(f"  Shapiro-Wilk test failed: {e}")
            
            # D'Agostino-Pearson test
            try:
                dagostino_stat, dagostino_p = normaltest(trace_means_array)
                results['normal_distribution_tests']['dagostino_pearson'] = {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'is_normal': dagostino_p > 0.05
                }
                print(f"  D'Agostino-Pearson: statistic={dagostino_stat:.4f}, p={dagostino_p:.4f}, normal={dagostino_p > 0.05}")
            except Exception as e:
                print(f"  D'Agostino-Pearson test failed: {e}")
            
            # Jarque-Bera test
            try:
                jb_stat, jb_p = jarque_bera(trace_means_array)
                results['normal_distribution_tests']['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > 0.05
                }
                print(f"  Jarque-Bera: statistic={jb_stat:.4f}, p={jb_p:.4f}, normal={jb_p > 0.05}")
            except Exception as e:
                print(f"  Jarque-Bera test failed: {e}")
            
            # Class distribution (generate deterministically like Stage 0/1)
            print(f"\nAnalyzing class distribution...")
            np.random.seed(42)  # Fixed seed for reproducibility
            class_labels = []
            for trace_idx in sample_indices:
                np.random.seed(trace_idx)
                rand = np.random.random()
                if rand < 0.5:
                    class_labels.append(0)  # Normal
                elif rand < 0.8:
                    class_labels.append(1)  # Anomaly
                else:
                    class_labels.append(2)  # Boundary
            
            class_counts = pd.Series(class_labels).value_counts().sort_index()
            class_percentages = (class_counts / len(class_labels) * 100).round(2)
            
            results['class_distribution'] = {
                'counts': class_counts.to_dict(),
                'percentages': class_percentages.to_dict(),
                'total': len(class_labels)
            }
            
            # Calculate balance score (entropy-based)
            class_probs = class_counts / len(class_labels)
            entropy = stats.entropy(class_probs, base=2)
            max_entropy = np.log2(len(class_probs))
            balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
            
            results['balance_score'] = float(balance_score)
            
            print(f"  Class distribution:")
            for class_id, count in class_counts.items():
                pct = class_percentages[class_id]
                class_name = ['Normal', 'Anomaly', 'Boundary'][int(class_id)]
                print(f"    {class_name} (class {class_id}): {count} ({pct}%)")
            print(f"  Balance score: {balance_score:.4f} (1.0 = perfectly balanced)")
        
        return results
    
    def analyze_test_dataset(self) -> Dict[str, Any]:
        """
        Analyze testing dataset from bronze layer.
        
        Returns:
            Dictionary with statistical analysis results
        """
        print("\n" + "=" * 60)
        print("Analyzing Testing Dataset")
        print("=" * 60)
        print(f"File: {self.test_data_path}")
        
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
        
        # Load test data
        df = pd.read_parquet(self.test_data_path)
        print(f"  Total traces: {len(df):,}")
        
        results = {
            'file': str(self.test_data_path),
            'total_traces': len(df),
            'samples_per_trace': int(df['num_samples'].iloc[0]) if len(df) > 0 else 0,
            'sample_rate_ms': float(df['sample_rate'].iloc[0] * 1000) if len(df) > 0 else 0.0,
            'trace_statistics': {},
            'normal_distribution_tests': {},
            'class_distribution': {},
            'balance_score': 0.0
        }
        
        # Extract trace statistics
        print("\nCalculating trace statistics...")
        trace_means = []
        trace_stds = []
        
        for idx, row in df.iterrows():
            trace_data = np.array(row['trace_data'], dtype=np.float64)
            trace_means.append(float(np.mean(trace_data)))
            trace_stds.append(float(np.std(trace_data)))
        
        results['trace_statistics'] = {
            'mean_of_means': float(np.mean(trace_means)),
            'std_of_means': float(np.std(trace_means)),
            'mean_of_stds': float(np.mean(trace_stds)),
            'std_of_stds': float(np.std(trace_stds)),
            'overall_min': float(min([np.min(np.array(row['trace_data'])) for _, row in df.iterrows()])),
            'overall_max': float(max([np.max(np.array(row['trace_data'])) for _, row in df.iterrows()])),
            'sample_size': len(df)
        }
        
        print(f"  Mean of means: {results['trace_statistics']['mean_of_means']:.4f}")
        print(f"  Mean of stds: {results['trace_statistics']['mean_of_stds']:.4f}")
        
        # Normal distribution tests
        print(f"\nTesting normal distribution of trace means...")
        trace_means_array = np.array(trace_means)
        
        # Shapiro-Wilk test
        if len(trace_means) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(trace_means_array)
                results['normal_distribution_tests']['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
                print(f"  Shapiro-Wilk: statistic={shapiro_stat:.4f}, p={shapiro_p:.4f}, normal={shapiro_p > 0.05}")
            except Exception as e:
                print(f"  Shapiro-Wilk test failed: {e}")
        
        # D'Agostino-Pearson test
        try:
            dagostino_stat, dagostino_p = normaltest(trace_means_array)
            results['normal_distribution_tests']['dagostino_pearson'] = {
                'statistic': float(dagostino_stat),
                'p_value': float(dagostino_p),
                'is_normal': dagostino_p > 0.05
            }
            print(f"  D'Agostino-Pearson: statistic={dagostino_stat:.4f}, p={dagostino_p:.4f}, normal={dagostino_p > 0.05}")
        except Exception as e:
            print(f"  D'Agostino-Pearson test failed: {e}")
        
        # Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera(trace_means_array)
            results['normal_distribution_tests']['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > 0.05
            }
            print(f"  Jarque-Bera: statistic={jb_stat:.4f}, p={jb_p:.4f}, normal={jb_p > 0.05}")
        except Exception as e:
            print(f"  Jarque-Bera test failed: {e}")
        
        # Class distribution
        print(f"\nAnalyzing class distribution...")
        if 'class_label' in df.columns:
            class_counts = df['class_label'].value_counts().sort_index()
            class_percentages = (class_counts / len(df) * 100).round(2)
            
            results['class_distribution'] = {
                'counts': class_counts.to_dict(),
                'percentages': class_percentages.to_dict(),
                'total': len(df)
            }
            
            # Calculate balance score
            class_probs = class_counts / len(df)
            entropy = stats.entropy(class_probs, base=2)
            max_entropy = np.log2(len(class_probs))
            balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
            
            results['balance_score'] = float(balance_score)
            
            print(f"  Class distribution:")
            for class_id, count in class_counts.items():
                pct = class_percentages[class_id]
                class_name = ['Normal', 'Anomaly', 'Boundary'][int(class_id)]
                print(f"    {class_name} (class {class_id}): {count} ({pct}%)")
            print(f"  Balance score: {balance_score:.4f} (1.0 = perfectly balanced)")
        else:
            print("  Warning: class_label column not found")
            results['class_distribution'] = {'counts': {}, 'percentages': {}, 'total': len(df)}
            results['balance_score'] = 0.0
        
        return results
    
    def compare_datasets(self, f3_results: Dict, test_results: Dict) -> Dict[str, Any]:
        """
        Compare F3 and testing dataset statistics.
        
        Args:
            f3_results: F3 dataset analysis results
            test_results: Testing dataset analysis results
            
        Returns:
            Comparison dictionary
        """
        print("\n" + "=" * 60)
        print("Comparison: F3 Dataset vs Testing Dataset")
        print("=" * 60)
        
        comparison = {
            'trace_statistics': {},
            'normal_distribution': {},
            'class_distribution': {},
            'balance_score': {}
        }
        
        # Compare trace statistics
        print("\nTrace Statistics Comparison:")
        f3_stats = f3_results['trace_statistics']
        test_stats = test_results['trace_statistics']
        
        comparison['trace_statistics'] = {
            'mean_of_means': {
                'f3': f3_stats['mean_of_means'],
                'test': test_stats['mean_of_means'],
                'difference': test_stats['mean_of_means'] - f3_stats['mean_of_means'],
                'relative_diff_pct': ((test_stats['mean_of_means'] - f3_stats['mean_of_means']) / abs(f3_stats['mean_of_means']) * 100) if f3_stats['mean_of_means'] != 0 else 0
            },
            'mean_of_stds': {
                'f3': f3_stats['mean_of_stds'],
                'test': test_stats['mean_of_stds'],
                'difference': test_stats['mean_of_stds'] - f3_stats['mean_of_stds'],
                'relative_diff_pct': ((test_stats['mean_of_stds'] - f3_stats['mean_of_stds']) / abs(f3_stats['mean_of_stds']) * 100) if f3_stats['mean_of_stds'] != 0 else 0
            }
        }
        
        print(f"  Mean of means:")
        print(f"    F3: {f3_stats['mean_of_means']:.4f}")
        print(f"    Test: {test_stats['mean_of_means']:.4f}")
        print(f"    Difference: {comparison['trace_statistics']['mean_of_means']['difference']:.4f} ({comparison['trace_statistics']['mean_of_means']['relative_diff_pct']:.2f}%)")
        
        print(f"  Mean of stds:")
        print(f"    F3: {f3_stats['mean_of_stds']:.4f}")
        print(f"    Test: {test_stats['mean_of_stds']:.4f}")
        print(f"    Difference: {comparison['trace_statistics']['mean_of_stds']['difference']:.4f} ({comparison['trace_statistics']['mean_of_stds']['relative_diff_pct']:.2f}%)")
        
        # Compare normal distribution tests
        print("\nNormal Distribution Tests Comparison:")
        f3_norm = f3_results['normal_distribution_tests']
        test_norm = test_results['normal_distribution_tests']
        
        comparison['normal_distribution'] = {}
        for test_name in ['dagostino_pearson', 'jarque_bera', 'shapiro_wilk']:
            if test_name in f3_norm and test_name in test_norm:
                f3_test = f3_norm[test_name]
                test_test = test_norm[test_name]
                
                comparison['normal_distribution'][test_name] = {
                    'f3': {
                        'is_normal': f3_test['is_normal'],
                        'p_value': f3_test['p_value']
                    },
                    'test': {
                        'is_normal': test_test['is_normal'],
                        'p_value': test_test['p_value']
                    },
                    'agreement': f3_test['is_normal'] == test_test['is_normal']
                }
                
                print(f"  {test_name.replace('_', ' ').title()}:")
                print(f"    F3: normal={f3_test['is_normal']}, p={f3_test['p_value']:.4f}")
                print(f"    Test: normal={test_test['is_normal']}, p={test_test['p_value']:.4f}")
                print(f"    Agreement: {f3_test['is_normal'] == test_test['is_normal']}")
        
        # Compare class distribution
        print("\nClass Distribution Comparison:")
        f3_classes = f3_results['class_distribution']
        test_classes = test_results['class_distribution']
        
        comparison['class_distribution'] = {}
        for class_id in [0, 1, 2]:
            class_name = ['Normal', 'Anomaly', 'Boundary'][class_id]
            f3_pct = f3_classes['percentages'].get(class_id, 0)
            test_pct = test_classes['percentages'].get(class_id, 0)
            
            comparison['class_distribution'][class_name] = {
                'f3_percentage': f3_pct,
                'test_percentage': test_pct,
                'difference': test_pct - f3_pct
            }
            
            print(f"  {class_name}:")
            print(f"    F3: {f3_pct}%")
            print(f"    Test: {test_pct}%")
            print(f"    Difference: {test_pct - f3_pct:.2f}%")
        
        # Compare balance scores
        print("\nBalance Score Comparison:")
        f3_balance = f3_results['balance_score']
        test_balance = test_results['balance_score']
        
        comparison['balance_score'] = {
            'f3': f3_balance,
            'test': test_balance,
            'difference': test_balance - f3_balance
        }
        
        print(f"  F3: {f3_balance:.4f}")
        print(f"  Test: {test_balance:.4f}")
        print(f"  Difference: {test_balance - f3_balance:.4f}")
        
        return comparison
    
    def generate_report(self, f3_results: Dict, test_results: Dict, 
                       comparison: Dict, output_path: Path):
        """
        Generate comprehensive comparison report.
        
        Args:
            f3_results: F3 dataset analysis
            test_results: Testing dataset analysis
            comparison: Comparison results
            output_path: Output file path
        """
        report = f"""Statistical Analysis Report: F3 Dataset vs Testing Dataset
{'=' * 80}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Original F3 Dataset Analysis

### Dataset Information
- File: {f3_results['file']}
- Total Traces: {f3_results['total_traces']:,}
- Samples per Trace: {f3_results['samples_per_trace']}
- Sample Rate: {f3_results['sample_rate_ms']:.2f} ms
- Analysis Sample Size: {f3_results['trace_statistics']['sample_size']:,} traces

### Trace Statistics
- Mean of Means: {f3_results['trace_statistics']['mean_of_means']:.4f}
- Std of Means: {f3_results['trace_statistics']['std_of_means']:.4f}
- Mean of Stds: {f3_results['trace_statistics']['mean_of_stds']:.4f}
- Std of Stds: {f3_results['trace_statistics']['std_of_stds']:.4f}
- Overall Range: [{f3_results['trace_statistics']['overall_min']:.2f}, {f3_results['trace_statistics']['overall_max']:.2f}]

### Normal Distribution Tests

"""
        
        for test_name, test_result in f3_results['normal_distribution_tests'].items():
            test_display = test_name.replace('_', ' ').title()
            report += f"**{test_display}:**\n"
            report += f"- Statistic: {test_result['statistic']:.4f}\n"
            report += f"- P-value: {test_result['p_value']:.4f}\n"
            report += f"- Is Normal (p > 0.05): {test_result['is_normal']}\n\n"
        
        report += f"""### Class Distribution
- Normal (class 0): {f3_results['class_distribution']['counts'].get(0, 0)} ({f3_results['class_distribution']['percentages'].get(0, 0)}%)
- Anomaly (class 1): {f3_results['class_distribution']['counts'].get(1, 0)} ({f3_results['class_distribution']['percentages'].get(1, 0)}%)
- Boundary (class 2): {f3_results['class_distribution']['counts'].get(2, 0)} ({f3_results['class_distribution']['percentages'].get(2, 0)}%)

### Balance Score
- Balance Score: {f3_results['balance_score']:.4f} (1.0 = perfectly balanced)

---

## 2. Testing Dataset Analysis

### Dataset Information
- File: {test_results['file']}
- Total Traces: {test_results['total_traces']:,}
- Samples per Trace: {test_results['samples_per_trace']}
- Sample Rate: {test_results['sample_rate_ms']:.2f} ms

### Trace Statistics
- Mean of Means: {test_results['trace_statistics']['mean_of_means']:.4f}
- Std of Means: {test_results['trace_statistics']['std_of_means']:.4f}
- Mean of Stds: {test_results['trace_statistics']['mean_of_stds']:.4f}
- Std of Stds: {test_results['trace_statistics']['std_of_stds']:.4f}
- Overall Range: [{test_results['trace_statistics']['overall_min']:.2f}, {test_results['trace_statistics']['overall_max']:.2f}]

### Normal Distribution Tests

"""
        
        for test_name, test_result in test_results['normal_distribution_tests'].items():
            test_display = test_name.replace('_', ' ').title()
            report += f"**{test_display}:**\n"
            report += f"- Statistic: {test_result['statistic']:.4f}\n"
            report += f"- P-value: {test_result['p_value']:.4f}\n"
            report += f"- Is Normal (p > 0.05): {test_result['is_normal']}\n\n"
        
        report += f"""### Class Distribution
- Normal (class 0): {test_results['class_distribution']['counts'].get(0, 0)} ({test_results['class_distribution']['percentages'].get(0, 0)}%)
- Anomaly (class 1): {test_results['class_distribution']['counts'].get(1, 0)} ({test_results['class_distribution']['percentages'].get(1, 0)}%)
- Boundary (class 2): {test_results['class_distribution']['counts'].get(2, 0)} ({test_results['class_distribution']['percentages'].get(2, 0)}%)

### Balance Score
- Balance Score: {test_results['balance_score']:.4f} (1.0 = perfectly balanced)

---

## 3. Comparison Analysis

### Trace Statistics Comparison

**Mean of Means:**
- F3: {comparison['trace_statistics']['mean_of_means']['f3']:.4f}
- Test: {comparison['trace_statistics']['mean_of_means']['test']:.4f}
- Difference: {comparison['trace_statistics']['mean_of_means']['difference']:.4f} ({comparison['trace_statistics']['mean_of_means']['relative_diff_pct']:.2f}%)

**Mean of Stds:**
- F3: {comparison['trace_statistics']['mean_of_stds']['f3']:.4f}
- Test: {comparison['trace_statistics']['mean_of_stds']['test']:.4f}
- Difference: {comparison['trace_statistics']['mean_of_stds']['difference']:.4f} ({comparison['trace_statistics']['mean_of_stds']['relative_diff_pct']:.2f}%)

### Normal Distribution Tests Comparison

"""
        
        for test_name, test_comp in comparison['normal_distribution'].items():
            test_display = test_name.replace('_', ' ').title()
            report += f"**{test_display}:**\n"
            report += f"- F3: normal={test_comp['f3']['is_normal']}, p={test_comp['f3']['p_value']:.4f}\n"
            report += f"- Test: normal={test_comp['test']['is_normal']}, p={test_comp['test']['p_value']:.4f}\n"
            report += f"- Agreement: {test_comp['agreement']}\n\n"
        
        report += f"""### Class Distribution Comparison

"""
        
        for class_name, class_comp in comparison['class_distribution'].items():
            report += f"**{class_name}:**\n"
            report += f"- F3: {class_comp['f3_percentage']}%\n"
            report += f"- Test: {class_comp['test_percentage']}%\n"
            report += f"- Difference: {class_comp['difference']:.2f}%\n\n"
        
        report += f"""### Balance Score Comparison
- F3: {comparison['balance_score']['f3']:.4f}
- Test: {comparison['balance_score']['test']:.4f}
- Difference: {comparison['balance_score']['difference']:.4f}

---

## 4. Key Findings

### Statistical Similarity
"""
        
        # Determine similarity
        mean_diff_pct = abs(comparison['trace_statistics']['mean_of_means']['relative_diff_pct'])
        std_diff_pct = abs(comparison['trace_statistics']['mean_of_stds']['relative_diff_pct'])
        balance_diff = abs(comparison['balance_score']['difference'])
        
        if mean_diff_pct < 5 and std_diff_pct < 5 and balance_diff < 0.1:
            report += "- **High Similarity**: Testing dataset closely matches F3 dataset characteristics\n"
        elif mean_diff_pct < 10 and std_diff_pct < 10 and balance_diff < 0.2:
            report += "- **Moderate Similarity**: Testing dataset generally matches F3 dataset with minor differences\n"
        else:
            report += "- **Low Similarity**: Testing dataset shows significant differences from F3 dataset\n"
        
        report += f"""
### Normal Distribution
"""
        
        # Check normal distribution agreement
        norm_agreements = [comp['agreement'] for comp in comparison['normal_distribution'].values()]
        if all(norm_agreements):
            report += "- All normal distribution tests show **agreement** between F3 and test datasets\n"
        else:
            report += "- Normal distribution tests show **disagreement** - datasets may have different distributions\n"
        
        report += f"""
### Class Balance
"""
        
        if balance_diff < 0.05:
            report += "- Class distributions are **very similar** between datasets\n"
        elif balance_diff < 0.1:
            report += "- Class distributions are **similar** with minor differences\n"
        else:
            report += "- Class distributions show **noticeable differences** - may affect model training\n"
        
        report += f"""

---

## 5. Recommendations

1. **Data Representativeness**: {"Testing dataset appears representative of F3 dataset" if mean_diff_pct < 10 else "Consider increasing sample size or adjusting sampling strategy"}
2. **Normal Distribution**: {"Both datasets show similar distribution characteristics" if all(norm_agreements) else "Distribution differences may require different preprocessing"}
3. **Class Balance**: {"Class distributions are well-balanced" if test_results['balance_score'] > 0.8 else "Consider class balancing techniques if needed"}
4. **Model Training**: {"Testing dataset suitable for model training" if mean_diff_pct < 10 and balance_diff < 0.1 else "Review sampling strategy to better match F3 characteristics"}

---

**End of Report**
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n[OK] Report saved to: {output_path}")
        return output_path


def main():
    """Execute statistical analysis."""
    analyzer = F3StatisticalAnalyzer(
        f3_file="Temp/f3_dataset.sgy",
        test_data_path="data/bronze/seismic_data.parquet"
    )
    
    # Analyze F3 dataset
    f3_results = analyzer.analyze_f3_dataset(sample_size=10000)
    
    # Analyze test dataset
    test_results = analyzer.analyze_test_dataset()
    
    # Compare
    comparison = analyzer.compare_datasets(f3_results, test_results)
    
    # Generate report
    output_path = Path("data/bronze/f3_vs_test_statistical_comparison.txt")
    analyzer.generate_report(f3_results, test_results, comparison, output_path)
    
    # Save JSON data
    json_path = Path("data/bronze/f3_vs_test_statistical_comparison.json")
    with open(json_path, 'w') as f:
        json.dump({
            'f3_results': f3_results,
            'test_results': test_results,
            'comparison': comparison,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2, default=str)
    
    print(f"[OK] JSON data saved to: {json_path}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
