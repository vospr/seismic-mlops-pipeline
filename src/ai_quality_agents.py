"""
AI Quality Agents for Data Quality Management

This module implements AI agents for comprehensive data quality analysis:
- Statistical Analysis Agent
- Domain-Specific Validation Agent
- Data Drift Detection Agent
- Data Quality Evaluation Agent (orchestrator)

All agents use LLM (Ollama) for intelligent analysis and generate reports.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from scipy import stats
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def call_ollama_with_timeout(prompt: str, model: str = "llama3.1:8b", 
                             timeout: int = 120, temperature: float = 0.3) -> str:
    """
    Call Ollama with timeout protection (Windows-compatible).
    Each call completes before returning to ensure sequential execution.
    
    Args:
        prompt: Prompt to send to LLM
        model: Model name
        timeout: Timeout in seconds (default 300 = 5 minutes)
        temperature: Temperature setting
        
    Returns:
        LLM response text
        
    Raises:
        TimeoutError: If call exceeds timeout
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama not available")
    
    print(f"    [LLM] Sending request to {model}...")
    print(f"    [LLM] Timeout set to {timeout}s (this may take 30-120 seconds)")
    import time
    start_time = time.time()
    
    try:
        # Direct call - Ollama should handle its own timeouts
        # We'll monitor elapsed time and provide progress updates
        import threading
        
        result_container = {'response': None, 'error': None, 'completed': False}
        progress_stop = threading.Event()
        
        def make_request():
            try:
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={"temperature": temperature}
                )
                result_container['response'] = response.get('response', '')
                result_container['completed'] = True
            except Exception as e:
                result_container['error'] = e
                result_container['completed'] = True
        
        def show_progress():
            """Show progress every 30 seconds."""
            elapsed = 0
            while not progress_stop.is_set() and elapsed < timeout:
                time.sleep(30)
                elapsed = time.time() - start_time
                if not result_container['completed']:
                    print(f"    [LLM] Still processing... ({elapsed:.0f}s elapsed)")
        
        # Start progress indicator
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        # Make the actual request
        request_thread = threading.Thread(target=make_request, daemon=False)
        request_thread.start()
        request_thread.join(timeout=timeout)
        
        progress_stop.set()
        elapsed = time.time() - start_time
        
        # Check if thread is still alive (timed out)
        if request_thread.is_alive():
            print(f"    [LLM] ERROR: Request exceeded timeout of {timeout}s")
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        
        if not result_container['completed']:
            print(f"    [LLM] ERROR: Request did not complete")
            raise RuntimeError("LLM call did not complete")
        
        if result_container['error']:
            print(f"    [LLM] ERROR: {result_container['error']}")
            raise result_container['error']
        
        if result_container['response']:
            result = result_container['response']
            print(f"    [LLM] SUCCESS: Response received in {elapsed:.1f}s ({len(result)} characters)")
            return result
        else:
            raise RuntimeError("No response received from LLM")
            
    except TimeoutError:
        print(f"    [LLM] Request timed out - continuing without LLM analysis for this step")
        raise
    except Exception as e:
        print(f"    [LLM] ERROR: {e}")
        raise


class LLMStatisticalAnalyzer:
    """
    Statistical Analysis Agent using LLM.
    Performs deep statistical analysis of ingested data.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and OLLAMA_AVAILABLE
    
    def analyze(self, df: pd.DataFrame, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis using LLM with actual data statistics.
        
        Args:
            df: DataFrame with ingested data
            validation_results: Validation results from Stage 1
            
        Returns:
            Dictionary with statistical analysis results
        """
        # Compute statistical properties
        stats_data = self._compute_statistics(df)
        
        if not self.use_llm:
            return {
                'statistics': stats_data,
                'llm_analysis': None,
                'status': 'LLM not available'
            }
        
        try:
            prompt = f"""
            Analyze these statistical properties of seismic data:
            1. Are amplitude ranges reasonable for seismic data?
            2. Are coordinate ranges consistent?
            3. Is class distribution balanced?
            4. Identify statistical anomalies
            5. Check for outliers in trace amplitudes
            6. Validate correlation patterns
            
            Statistics: {json.dumps(stats_data, indent=2, default=str)}
            Validation Results: {json.dumps(validation_results, indent=2, default=str)}
            
            Provide detailed analysis with specific concerns and recommendations.
            """
            
            llm_response = call_ollama_with_timeout(
                prompt=prompt,
                model="llama3.1:8b",
                timeout=180,
                temperature=0.3
            )
            
            return {
                'statistics': stats_data,
                'llm_analysis': llm_response,
                'status': 'success'
            }
        except Exception as e:
            return {
                'statistics': stats_data,
                'llm_analysis': None,
                'status': f'LLM analysis failed: {e}'
            }
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical properties of the data."""
        stats_dict = {}
        
        # Trace data statistics (sample first 100 traces for performance)
        trace_stats_list = []
        sample_size = min(100, len(df))
        for idx in range(sample_size):
            trace_data = df.iloc[idx]['trace_data']
            if isinstance(trace_data, list):
                trace_array = np.array(trace_data)
                trace_stats_list.append({
                    'mean': float(np.mean(trace_array)),
                    'std': float(np.std(trace_array)),
                    'min': float(np.min(trace_array)),
                    'max': float(np.max(trace_array)),
                    'variance': float(np.var(trace_array)),
                })
        
        if trace_stats_list:
            stats_dict['trace_data_stats'] = {
                'mean_of_means': np.mean([s['mean'] for s in trace_stats_list]),
                'mean_of_stds': np.mean([s['std'] for s in trace_stats_list]),
                'overall_min': min([s['min'] for s in trace_stats_list]),
                'overall_max': max([s['max'] for s in trace_stats_list]),
                'dead_traces': sum(1 for s in trace_stats_list if s['variance'] == 0),
                'sample_size': sample_size,
            }
        
        # Coordinate ranges
        if 'cdp_x' in df.columns and 'cdp_y' in df.columns:
            stats_dict['coordinate_ranges'] = {
                'cdp_x': {
                    'min': float(df['cdp_x'].min()),
                    'max': float(df['cdp_x'].max()),
                    'mean': float(df['cdp_x'].mean()),
                    'std': float(df['cdp_x'].std()),
                },
                'cdp_y': {
                    'min': float(df['cdp_y'].min()),
                    'max': float(df['cdp_y'].max()),
                    'mean': float(df['cdp_y'].mean()),
                    'std': float(df['cdp_y'].std()),
                }
            }
        
        # Class distribution
        if 'class_label' in df.columns:
            class_counts = df['class_label'].value_counts()
            stats_dict['class_distribution'] = {
                'counts': class_counts.to_dict(),
                'percentages': (class_counts / len(df) * 100).to_dict(),
                'balance_score': float(1 - stats.entropy(class_counts / len(df), base=2) / np.log2(len(class_counts)))
            }
        
        # Sample rate and num_samples
        if 'sample_rate' in df.columns:
            stats_dict['sample_rate'] = {
                'value_ms': float(df['sample_rate'].iloc[0] * 1000),
                'consistency': bool(df['sample_rate'].nunique() == 1),
            }
        
        if 'num_samples' in df.columns:
            stats_dict['num_samples'] = {
                'value': int(df['num_samples'].iloc[0]),
                'consistency': bool(df['num_samples'].nunique() == 1),
            }
        
        return stats_dict


class LLMDomainValidator:
    """
    Domain-Specific Validation Agent using LLM.
    Validates seismic data using domain expertise.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and OLLAMA_AVAILABLE
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Domain-specific seismic data validation using LLM.
        
        Args:
            df: DataFrame with ingested data
            
        Returns:
            Dictionary with domain validation results
        """
        # Perform domain-specific checks
        domain_checks = self._perform_domain_checks(df)
        
        if not self.use_llm:
            return {
                'domain_checks': domain_checks,
                'llm_analysis': None,
                'status': 'LLM not available'
            }
        
        try:
            prompt = f"""
            As a seismic data expert, validate this data:
            1. Are trace amplitudes within expected seismic ranges?
            2. Is sample rate within industry standards (0.5-4ms)?
            3. Are spatial coordinates consistent and reasonable?
            4. Are there dead traces (zero variance)?
            5. Check for geophysical consistency (inline/crossline relationships)
            6. Validate data completeness
            
            Domain Checks: {json.dumps(domain_checks, indent=2, default=str)}
            
            Provide expert validation with specific issues and recommendations.
            """
            
            llm_response = call_ollama_with_timeout(
                prompt=prompt,
                model="llama3.1:8b",
                timeout=180,
                temperature=0.3
            )
            
            return {
                'domain_checks': domain_checks,
                'llm_analysis': llm_response,
                'status': 'success'
            }
        except Exception as e:
            return {
                'domain_checks': domain_checks,
                'llm_analysis': None,
                'status': f'LLM analysis failed: {e}'
            }
    
    def _perform_domain_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform domain-specific validation checks."""
        checks = {}
        
        # Trace amplitude ranges
        trace_amplitudes = []
        dead_traces = 0
        sample_size = min(100, len(df))
        
        for idx in range(sample_size):
            trace_data = df.iloc[idx]['trace_data']
            if isinstance(trace_data, list):
                trace_array = np.array(trace_data)
                trace_amplitudes.extend(trace_array.tolist())
                if np.std(trace_array) == 0:
                    dead_traces += 1
        
        if trace_amplitudes:
            checks['trace_amplitude_ranges'] = {
                'min_amplitude': float(np.min(trace_amplitudes)),
                'max_amplitude': float(np.max(trace_amplitudes)),
                'mean_amplitude': float(np.mean(trace_amplitudes)),
                'std_amplitude': float(np.std(trace_amplitudes)),
                'dead_traces': dead_traces,
                'dead_traces_percentage': float(dead_traces / sample_size * 100),
                'sample_size': sample_size,
            }
        
        # Sample rate validation
        if 'sample_rate' in df.columns:
            sample_rate_ms = df['sample_rate'].iloc[0] * 1000
            checks['sample_rate_validation'] = {
                'sample_rate_ms': float(sample_rate_ms),
                'within_standards': bool(0.5 <= sample_rate_ms <= 4.0),
                'standard_range': '0.5-4.0 ms',
            }
        
        # Spatial consistency
        if 'inline' in df.columns and 'crossline' in df.columns:
            checks['spatial_consistency'] = {
                'inline_range': [int(df['inline'].min()), int(df['inline'].max())],
                'crossline_range': [int(df['crossline'].min()), int(df['crossline'].max())],
                'unique_inlines': int(df['inline'].nunique()),
                'unique_crosslines': int(df['crossline'].nunique()),
            }
        
        # Coordinate consistency
        if 'cdp_x' in df.columns and 'cdp_y' in df.columns:
            checks['coordinate_consistency'] = {
                'cdp_x_range': [float(df['cdp_x'].min()), float(df['cdp_x'].max())],
                'cdp_y_range': [float(df['cdp_y'].min()), float(df['cdp_y'].max())],
                'unique_cdp_x': int(df['cdp_x'].nunique()),
                'unique_cdp_y': int(df['cdp_y'].nunique()),
            }
        
        # Data completeness
        checks['data_completeness'] = {
            'total_traces': len(df),
            'total_files': int(df['file_id'].nunique()) if 'file_id' in df.columns else 0,
            'missing_trace_data': int(df['trace_data'].isna().sum()) if 'trace_data' in df.columns else 0,
            'missing_coordinates': int(df[['cdp_x', 'cdp_y']].isna().any(axis=1).sum()) if all(c in df.columns for c in ['cdp_x', 'cdp_y']) else 0,
        }
        
        return checks


class LLMDriftDetector:
    """
    Data Drift Detection Agent using LLM.
    Compares current data with historical patterns.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and OLLAMA_AVAILABLE
    
    def detect(self, df: pd.DataFrame, historical_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze data drift using LLM with historical comparison.
        
        Args:
            df: Current DataFrame
            historical_stats: Historical statistics (if available)
            
        Returns:
            Dictionary with drift analysis results
        """
        current_stats = self._compute_statistics(df)
        
        if historical_stats is None:
            return {
                'current_stats': current_stats,
                'drift_indicators': None,
                'llm_analysis': 'No historical data available for comparison',
                'status': 'no_history'
            }
        
        # Compute drift indicators
        drift_indicators = self._compute_drift_indicators(current_stats, historical_stats)
        
        if not self.use_llm:
            return {
                'current_stats': current_stats,
                'historical_stats': historical_stats,
                'drift_indicators': drift_indicators,
                'llm_analysis': None,
                'status': 'LLM not available'
            }
        
        try:
            prompt = f"""
            Analyze data drift between current and historical data:
            1. Has class distribution changed significantly?
            2. Are amplitude characteristics shifting?
            3. Is data volume consistent?
            4. Are there schema changes?
            5. Recommend actions if drift detected
            
            Current Statistics: {json.dumps(current_stats, indent=2, default=str)}
            Historical Statistics: {json.dumps(historical_stats, indent=2, default=str)}
            Drift Indicators: {json.dumps(drift_indicators, indent=2, default=str)}
            
            Provide detailed drift analysis with severity assessment.
            """
            
            llm_response = call_ollama_with_timeout(
                prompt=prompt,
                model="llama3.1:8b",
                timeout=180,
                temperature=0.3
            )
            
            return {
                'current_stats': current_stats,
                'historical_stats': historical_stats,
                'drift_indicators': drift_indicators,
                'llm_analysis': llm_response,
                'status': 'success'
            }
        except Exception as e:
            return {
                'current_stats': current_stats,
                'historical_stats': historical_stats,
                'drift_indicators': drift_indicators,
                'llm_analysis': None,
                'status': f'LLM analysis failed: {e}'
            }
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics for drift comparison."""
        stats = {
            'total_traces': len(df),
            'total_files': int(df['file_id'].nunique()) if 'file_id' in df.columns else 0,
        }
        
        if 'class_label' in df.columns:
            class_counts = df['class_label'].value_counts()
            stats['class_distribution'] = (class_counts / len(df)).to_dict()
        
        # Sample trace statistics
        if 'trace_data' in df.columns:
            sample_traces = df['trace_data'].head(50)
            amplitudes = []
            for trace in sample_traces:
                if isinstance(trace, list):
                    amplitudes.extend(trace)
            
            if amplitudes:
                stats['mean_amplitude'] = float(np.mean(amplitudes))
                stats['std_amplitude'] = float(np.std(amplitudes))
        
        return stats
    
    def _compute_drift_indicators(self, current: Dict, historical: Dict) -> Dict[str, Any]:
        """Compute drift indicators between current and historical stats."""
        indicators = {}
        
        # Class distribution change
        if 'class_distribution' in current and 'class_distribution' in historical:
            current_dist = current['class_distribution']
            historical_dist = historical.get('class_distribution', {})
            
            max_change = 0
            for class_label in current_dist:
                current_pct = current_dist.get(class_label, 0)
                historical_pct = historical_dist.get(class_label, 0)
                change = abs(current_pct - historical_pct)
                max_change = max(max_change, change)
            
            indicators['class_distribution_change'] = {
                'max_change': float(max_change),
                'significant': max_change > 0.1,  # More than 10% change
            }
        
        # Amplitude shift
        if 'mean_amplitude' in current and 'mean_amplitude' in historical:
            shift = abs(current['mean_amplitude'] - historical.get('mean_amplitude', 0))
            indicators['amplitude_shift'] = {
                'absolute_shift': float(shift),
                'significant': shift > 100,  # Threshold for significance
            }
        
        # Volume change
        if 'total_traces' in current and 'total_traces' in historical:
            volume_change = abs(current['total_traces'] - historical.get('total_traces', 0))
            volume_change_pct = (volume_change / max(historical.get('total_traces', 1), 1)) * 100
            indicators['volume_change'] = {
                'absolute_change': int(volume_change),
                'percentage_change': float(volume_change_pct),
                'significant': volume_change_pct > 20,  # More than 20% change
            }
        
        return indicators


class DataQualityAgent:
    """
    Main Data Quality Evaluation Agent.
    Orchestrates all analysis agents and generates comprehensive reports.
    """
    
    def __init__(self, validation_results_path: str, 
                 output_dir: str = "data/bronze",
                 use_llm: bool = True):
        self.validation_results_path = Path(validation_results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm
        
        # Load validation results
        self.validation_results = self._load_validation_results()
        
        # Initialize analyzers
        self.statistical_analyzer = LLMStatisticalAnalyzer(use_llm=use_llm)
        self.domain_validator = LLMDomainValidator(use_llm=use_llm)
        self.drift_detector = LLMDriftDetector(use_llm=use_llm)
    
    def _load_validation_results(self) -> Dict[str, Any]:
        """Load validation results from Stage 1."""
        if self.validation_results_path.exists():
            with open(self.validation_results_path, 'r') as f:
                return json.load(f)
        return {}
    
    def evaluate(self, df: pd.DataFrame, 
                  historical_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive quality evaluation with all AI agents.
        
        Args:
            df: DataFrame with ingested data
            historical_stats: Historical statistics for drift detection
            
        Returns:
            Dictionary with complete quality evaluation
        """
        print("Running AI Quality Agents...")
        print("  (Each LLM call will complete before the next one starts)\n")
        
        # Determine priority
        priority = self._determine_priority()
        print(f"  Priority: {priority}\n")
        
        # Run analyses sequentially (one after another)
        analyses = {}
        
        print("=" * 60)
        print("Step 1/4: Statistical Analysis")
        print("=" * 60)
        try:
            analyses['statistical'] = self.statistical_analyzer.analyze(df, self.validation_results)
            print(f"  [COMPLETE] Statistical analysis finished\n")
        except (TimeoutError, Exception) as e:
            print(f"  [WARNING] Statistical analysis failed: {e}")
            print(f"  [CONTINUING] Proceeding without LLM analysis for this step\n")
            analyses['statistical'] = {
                'statistics': {},
                'llm_analysis': None,
                'status': f'Failed: {e}'
            }
        
        print("=" * 60)
        print("Step 2/4: Domain Validation")
        print("=" * 60)
        try:
            analyses['domain'] = self.domain_validator.validate(df)
            print(f"  [COMPLETE] Domain validation finished\n")
        except (TimeoutError, Exception) as e:
            print(f"  [WARNING] Domain validation failed: {e}")
            print(f"  [CONTINUING] Proceeding without LLM analysis for this step\n")
            analyses['domain'] = {
                'domain_checks': {},
                'llm_analysis': None,
                'status': f'Failed: {e}'
            }
        
        print("=" * 60)
        print("Step 3/4: Drift Detection")
        print("=" * 60)
        try:
            if historical_stats is not None:
                analyses['drift'] = self.drift_detector.detect(df, historical_stats)
            else:
                analyses['drift'] = self.drift_detector.detect(df, None)
            print(f"  [COMPLETE] Drift detection finished\n")
        except (TimeoutError, Exception) as e:
            print(f"  [WARNING] Drift detection failed: {e}")
            print(f"  [CONTINUING] Proceeding without LLM analysis for this step\n")
            analyses['drift'] = {
                'current_stats': {},
                'drift_indicators': None,
                'llm_analysis': None,
                'status': f'Failed: {e}'
            }
        
        # Generate action recommendation
        print("=" * 60)
        print("Step 4/4: Action Recommendation")
        print("=" * 60)
        try:
            action = self._recommend_action(analyses, priority)
            print(f"  [COMPLETE] Action recommendation: {action['action']}\n")
        except (TimeoutError, Exception) as e:
            print(f"  [WARNING] Action recommendation failed: {e}")
            print(f"  [CONTINUING] Using rule-based recommendation\n")
            # Fallback to rule-based
            if priority == 'HIGH':
                action = {'action': 'FAIL', 'reasoning': 'Validation failed (LLM unavailable)'}
            elif priority == 'MEDIUM':
                action = {'action': 'WARN', 'reasoning': 'Anomalies detected (LLM unavailable)'}
            else:
                action = {'action': 'PASS', 'reasoning': 'Quality acceptable (LLM unavailable)'}
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'validation_results': self.validation_results,
            'analyses': analyses,
            'recommended_action': action,
        }
        
        return report
    
    def _determine_priority(self) -> str:
        """Determine evaluation priority based on validation results."""
        if not self.validation_results.get('validation_passed', True):
            return 'HIGH'
        if len(self.validation_results.get('anomalies', [])) > 0:
            return 'MEDIUM'
        return 'LOW'
    
    def _recommend_action(self, analyses: Dict, priority: str) -> Dict[str, Any]:
        """
        Use LLM to recommend action based on all analyses.
        """
        if not self.use_llm or not OLLAMA_AVAILABLE:
            # Simple rule-based recommendation
            if priority == 'HIGH':
                return {'action': 'FAIL', 'reasoning': 'Validation failed'}
            elif priority == 'MEDIUM':
                return {'action': 'WARN', 'reasoning': 'Anomalies detected'}
            else:
                return {'action': 'PASS', 'reasoning': 'Quality acceptable'}
        
        try:
            # Summarize analyses for LLM
            analysis_summary = {}
            for key, value in analyses.items():
                if isinstance(value, dict):
                    analysis_summary[key] = {
                        'status': value.get('status', 'unknown'),
                        'has_llm_analysis': value.get('llm_analysis') is not None,
                    }
            
            prompt = f"""
            Based on these quality analyses, recommend action:
            - PASS: Data quality acceptable, proceed with pipeline
            - WARN: Minor issues detected, proceed with caution and review
            - FAIL: Critical issues, block pipeline and require manual intervention
            
            Priority Level: {priority}
            Validation Passed: {self.validation_results.get('validation_passed', True)}
            Anomalies Count: {len(self.validation_results.get('anomalies', []))}
            Analysis Summary: {json.dumps(analysis_summary, indent=2)}
            
            Provide your recommendation (PASS/WARN/FAIL) and brief reasoning (1-2 sentences).
            Format: ACTION: [PASS/WARN/FAIL]
            REASONING: [your reasoning]
            """
            
            action_text = call_ollama_with_timeout(
                prompt=prompt,
                model="llama3.1:8b",
                timeout=120,
                temperature=0.2
            )
            
            # Parse response
            action = 'WARN'  # Default
            reasoning = action_text
            
            if 'PASS' in action_text.upper():
                action = 'PASS'
            elif 'FAIL' in action_text.upper():
                action = 'FAIL'
            
            return {
                'action': action,
                'reasoning': reasoning,
                'llm_response': action_text
            }
        except Exception as e:
            # Fallback to rule-based
            if priority == 'HIGH':
                return {'action': 'FAIL', 'reasoning': f'LLM failed: {e}, validation failed'}
            elif priority == 'MEDIUM':
                return {'action': 'WARN', 'reasoning': f'LLM failed: {e}, anomalies detected'}
            else:
                return {'action': 'PASS', 'reasoning': f'LLM failed: {e}, quality acceptable'}


class QualityReportGenerator:
    """
    Generates comprehensive quality reports in multiple formats.
    """
    
    def __init__(self, output_dir: str = "data/bronze"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_reports(self, quality_report: Dict[str, Any], 
                        format: str = "both") -> Dict[str, Path]:
        """
        Generate reports in specified format(s).
        
        Args:
            quality_report: Complete quality evaluation report
            format: "txt", "md", or "both"
            
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports = {}
        
        if format in ["txt", "both"]:
            txt_path = self.output_dir / f"quality_report_{timestamp}.txt"
            self._generate_txt_report(quality_report, txt_path)
            reports['txt'] = txt_path
        
        if format in ["md", "both"]:
            md_path = self.output_dir / f"quality_report_{timestamp}.md"
            self._generate_md_report(quality_report, md_path)
            reports['md'] = md_path
        
        return reports
    
    def _generate_txt_report(self, report: Dict, path: Path):
        """Generate plain text report."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA QUALITY EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Priority: {report['priority']}\n")
            f.write(f"Recommended Action: {report['recommended_action']['action']}\n")
            f.write(f"\nReasoning: {report['recommended_action'].get('reasoning', 'N/A')}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Validation Results
            f.write("VALIDATION RESULTS\n")
            f.write("-" * 80 + "\n")
            validation = report['validation_results']
            f.write(f"Total Traces: {validation.get('total_traces', 'N/A')}\n")
            f.write(f"Total Files: {validation.get('total_files', 'N/A')}\n")
            f.write(f"Validation Passed: {validation.get('validation_passed', False)}\n")
            f.write(f"Quality Score: {validation.get('quality_score', 'N/A')}\n")
            
            if validation.get('anomalies'):
                f.write(f"\nAnomalies ({len(validation['anomalies'])}):\n")
                for anomaly in validation['anomalies']:
                    f.write(f"  - {anomaly}\n")
            f.write("\n")
            
            # Statistical Analysis
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-" * 80 + "\n")
            stat_analysis = report['analyses'].get('statistical', {})
            if stat_analysis.get('llm_analysis'):
                f.write(stat_analysis['llm_analysis'])
                f.write("\n\n")
            else:
                f.write("LLM Analysis: Not available\n")
                f.write(f"Status: {stat_analysis.get('status', 'unknown')}\n\n")
            
            # Domain Validation
            f.write("=" * 80 + "\n")
            f.write("DOMAIN-SPECIFIC VALIDATION\n")
            f.write("-" * 80 + "\n")
            domain_analysis = report['analyses'].get('domain', {})
            if domain_analysis.get('llm_analysis'):
                f.write(domain_analysis['llm_analysis'])
                f.write("\n\n")
            else:
                f.write("LLM Analysis: Not available\n")
                f.write(f"Status: {domain_analysis.get('status', 'unknown')}\n\n")
            
            # Drift Detection
            f.write("=" * 80 + "\n")
            f.write("DATA DRIFT DETECTION\n")
            f.write("-" * 80 + "\n")
            drift_analysis = report['analyses'].get('drift', {})
            if drift_analysis.get('llm_analysis'):
                f.write(drift_analysis['llm_analysis'])
                f.write("\n\n")
            else:
                f.write(f"Status: {drift_analysis.get('status', 'unknown')}\n")
                if drift_analysis.get('status') == 'no_history':
                    f.write("No historical data available for comparison.\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
    
    def _generate_md_report(self, report: Dict, path: Path):
        """Generate Markdown report."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Data Quality Evaluation Report\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n\n")
            f.write(f"**Priority:** {report['priority']}\n\n")
            f.write(f"**Recommended Action:** `{report['recommended_action']['action']}`\n\n")
            f.write(f"**Reasoning:** {report['recommended_action'].get('reasoning', 'N/A')}\n\n")
            f.write("---\n\n")
            
            # Validation Results
            f.write("## Validation Results\n\n")
            validation = report['validation_results']
            f.write(f"- **Total Traces:** {validation.get('total_traces', 'N/A')}\n")
            f.write(f"- **Total Files:** {validation.get('total_files', 'N/A')}\n")
            f.write(f"- **Validation Passed:** {validation.get('validation_passed', False)}\n")
            f.write(f"- **Quality Score:** {validation.get('quality_score', 'N/A')}\n\n")
            
            if validation.get('anomalies'):
                f.write(f"### Anomalies ({len(validation['anomalies'])})\n\n")
                for anomaly in validation['anomalies']:
                    f.write(f"- {anomaly}\n")
                f.write("\n")
            
            # Statistical Analysis
            f.write("## Statistical Analysis\n\n")
            stat_analysis = report['analyses'].get('statistical', {})
            if stat_analysis.get('llm_analysis'):
                f.write(stat_analysis['llm_analysis'])
                f.write("\n\n")
            else:
                f.write(f"*Status: {stat_analysis.get('status', 'unknown')}*\n\n")
            
            # Domain Validation
            f.write("## Domain-Specific Validation\n\n")
            domain_analysis = report['analyses'].get('domain', {})
            if domain_analysis.get('llm_analysis'):
                f.write(domain_analysis['llm_analysis'])
                f.write("\n\n")
            else:
                f.write(f"*Status: {domain_analysis.get('status', 'unknown')}*\n\n")
            
            # Drift Detection
            f.write("## Data Drift Detection\n\n")
            drift_analysis = report['analyses'].get('drift', {})
            if drift_analysis.get('llm_analysis'):
                f.write(drift_analysis['llm_analysis'])
                f.write("\n\n")
            else:
                f.write(f"*Status: {drift_analysis.get('status', 'unknown')}*\n\n")
                if drift_analysis.get('status') == 'no_history':
                    f.write("> No historical data available for comparison.\n\n")
            
            f.write("---\n\n")
            f.write("*End of Report*\n")
