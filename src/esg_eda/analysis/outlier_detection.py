"""Outlier detection module for ESG EDA pipeline.

This module provides functionality for detecting and handling outliers
in ESG score datasets using multiple methods (IQR, Z-score, Isolation Forest).
"""

from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import the existing implementation
from src.analysis.outlier_detection import (
    detect_iqr_outliers,
    detect_zscore_outliers,
    detect_isolation_forest_outliers,
    plot_outlier_comparison,
    plot_boxplots,
    analyze_outliers,
    process_outliers_with_mean,
    process_outliers_with_median
)

# Re-export all functions
__all__ = [
    'detect_iqr_outliers',
    'detect_zscore_outliers',
    'detect_isolation_forest_outliers',
    'plot_outlier_comparison',
    'plot_boxplots',
    'analyze_outliers',
    'process_outliers_with_mean',
    'process_outliers_with_median',
    'OutlierDetector'
]

logger = logging.getLogger(__name__)


class OutlierDetector:
    """Detector for outliers in ESG data.
    
    This class provides a high-level interface for outlier detection,
    supporting multiple detection methods and visualization options.
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the detector.
        
        Args:
            data: Input DataFrame
        """
        self.data = data
        self.outliers_iqr: Optional[pd.DataFrame] = None
        self.outliers_zscore: Optional[pd.DataFrame] = None
        self.outliers_forest: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[Dict] = None
        
    def detect_all_methods(self, 
                          iqr_threshold: float = 1.5,
                          z_threshold: float = 3.0,
                          contamination: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Run all outlier detection methods.
        
        Args:
            iqr_threshold: Threshold for IQR method
            z_threshold: Threshold for Z-score method
            contamination: Contamination parameter for Isolation Forest
            
        Returns:
            Dictionary containing results from each method
        """
        # Run comprehensive analysis
        self.analysis_results = analyze_outliers(
            self.data,
            iqr_threshold=iqr_threshold,
            zscore_threshold=z_threshold,
            forest_contamination=contamination
        )
        
        # Extract individual results
        self.outliers_iqr = self.data[[col for col in self.data.columns if col.startswith('iqr_out_')]]
        self.outliers_zscore = self.data[[col for col in self.data.columns if col.startswith('zscore_out_')]]
        
        if 'outlier_summary' in self.analysis_results:
            logger.info(f"Outlier detection complete: {self.analysis_results['outlier_summary']}")
        
        return {
            'iqr': self.outliers_iqr,
            'zscore': self.outliers_zscore,
            'analysis': self.analysis_results
        }
    
    def visualize(self, output_dir: Union[str, Path]) -> None:
        """Generate all outlier visualizations.
        
        Args:
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Identify numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Plot boxplots
        if numerical_cols:
            plot_boxplots(
                self.data,
                numerical_cols,
                save_path=str(output_dir / "boxplots.png")
            )
        
        # Plot comparison if we have multiple methods
        if self.outliers_iqr is not None and self.outliers_zscore is not None:
            plot_outlier_comparison(
                self.outliers_iqr,
                self.outliers_zscore,
                forest_outliers=self.outliers_forest,
                save_path=str(output_dir / "outliers_comparison.png")
            )
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """Get summary statistics of detected outliers.
        
        Returns:
            DataFrame with outlier counts by column and method
        """
        summary_data = []
        
        if self.outliers_iqr is not None:
            for col in self.outliers_iqr.columns:
                base_col = col.replace('iqr_out_', '')
                count = self.outliers_iqr[col].sum()
                summary_data.append({
                    'column': base_col,
                    'method': 'IQR',
                    'outlier_count': count,
                    'outlier_pct': (count / len(self.data)) * 100
                })
        
        if self.outliers_zscore is not None:
            for col in self.outliers_zscore.columns:
                base_col = col.replace('zscore_out_', '')
                count = self.outliers_zscore[col].sum()
                summary_data.append({
                    'column': base_col,
                    'method': 'Z-Score',
                    'outlier_count': count,
                    'outlier_pct': (count / len(self.data)) * 100
                })
                
        return pd.DataFrame(summary_data)
    
    def process_outliers(self, method: str = 'mean', 
                        outlier_type: str = 'iqr') -> pd.DataFrame:
        """Process outliers using specified method.
        
        Args:
            method: Processing method ('mean', 'median', 'winsorize', 'trim')
            outlier_type: Which outliers to process ('iqr', 'zscore')
            
        Returns:
            DataFrame with processed values
        """
        prefix = f'{outlier_type}_out_'
        
        if method == 'mean':
            return process_outliers_with_mean(self.data, prefix)
        elif method == 'median':
            return process_outliers_with_median(self.data, prefix)
        else:
            raise ValueError(f"Unsupported processing method: {method}")
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save all detection results.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save outlier DataFrames
        if self.outliers_iqr is not None and not self.outliers_iqr.empty:
            self.outliers_iqr.to_csv(output_dir / "outliers_iqr.csv", index=False)
            
        if self.outliers_zscore is not None and not self.outliers_zscore.empty:
            self.outliers_zscore.to_csv(output_dir / "outliers_zscore.csv", index=False)
            
        # Save summary
        summary = self.get_outlier_summary()
        if not summary.empty:
            summary.to_csv(output_dir / "outlier_summary.csv", index=False)
            
        # Save analysis results
        if self.analysis_results:
            import json
            with open(output_dir / "outlier_analysis.json", 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                results_json = {k: v.tolist() if hasattr(v, 'tolist') else v 
                              for k, v in self.analysis_results.items()}
                json.dump(results_json, f, indent=2)