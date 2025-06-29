"""Missing values analysis module for ESG EDA pipeline.

This module provides functionality for analyzing and handling missing values
in ESG score datasets, including visualization and imputation strategies.
"""

from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import the existing implementation
from src.analysis.missing_values import (
    plot_missing_values,
    plot_missingness_by_sector,
    test_sector_missingness_association,
    evaluate_imputation_strategies,
    plot_imputation_comparison,
    apply_best_imputation,
    prepare_data_for_analysis,
    save_processed_data
)

# Re-export all functions
__all__ = [
    'plot_missing_values',
    'plot_missingness_by_sector',
    'test_sector_missingness_association',
    'evaluate_imputation_strategies',
    'plot_imputation_comparison',
    'apply_best_imputation',
    'prepare_data_for_analysis',
    'save_processed_data',
    'MissingValuesAnalyzer'
]

logger = logging.getLogger(__name__)


class MissingValuesAnalyzer:
    """Analyzer for missing values in ESG data.
    
    This class provides a high-level interface for missing values analysis,
    wrapping the existing functional implementation with additional features
    like configuration management and state tracking.
    """
    
    def __init__(self, data: pd.DataFrame, sector_column: str = 'gics_sector'):
        """Initialize the analyzer.
        
        Args:
            data: Input DataFrame
            sector_column: Column name for sector grouping
        """
        self.data = data
        self.sector_column = sector_column
        self.missing_stats: Optional[pd.DataFrame] = None
        self.imputation_results: Optional[Dict] = None
        
    def analyze(self) -> Dict[str, Union[pd.DataFrame, float]]:
        """Run complete missing values analysis.
        
        Returns:
            Dictionary containing:
                - missing_stats: DataFrame with missing value statistics
                - sector_association: Chi-square test results
                - recommended_strategy: Best imputation strategy
        """
        # Calculate missing value statistics
        self.missing_stats = self._calculate_missing_stats()
        
        # Test sector association
        sector_results = test_sector_missingness_association(
            self.data, 
            self.sector_column
        )
        
        # Evaluate imputation strategies
        self.imputation_results = evaluate_imputation_strategies(
            self.data,
            self.sector_column
        )
        
        return {
            'missing_stats': self.missing_stats,
            'sector_association': sector_results,
            'imputation_results': self.imputation_results
        }
    
    def _calculate_missing_stats(self) -> pd.DataFrame:
        """Calculate missing value statistics.
        
        Returns:
            DataFrame with missing value counts and percentages
        """
        missing_count = self.data.isnull().sum()
        missing_pct = (missing_count / len(self.data)) * 100
        
        stats = pd.DataFrame({
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })
        
        return stats.sort_values('missing_pct', ascending=False)
    
    def visualize(self, output_dir: Union[str, Path]) -> None:
        """Generate all missing value visualizations.
        
        Args:
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot overall missing values
        plot_missing_values(
            self.data,
            save_path=str(output_dir),
            filename="missing_values.png"
        )
        
        # Plot by sector if sector column exists
        if self.sector_column in self.data.columns:
            plot_missingness_by_sector(
                self.data,
                self.sector_column,
                save_path=str(output_dir)
            )
    
    def impute(self, strategy: Optional[str] = None) -> pd.DataFrame:
        """Apply imputation to handle missing values.
        
        Args:
            strategy: Imputation strategy ('global_mean', 'sector_mean', etc.)
                     If None, uses the best strategy from evaluation
        
        Returns:
            DataFrame with imputed values
        """
        if strategy is None and self.imputation_results:
            # Use best strategy from evaluation
            strategy = min(
                self.imputation_results.items(),
                key=lambda x: x[1]['avg_rmse']
            )[0]
            
        logger.info(f"Applying imputation strategy: {strategy}")
        
        return apply_best_imputation(
            self.data,
            self.sector_column,
            strategy
        )