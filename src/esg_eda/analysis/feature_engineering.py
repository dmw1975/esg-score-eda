"""Feature engineering module for ESG EDA pipeline.

This module provides functionality for feature transformation and engineering
including one-hot encoding, normalization, and feature creation.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import the existing implementation
from src.analysis.feature_engineering import (
    identify_categorical_columns,
    one_hot_encode,
    normalize_features,
    engineer_features,
    create_feature_expansion_table,
    create_feature_expansion_visualizations
)

# Re-export all functions
__all__ = [
    'identify_categorical_columns',
    'one_hot_encode',
    'normalize_features',
    'engineer_features',
    'create_feature_expansion_table',
    'create_feature_expansion_visualizations',
    'FeatureEngineer'
]

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for ESG data.
    
    This class provides a high-level interface for feature engineering tasks
    including encoding, transformation, and feature creation.
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the feature engineer.
        
        Args:
            data: Input DataFrame
        """
        self.data = data
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.encoded_data: Optional[pd.DataFrame] = None
        self.normalized_data: Optional[pd.DataFrame] = None
        self.feature_info: Dict[str, Any] = {}
        
    def identify_features(self, 
                         exclude_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Identify categorical and numerical features.
        
        Args:
            exclude_cols: Columns to exclude from feature identification
            
        Returns:
            Dictionary with 'categorical' and 'numerical' keys
        """
        # Identify categorical columns
        self.categorical_cols = identify_categorical_columns(
            self.data,
            exclude_columns=exclude_cols
        )
        
        # Identify numerical columns
        all_cols = set(self.data.columns)
        exclude_set = set(exclude_cols or [])
        categorical_set = set(self.categorical_cols)
        
        self.numerical_cols = [col for col in all_cols 
                              if col not in exclude_set 
                              and col not in categorical_set
                              and pd.api.types.is_numeric_dtype(self.data[col])]
        
        logger.info(f"Identified {len(self.categorical_cols)} categorical and {len(self.numerical_cols)} numerical features")
        
        return {
            'categorical': self.categorical_cols,
            'numerical': self.numerical_cols
        }
    
    def encode_categorical(self, 
                          max_categories: int = 50,
                          drop_first: bool = False) -> pd.DataFrame:
        """Apply one-hot encoding to categorical features.
        
        Args:
            max_categories: Maximum categories to encode per feature
            drop_first: Whether to drop first category to avoid multicollinearity
            
        Returns:
            DataFrame with encoded features
        """
        if not self.categorical_cols:
            logger.warning("No categorical columns identified. Run identify_features first.")
            return self.data
            
        encoded_df, feature_mapping = one_hot_encode(
            self.data,
            categorical_columns=self.categorical_cols,
            max_categories=max_categories,
            drop_first=drop_first
        )
        
        self.encoded_data = encoded_df
        
        # Track encoding info
        self.feature_info['encoding'] = {
            'method': 'one_hot',
            'max_categories': max_categories,
            'drop_first': drop_first,
            'original_features': len(self.categorical_cols),
            'feature_mapping': feature_mapping
        }
        
        return self.encoded_data
    
    def normalize_numerical(self, 
                           method: str = 'standard',
                           exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply normalization to numerical features.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
            exclude_cols: Columns to exclude from normalization
            
        Returns:
            DataFrame with normalized features
        """
        if not self.numerical_cols:
            logger.warning("No numerical columns identified. Run identify_features first.")
            return self.data
            
        self.normalized_data = normalize_features(
            self.data,
            columns=self.numerical_cols,
            method=method,
            exclude_columns=exclude_cols
        )
        
        # Track normalization info
        self.feature_info['normalization'] = {
            'method': method,
            'features_normalized': len(self.numerical_cols)
        }
        
        return self.normalized_data
    
    def apply_all_engineering(self) -> pd.DataFrame:
        """Apply all feature engineering steps.
        
        Returns:
            DataFrame with all engineered features
        """
        result_df = engineer_features(
            self.data,
            categorical_cols=self.categorical_cols,
            one_hot=True,
            normalization='standard'
        )
        
        self.feature_info['complete_engineering'] = {
            'original_features': len(self.data.columns),
            'engineered_features': len(result_df.columns)
        }
        
        return result_df
    
    def visualize_feature_expansion(self, output_dir: Union[str, Path]) -> None:
        """Generate feature expansion visualizations.
        
        Args:
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.encoded_data is not None and 'feature_mapping' in self.feature_info.get('encoding', {}):
            # Create expansion table
            expansion_df = create_feature_expansion_table(
                self.data,
                self.feature_info['encoding']['feature_mapping'],
                save_path=str(output_dir / "feature_expansion_table.csv")
            )
            
            # Create visualizations
            create_feature_expansion_visualizations(
                expansion_df,
                save_path=str(output_dir / "feature_expansion")
            )
    
    def save_feature_info(self, output_path: Union[str, Path]) -> None:
        """Save feature engineering information.
        
        Args:
            output_path: Path to save feature info
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare info for JSON serialization
        info_to_save = {
            'categorical_features': self.categorical_cols,
            'numerical_features': self.numerical_cols,
            'feature_info': self.feature_info
        }
        
        with open(output_path, 'w') as f:
            json.dump(info_to_save, f, indent=2)
            
        logger.info(f"Feature info saved to {output_path}")