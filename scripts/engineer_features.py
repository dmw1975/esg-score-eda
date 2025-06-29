#!/usr/bin/env python3
"""
ESG Score Feature Engineering Script

This script performs feature engineering on ESG score datasets, including
one-hot encoding, normalization, and other transformations.
"""
import os
import sys
import argparse
import pandas as pd
import logging
import json
from pathlib import Path

# Add the project root to Python path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the analysis module
from src.analysis.feature_engineering import (
    engineer_features,
    identify_categorical_columns,
    one_hot_encode,
    normalize_features,
    create_interaction_features,
    polynomial_features
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Perform feature engineering on ESG score data')
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='data/processed/outliers_mean_processed.csv',
        help='Path to input CSV file (default: data/processed/outliers_mean_processed.csv)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/processed',
        help='Directory to save processed data (default: data/processed)'
    )
    parser.add_argument(
        '--vis-dir', 
        type=str, 
        default='visualizations/feature_engineering',
        help='Directory to save visualizations (default: visualizations/feature_engineering)'
    )
    parser.add_argument(
        '--one-hot', 
        action='store_true',
        help='Apply one-hot encoding to categorical variables'
    )
    parser.add_argument(
        '--normalization', 
        type=str, 
        choices=['standard', 'minmax', 'robust', 'none'],
        default='standard',
        help='Normalization method for numerical features (default: standard)'
    )
    parser.add_argument(
        '--interactions', 
        action='store_true',
        help='Create interaction features for key numerical variables'
    )
    parser.add_argument(
        '--polynomials', 
        action='store_true',
        help='Create polynomial features for numerical variables'
    )
    parser.add_argument(
        '--poly-degree', 
        type=int, 
        default=2,
        help='Degree of polynomial features (default: 2)'
    )
    parser.add_argument(
        '--categorical-file', 
        type=str, 
        default=None,
        help='JSON file listing categorical columns (optional)'
    )
    parser.add_argument(
        '--exclude-file', 
        type=str, 
        default=None,
        help='JSON file listing columns to exclude (optional)'
    )
    parser.add_argument(
        '--interaction-file', 
        type=str, 
        default=None,
        help='JSON file listing feature pairs for interactions (optional)'
    )
    parser.add_argument(
        '--poly-columns-file', 
        type=str, 
        default=None,
        help='JSON file listing columns for polynomial features (optional)'
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for output visualizations (default: 300)'
    )
    
    return parser.parse_args()


def load_json_file(file_path, default=None):
    """Load data from a JSON file."""
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
    return default


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Load configuration files
    categorical_cols = load_json_file(args.categorical_file)
    exclude_cols = load_json_file(args.exclude_file)
    interaction_pairs = load_json_file(args.interaction_file)
    poly_columns = load_json_file(args.poly_columns_file)
    
    # Adjust normalization if 'none' is specified
    normalization = None if args.normalization == 'none' else args.normalization
    
    # If interaction_pairs is not specified but interactions flag is set,
    # create default interaction pairs
    if args.interactions and not interaction_pairs:
        # Get key financial ratios for interactions
        financial_cols = [col for col in df.columns if any(x in col.lower() for x in 
                        ['ratio', 'return', 'margin', 'yield', 'growth'])]
        
        # Create pairs for interaction (limit to 10 pairs to avoid explosion)
        interaction_pairs = []
        for i, col1 in enumerate(financial_cols[:5]):
            for col2 in financial_cols[i+1:i+5]:
                interaction_pairs.append([col1, col2])
        
        logger.info(f"Created {len(interaction_pairs)} default interaction pairs")
    
    # If poly_columns is not specified but polynomials flag is set,
    # create default poly_columns
    if args.polynomials and not poly_columns:
        # Get key metrics for polynomial features
        poly_columns = [col for col in df.columns if any(x in col.lower() for x in 
                        ['ratio', 'return', 'margin', 'yield'])][:5]
        
        logger.info(f"Selected {len(poly_columns)} default columns for polynomial features")
    
    # Apply feature engineering
    logger.info("Applying feature engineering pipeline")
    processed_df, feature_info = engineer_features(
        df,
        categorical_cols=categorical_cols,
        exclude_cols=exclude_cols,
        one_hot=args.one_hot,
        normalization=normalization,
        interactions=interaction_pairs if args.interactions else None,
        polynomials=poly_columns if args.polynomials else None,
        poly_degree=args.poly_degree,
        save_path=args.vis_dir,
        dpi=args.dpi
    )
    
    # Save processed data
    output_path = os.path.join(args.output_dir, 'engineered_features.csv')
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    # Save feature information
    info_path = os.path.join(args.vis_dir, 'feature_engineering_info.json')
    
    # Convert feature_info to be JSON serializable
    for trans in feature_info['transformations']:
        if 'columns' in trans:
            trans['columns'] = list(trans['columns'])
        if 'feature_mapping' in trans:
            serializable_mapping = {}
            for k, v in trans['feature_mapping'].items():
                serializable_mapping[k] = list(v)
            trans['feature_mapping'] = serializable_mapping
    
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info(f"Feature engineering information saved to {info_path}")
    
    # Summary of transformations
    logger.info("Feature engineering summary:")
    logger.info(f"  - Original shape: {feature_info['original_shape']}")
    logger.info(f"  - Final shape: {feature_info['final_shape']}")
    logger.info(f"  - Feature count change: +{feature_info['feature_count_change']}")
    
    for i, trans in enumerate(feature_info['transformations']):
        logger.info(f"  - Transformation {i+1}: {trans['type']}")
    
    logger.info("Feature engineering complete!")


if __name__ == "__main__":
    main()