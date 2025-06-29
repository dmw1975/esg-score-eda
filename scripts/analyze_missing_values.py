#!/usr/bin/env python3
"""
ESG Score Missing Values Analysis Script

This script analyzes missing values in ESG score datasets, identifies patterns,
and applies appropriate imputation strategies.
"""
import os
import sys
import argparse
import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to Python path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the analysis module
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("missing_values_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze missing values in ESG score data')
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='data/raw/df_cleaned.csv',
        help='Path to input CSV file (default: data/raw/df_cleaned.csv)'
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
        default='visualizations',
        help='Directory to save visualizations (default: visualizations)'
    )
    parser.add_argument(
        '--imputation-method', 
        type=str, 
        choices=['global_mean', 'sector_mean', 'sector_median'],
        default='sector_mean',
        help='Imputation method to use (default: sector_mean)'
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for output visualizations (default: 300)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualization_dirs = {
        'missing_values': os.path.join(args.vis_dir, 'missing_values'),
        'imputation': os.path.join(args.vis_dir, 'imputation'),
    }
    
    for dir_path in visualization_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Prepare data for analysis
    df, categorical_cols, numerical_cols = prepare_data_for_analysis(df)
    
    # 1. Analyze missing values
    logger.info("Analyzing missing values")
    
    # Plot overall missing values
    _ = plot_missing_values(
        df, 
        save_path=visualization_dirs['missing_values'],
        filename="missing_values.png",
        dpi=args.dpi
    )
    
    # Plot missingness by sector for numerical variables
    _ = plot_missingness_by_sector(
        df, 
        numerical_cols,
        save_path=visualization_dirs['missing_values'],
        filename="missingness_numerical_by_sector.png",
        dpi=args.dpi
    )
    
    # Plot missingness by sector for categorical variables
    _ = plot_missingness_by_sector(
        df, 
        categorical_cols,
        save_path=visualization_dirs['missing_values'],
        filename="missingness_categorical_by_sector.png",
        dpi=args.dpi
    )
    
    # 2. Test for association between sector and missingness
    logger.info("Testing for association between sector and missingness patterns")
    _, _, results = test_sector_missingness_association(
        df,
        save_path=visualization_dirs['missing_values'],
        filename="chi_square_results.png",
        dpi=args.dpi
    )
    
    # Save results as JSON
    import json
    with open(os.path.join(visualization_dirs['missing_values'], 'sector_missingness_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 3. Evaluate imputation strategies
    logger.info("Evaluating imputation strategies")
    # Include all variables with any missing values
    variables_with_missing = [col for col in numerical_cols if df[col].isnull().sum() > 0]
    
    # Use all numerical variables with missing values
    variables_to_test = variables_with_missing
    
    if variables_to_test:
        logger.info(f"Testing imputation on {len(variables_to_test)} variables")
        results_df = evaluate_imputation_strategies(
            df, 
            variables_to_test=variables_to_test
        )
        
        # Plot imputation comparison
        plot_imputation_comparison(
            results_df,
            save_path=visualization_dirs['imputation'],
            filename="imputation_comparison.png",
            dpi=args.dpi
        )
    else:
        logger.warning("No suitable variables found for imputation testing")
    
    # 4. Apply the best imputation method
    logger.info(f"Applying {args.imputation_method} imputation to all missing values")
    imputed_df = apply_best_imputation(
        df,
        method=args.imputation_method
    )
    
    # 5. Save processed data
    logger.info("Saving processed data")
    cat_path, num_path, full_path = save_processed_data(
        imputed_df,
        categorical_cols,
        numerical_cols,
        args.output_dir
    )
    
    logger.info("Analysis complete!")
    logger.info(f"Categorical data saved to: {cat_path}")
    logger.info(f"Numerical data saved to: {num_path}")
    logger.info(f"Complete imputed data saved to: {full_path}")


if __name__ == "__main__":
    main()