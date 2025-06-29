#!/usr/bin/env python3
"""
ESG Score Outlier Detection Script

This script detects outliers in ESG score datasets using multiple methods,
adds outlier flags to the data, and processes outliers using the specified method.
"""
import os
import sys
import argparse
import pandas as pd
import logging
from pathlib import Path

# Add the project root to Python path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the analysis module
from src.analysis.outlier_detection import (
    analyze_outliers,
    process_and_evaluate_outliers,
    process_outliers_with_mean,
    process_outliers_with_median, 
    process_outliers_with_winsorization,
    process_outliers_with_trimming
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("outlier_detection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect and process outliers in ESG score data')
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='data/processed/imputed_data.csv',
        help='Path to input CSV file (default: data/processed/imputed_data.csv)'
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
        default='visualizations/outliers',
        help='Directory to save visualizations (default: visualizations/outliers)'
    )
    parser.add_argument(
        '--detection-method', 
        type=str, 
        choices=['iqr', 'zscore', 'isolation_forest', 'all'],
        default='all',
        help='Outlier detection method to use (default: all)'
    )
    parser.add_argument(
        '--zscore-threshold',
        type=float,
        default=3.0,
        help='Threshold for Z-score outlier detection (default: 3.0)'
    )
    parser.add_argument(
        '--iqr-multiplier',
        type=float,
        default=1.5,
        help='Multiplier for IQR outlier detection (default: 1.5)'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Contamination parameter for Isolation Forest (default: 0.05)'
    )
    parser.add_argument(
        '--processing-method',
        type=str,
        choices=['mean', 'median', 'winsorize', 'trim', 'none'],
        default='mean',
        help='Method to process outliers (default: mean)'
    )
    parser.add_argument(
        '--flag-prefix',
        type=str,
        default='',
        help='Prefix for outlier flag columns to use for processing (default: auto-select based on method)'
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
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # 1. Detect outliers
    add_isolation_forest = args.detection_method in ['isolation_forest', 'all']
    
    if args.detection_method == 'isolation_forest' and not add_isolation_forest:
        logger.error("Isolation Forest detection requested but sklearn is not available")
        sys.exit(1)
    
    logger.info(f"Detecting outliers using method: {args.detection_method}")
    df_with_flags, outlier_results = analyze_outliers(
        df,
        iqr_threshold=args.iqr_multiplier,
        zscore_threshold=args.zscore_threshold,
        add_isolation_forest=add_isolation_forest,
        contamination=args.contamination,
        save_path=args.vis_dir,
        dpi=args.dpi
    )
    
    # 2. Process outliers if requested
    if args.processing_method != 'none':
        # Determine which flag columns to use for processing
        if args.flag_prefix:
            flag_prefix = args.flag_prefix
        elif args.detection_method == 'iqr':
            flag_prefix = 'iqr_out_'
        elif args.detection_method == 'zscore':
            flag_prefix = 'zscore_out_'
        elif args.detection_method == 'isolation_forest':
            flag_prefix = 'forest_out_'
        else:
            # Default to IQR for 'all' method
            flag_prefix = 'iqr_out_'
        
        logger.info(f"Processing outliers using method: {args.processing_method} with flags: {flag_prefix}")
        processed_df = process_and_evaluate_outliers(
            df_with_flags,
            method=args.processing_method,
            outlier_flags_prefix=flag_prefix,
            save_path=args.vis_dir,
            dpi=args.dpi
        )
        
        # Save processed data
        output_path = os.path.join(args.output_dir, f'outliers_{args.processing_method}_processed.csv')
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    else:
        # Save data with outlier flags
        output_path = os.path.join(args.output_dir, 'outliers_flagged.csv')
        df_with_flags.to_csv(output_path, index=False)
        logger.info(f"Data with outlier flags saved to {output_path}")
    
    # 3. Save outlier detection results
    logger.info("Saving outlier detection summary")
    
    # Create a summary report
    with open(os.path.join(args.vis_dir, 'outlier_detection_report.txt'), 'w') as f:
        f.write(f"ESG Score Data Outlier Detection Report\n")
        f.write(f"=====================================\n\n")
        f.write(f"Input file: {args.input}\n")
        f.write(f"Detection method: {args.detection_method}\n\n")
        
        if 'iqr' in outlier_results:
            f.write(f"IQR Outlier Detection Results:\n")
            f.write(f"-----------------------------\n")
            f.write(f"Total outliers: {outlier_results['iqr']['counts'].sum()}\n")
            f.write(f"Outliers in lower tail: {outlier_results['iqr']['details']['lower_count']}\n")
            f.write(f"Outliers in upper tail: {outlier_results['iqr']['details']['upper_count']}\n\n")
            
            f.write("Top 10 variables with most outliers:\n")
            top_vars = outlier_results['iqr']['counts'].sort_values(ascending=False).head(10)
            for var, count in top_vars.items():
                f.write(f"  {var}: {count} outliers\n")
            f.write("\n")
        
        if 'zscore' in outlier_results:
            f.write(f"Z-Score Outlier Detection Results:\n")
            f.write(f"--------------------------------\n")
            f.write(f"Z-score threshold: {args.zscore_threshold}\n")
            f.write(f"Total outliers: {outlier_results['zscore']['counts'].sum()}\n")
            f.write(f"Outliers in lower tail: {outlier_results['zscore']['details']['lower_count']}\n")
            f.write(f"Outliers in upper tail: {outlier_results['zscore']['details']['upper_count']}\n\n")
            
            f.write("Top 10 variables with most outliers:\n")
            top_vars = outlier_results['zscore']['counts'].sort_values(ascending=False).head(10)
            for var, count in top_vars.items():
                f.write(f"  {var}: {count} outliers\n")
            f.write("\n")
        
        if args.processing_method != 'none':
            f.write(f"Outlier Processing:\n")
            f.write(f"------------------\n")
            f.write(f"Method: {args.processing_method}\n")
            f.write(f"Flag columns used: {flag_prefix}*\n")
            f.write(f"Output file: {output_path}\n")
    
    logger.info("Outlier detection and processing complete!")


if __name__ == "__main__":
    main()