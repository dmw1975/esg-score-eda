#!/usr/bin/env python3
"""
Generate Outlier Comparison Plots Script
This script generates multi-boxplot comparison visualizations for outlier handling.
It uses the same approach as in notebook 4_outlier_processing_v1.2.ipynb but
encapsulated in a reusable script.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root to Python path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import location utilities and analysis functions
try:
    from location import Location
except ImportError:
    sys.path.insert(0, os.getcwd())
    from location import Location

from src.analysis.outlier_detection import plot_multi_boxplot_comparisons

def main():
    """Main function to generate outlier comparison plots."""
    # Set up file paths
    base_dir = os.getcwd()
    location = Location(base_dir)
    
    # Import processed data with outliers
    subdirectory = 'data/processed'
    file_name = 'outliers_mean_processed.csv'
    full_path = location.get_path(subdirectory, file_name)
    
    # Check if the file exists
    if not os.path.exists(full_path):
        print(f"Error: File {full_path} does not exist.")
        print("Please run the outlier detection scripts first.")
        sys.exit(1)
    
    # Load the data
    print(f"Loading data from {full_path}")
    df = pd.read_csv(full_path, index_col='issuer_name')
    df = df.convert_dtypes()
    
    # Define parameter scope based on column headings
    forest_outscope = ["forest_outlier", "forest_outlier_tuned", "anomaly_score", "predicted_outlier"]
    forest_inscope = ["anomaly_score", "predicted_outlier"]
    
    # Define column scopes for IQR and Z-score analysis
    iqr_scope = [col for col in df.columns if not (col.startswith('zscore_out_') or col in forest_outscope)]
    z_scope = [col for col in df.columns if not (col.startswith('iqr_out_') or col in forest_outscope)]
    
    # Create dataframes for IQR and Z-score analysis
    iqr_df = df[iqr_scope]
    z_df = df[z_scope]
    
    # Process IQR outliers
    print("Processing IQR outliers...")
    
    # Get non-IQR-out columns that are numeric
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_iqr_out_cols = [col for col in iqr_df.columns if 
                        not col.startswith("iqr_out_") and col in numeric_cols]
    
    print(f"Found {len(non_iqr_out_cols)} numeric columns for analysis")
    
    # Add modified columns with 'iqr_mod_' prefix
    for col in non_iqr_out_cols:
        iqr_df = iqr_df.copy()  # Ensure we're working on a copy
        iqr_df.loc[:, f"iqr_mod_{col}"] = iqr_df[col].copy()
    
    # Calculate the mean for each column
    means_iqr = {col: iqr_df[col].mean() for col in non_iqr_out_cols}
    
    # Replace outliers with mean values
    iqr_out_cols = [col for col in iqr_df.columns if col.startswith("iqr_out_")]
    
    for col in non_iqr_out_cols:
        iqr_mod_col = f"iqr_mod_{col}"
        corresponding_iqr_out_col = f"iqr_out_{col}"
        
        if corresponding_iqr_out_col in iqr_out_cols:
            # Create mask for outliers
            mask = iqr_df[corresponding_iqr_out_col] == 1
            
            # Convert mean to match column dtype if needed
            if pd.api.types.is_integer_dtype(iqr_df[iqr_mod_col]):
                try:
                    iqr_df.loc[mask, iqr_mod_col] = int(means_iqr[col])
                except (ValueError, TypeError):
                    # If we can't convert to int, create a new column as float
                    # and use that instead
                    new_col_name = f"iqr_mod_float_{col}"
                    iqr_df[new_col_name] = iqr_df[col].astype(float)
                    iqr_df.loc[mask, new_col_name] = means_iqr[col]
                    # Replace the column name in our list for plotting
                    idx = non_iqr_out_cols.index(col)
                    non_iqr_out_cols[idx] = new_col_name
            else:
                iqr_df.loc[mask, iqr_mod_col] = means_iqr[col]
    
    # Process Z-score outliers
    print("Processing Z-score outliers...")
    
    # Get non-Z-out columns that are numeric
    non_z_out_cols = [col for col in z_df.columns if 
                      not col.startswith("zscore_out_") and col in numeric_cols]
    
    print(f"Found {len(non_z_out_cols)} numeric columns for Z-score analysis")
    
    # Add modified columns with 'z_mod_' prefix
    for col in non_z_out_cols:
        z_df = z_df.copy()  # Ensure we're working on a copy
        z_df.loc[:, f"z_mod_{col}"] = z_df[col].copy()
    
    # Calculate the mean for each column
    means_z = {col: z_df[col].mean() for col in non_z_out_cols}
    
    # Replace outliers with mean values
    z_out_cols = [col for col in z_df.columns if col.startswith("zscore_out_")]
    
    for col in non_z_out_cols:
        z_mod_col = f"z_mod_{col}"
        corresponding_z_out_col = f"zscore_out_{col}"
        
        if corresponding_z_out_col in z_out_cols:
            mask = z_df[corresponding_z_out_col] == 1
            
            # Convert mean to match column dtype if needed
            if pd.api.types.is_integer_dtype(z_df[z_mod_col]):
                try:
                    z_df.loc[mask, z_mod_col] = int(means_z[col])
                except (ValueError, TypeError):
                    # If we can't convert to int, create a new column as float
                    # and use that instead
                    new_col_name = f"z_mod_float_{col}"
                    z_df[new_col_name] = z_df[col].astype(float)
                    z_df.loc[mask, new_col_name] = means_z[col]
                    # Replace the column name in our list for plotting
                    idx = non_z_out_cols.index(col)
                    non_z_out_cols[idx] = new_col_name
            else:
                z_df.loc[mask, z_mod_col] = means_z[col]
    
    # Create output directory
    output_dir = location.get_path('visualizations/outliers', 'boxplots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comparison plots
    print("Generating IQR comparison plots...")
    iqr_figures = plot_multi_boxplot_comparisons(
        df=iqr_df,
        original_cols=non_iqr_out_cols,
        modified_cols_prefix='iqr_mod_',
        num_cols=3,
        plots_per_figure=3,
        save_dir=output_dir,
        dpi=300,
        fig_name_prefix='comparison_boxplots_iqr_handling'
    )
    
    print("Generating Z-score comparison plots...")
    z_figures = plot_multi_boxplot_comparisons(
        df=z_df,
        original_cols=non_z_out_cols,
        modified_cols_prefix='z_mod_',
        num_cols=3,
        plots_per_figure=3,
        save_dir=output_dir,
        dpi=300,
        fig_name_prefix='comparison_boxplots_z_handling'
    )
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()