#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outlier Imputation script for ESG Score analysis

This script handles outlier imputation using mean values for both IQR and Z-score
detected outliers and creates visualizations comparing original and imputed distributions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

# Add the project root to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from location import Location
from src.analysis.outlier_detection import plot_multi_boxplot_comparisons


def impute_iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute outliers detected by IQR method with mean values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data with IQR outlier indicators

    Returns
    -------
    pd.DataFrame
        DataFrame with IQR outliers imputed
    """
    # First check if we have any IQR outlier columns
    iqr_out_cols_check = [col for col in df.columns if col.startswith("iqr_out_")]
    if not iqr_out_cols_check:
        print("No IQR outlier columns found. Skipping IQR imputation.")
        return pd.DataFrame()  # Return empty dataframe if no columns to process
    
    # Define parameter scope based on column headings 
    forest_outscope = ["forest_outlier", "forest_outlier_tuned", "anomaly_score", "predicted_outlier"]
    iqr_scope = [col for col in df.columns if not (col.startswith('zscore_out_') or col in forest_outscope)]

    # Define df with iqr scope
    iqr_df = df[iqr_scope].copy()

    # Step a: Add additional columns for all columns not starting with "iqr_out_" with the prefix "iqr_mod_"
    non_iqr_out_cols = [col for col in iqr_df.columns if not col.startswith("iqr_out_")]
    for col in non_iqr_out_cols:
        iqr_df.loc[:, f"iqr_mod_{col}"] = iqr_df[col].copy()  # Explicit copy to avoid warnings

    # Step b: Calculate the mean value of the columns not starting with "iqr_out_"
    means = {col: iqr_df[col].mean() for col in non_iqr_out_cols}

    # Step c: Update "iqr_mod_" columns based on conditions
    iqr_out_cols = [col for col in iqr_df.columns if col.startswith("iqr_out_")]

    for col in non_iqr_out_cols:
        iqr_mod_col = f"iqr_mod_{col}"
        corresponding_iqr_out_col = f"iqr_out_{col}"  # Assuming the naming pattern is consistent

        if corresponding_iqr_out_col in iqr_out_cols:
            # Explicitly creating a mask to avoid potential SettingWithCopyWarning
            mask = iqr_df[corresponding_iqr_out_col] == 1

            # Convert mean to match the column dtype
            if pd.api.types.is_integer_dtype(iqr_df[iqr_mod_col]):
                try:
                    iqr_df.loc[mask, iqr_mod_col] = int(means[col])  # Cast mean to int
                except (ValueError, TypeError):
                    # If we can't convert to int, use float
                    iqr_df.loc[mask, iqr_mod_col] = means[col]
            else:
                iqr_df.loc[mask, iqr_mod_col] = means[col]  # Assign float if dtype is float

    return iqr_df


def impute_zscore_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute outliers detected by Z-score method with mean values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data with Z-score outlier indicators

    Returns
    -------
    pd.DataFrame
        DataFrame with Z-score outliers imputed
    """
    # First check if we have any Z-score outlier columns
    z_out_cols_check = [col for col in df.columns if col.startswith("zscore_out_")]
    if not z_out_cols_check:
        print("No Z-score outlier columns found. Skipping Z-score imputation.")
        return pd.DataFrame()  # Return empty dataframe if no columns to process
    
    # Define parameter scope based on column headings 
    forest_outscope = ["forest_outlier", "forest_outlier_tuned", "anomaly_score", "predicted_outlier"]
    z_scope = [col for col in df.columns if not (col.startswith('iqr_out_') or col in forest_outscope)]

    # Define df with z scope
    z_df = df[z_scope].copy()

    # Step a: Add additional columns for all columns not starting with "zscore_out_" with the prefix "z_mod_"
    non_z_out_cols = [col for col in z_df.columns if not col.startswith('zscore_out_')]
    for col in non_z_out_cols:
        z_df.loc[:, f"z_mod_{col}"] = z_df[col].copy()

    # Step b: Calculate the mean value of the columns in "z_scope"
    means = {col: z_df[col].mean() for col in non_z_out_cols}

    # Step c: Update "z_mod_" columns based on conditions
    z_out_cols = [col for col in z_df.columns if col.startswith("zscore_out_")]

    for col in non_z_out_cols:
        z_mod_col = f"z_mod_{col}"
        corresponding_z_out_col = f"zscore_out_{col}"  # Assuming the naming pattern is consistent

        if corresponding_z_out_col in z_out_cols:
            # Explicitly creating a mask to avoid potential SettingWithCopyWarning
            mask = z_df[corresponding_z_out_col] == 1

            # Convert mean to match the column dtype
            if pd.api.types.is_integer_dtype(z_df[z_mod_col]):
                try:
                    z_df.loc[mask, z_mod_col] = int(means[col])  # Cast mean to int
                except (ValueError, TypeError):
                    # If we can't convert to int, use float
                    z_df.loc[mask, z_mod_col] = means[col]
            else:
                z_df.loc[mask, z_mod_col] = means[col]  # Assign float if dtype is float

    return z_df


def process_and_impute_outliers(location: Location) -> None:
    """
    Process the outlier data by imputing outliers detected by IQR and Z-score methods
    with mean values and create visualizations comparing original and imputed distributions.

    Parameters
    ----------
    location : Location
        Location object for handling file paths
    """
    # Import outlier data
    # Try to load from multiple locations
    try:
        subdirectory = 'data/processed'
        file_name = 'outlier_all.csv'
        full_path = location.get_path(subdirectory, file_name)
        df = pd.read_csv(full_path, index_col='issuer_name')
        print(f"Using data file: {full_path}")
    except FileNotFoundError:
        try:
            # Fallback to data directory
            subdirectory = 'data'
            file_name = 'outlier_all.csv'
            full_path = location.get_path(subdirectory, file_name)
            df = pd.read_csv(full_path, index_col='issuer_name')
            print(f"Using data file: {full_path}")
        except FileNotFoundError:
            try:
                # Check if the outliers_iqr_z.csv exists
                subdirectory = 'data/processed'
                file_name = 'outliers_iqr_z.csv'
                full_path = location.get_path(subdirectory, file_name)
                df = pd.read_csv(full_path, index_col='issuer_name')
                print(f"Falling back to data file: {full_path}")
            except FileNotFoundError:
                try:
                    # Try to find the mean processed data
                    subdirectory = 'data/processed'
                    file_name = 'outliers_mean_processed.csv'
                    full_path = location.get_path(subdirectory, file_name)
                    df = pd.read_csv(full_path, index_col='issuer_name')
                    print(f"Using mean processed data file: {full_path}")
                except FileNotFoundError:
                    raise FileNotFoundError("Required outlier detection files not found. Please run outlier detection first.")
    
    # Check if we have the necessary columns for imputation
    has_iqr_cols = any(col.startswith("iqr_out_") for col in df.columns)
    has_zscore_cols = any(col.startswith("zscore_out_") for col in df.columns)
    
    if not (has_iqr_cols or has_zscore_cols):
        print("Warning: No IQR or Z-score outlier columns found. This is expected if only isolation forest outlier detection was run.")
        print("Creating forest outlier imputation data instead.")
        
        # Create a copy of the dataframe to work with
        forest_df = df.copy()
        
        # Check if we have forest outlier columns
        if "forest_outlier_tuned" in forest_df.columns:
            print("Processing tuned isolation forest outliers...")
            # Get numeric columns
            numeric_cols = forest_df.select_dtypes(include=['number']).columns.tolist()
            # Exclude outlier flags and other metadata
            exclude_cols = ["forest_outlier", "forest_outlier_tuned", "anomaly_score", "predicted_outlier"]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Create columns with imputed values (using mean)
            for col in numeric_cols:
                forest_df[f"forest_mod_{col}"] = forest_df[col].copy()
                # Get the mean value
                col_mean = forest_df[col].mean()
                # Create mask for outliers
                mask = forest_df["forest_outlier_tuned"] == -1
                # Apply imputation
                forest_df.loc[mask, f"forest_mod_{col}"] = col_mean
                
            # Save the imputed data
            file_name = 'outlier_forest_imputed.csv'
            full_path = location.get_path('data/processed', file_name)
            forest_df.to_csv(full_path, index=True)
            print(f"Saved forest outlier imputed data to: {full_path}")
            
            # Generate comparison plots
            print("Generating isolation forest comparison plots...")
            # We'll create a subset of columns for visualization (top 15 by variance)
            try:
                var_series = forest_df[numeric_cols].var()
                top_cols = var_series.sort_values(ascending=False).head(15).index.tolist()
                
                forest_figures = plot_multi_boxplot_comparisons(
                    df=forest_df, 
                    original_cols=top_cols, 
                    modified_cols_prefix='forest_mod_',
                    num_cols=3, 
                    plots_per_figure=5, 
                    save_dir='visualizations/outliers/boxplots', 
                    dpi=300, 
                    fig_name_prefix='comparison_boxplots_forest_handling'
                )
                print("Isolation forest comparison plots generated successfully.")
            except Exception as e:
                print(f"Warning: Could not generate forest comparison plots: {e}")
            
        else:
            print("No isolation forest outlier columns found. Skipping imputation.")
            
        return
    
    # Automatically infer best dtypes for each column
    df = df.convert_dtypes()

    # Impute IQR outliers
    iqr_df = impute_iqr_outliers(df)

    # Impute Z-score outliers
    z_df = impute_zscore_outliers(df)

    # Process based on what we have
    if not iqr_df.empty or not z_df.empty:
        # Create output df
        result_df = df.copy()
        
        # Add IQR imputed columns if available
        if not iqr_df.empty:
            # Ensure the index alignment is the same
            iqr_df = iqr_df.sort_index()
            # Select columns starting with 'iqr_mod_'
            iqr_columns_to_add = [col for col in iqr_df.columns if col.startswith('iqr_mod_')]
            # Merge the selected columns
            for col in iqr_columns_to_add:
                result_df[col] = iqr_df[col]
                
            # Generate IQR comparison boxplots
            print("Generating IQR comparison boxplots...")
            non_iqr_out_cols = [col for col in iqr_df.columns if not col.startswith("iqr_out_")]
            try:
                iqr_figures = plot_multi_boxplot_comparisons(
                    df=iqr_df, 
                    original_cols=non_iqr_out_cols, 
                    modified_cols_prefix='iqr_mod_',
                    num_cols=3, 
                    plots_per_figure=3, 
                    save_dir='visualizations/outliers/boxplots', 
                    dpi=300, 
                    fig_name_prefix='comparison_boxplots_iqr_handling'
                )
            except Exception as e:
                print(f"Warning: Could not generate IQR comparison plots: {e}")
            
        # Add Z-score imputed columns if available
        if not z_df.empty:
            # Ensure the index alignment is the same
            z_df = z_df.sort_index()
            # Select columns starting with 'z_mod_'
            z_columns_to_add = [col for col in z_df.columns if col.startswith('z_mod_')]
            # Merge the selected columns
            for col in z_columns_to_add:
                result_df[col] = z_df[col]
                
            # Generate Z-score comparison boxplots
            print("Generating Z-score comparison boxplots...")
            non_z_out_cols = [col for col in z_df.columns if not col.startswith('zscore_out_')]
            try:
                z_figures = plot_multi_boxplot_comparisons(
                    df=z_df, 
                    original_cols=non_z_out_cols, 
                    modified_cols_prefix='z_mod_',
                    num_cols=3, 
                    plots_per_figure=3, 
                    save_dir='visualizations/outliers/boxplots', 
                    dpi=300, 
                    fig_name_prefix='comparison_boxplots_z_handling'
                )
            except Exception as e:
                print(f"Warning: Could not generate Z-score comparison plots: {e}")
        
        # Save the resulting DataFrame with imputed values
        file_name = 'outlier_mod_all.csv'
        full_path = location.get_path('data/processed', file_name)
        result_df.to_csv(full_path, index=True)
        print(f"Saved imputed data to: {full_path}")
    else:
        print("No outliers to impute. Skipping imputation step.")


if __name__ == "__main__":
    # Initialize location
    base_dir = os.getcwd()    
    location = Location(base_dir)
    
    # Process and impute outliers
    process_and_impute_outliers(location)
    
    print("Outlier imputation completed successfully.")