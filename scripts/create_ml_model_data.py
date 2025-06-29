#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create ML Model Data script for ESG Score analysis

This script creates the combined_df_for_ml_models.csv file needed by the notebooks.
"""

import os
import sys
import pandas as pd

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location

def create_ml_model_data(location):
    """
    Create the ML model data file by combining the yeo_johnson numerical data
    with one-hot encoded categorical data.
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
    """
    print("Creating ML model data...")
    
    # Load the Yeo-Johnson transformed numerical data
    try:
        # First try all_yeo_johnson.csv which should have all numerical columns
        numerical_path = location.get_path('data/processed', 'all_yeo_johnson.csv')
        numerical_df = pd.read_csv(numerical_path, index_col='issuer_name')
        print(f"Loaded numerical Yeo-Johnson data from {numerical_path}")
        
        # Keep only the Yeo-Johnson transformed columns (with yeo_joh_ prefix)
        yeo_cols = [col for col in numerical_df.columns if col.startswith('yeo_joh_')]
        numerical_df = numerical_df[yeo_cols]
        print(f"Using {len(yeo_cols)} Yeo-Johnson transformed numerical features")
        
    except FileNotFoundError:
        print("Error: Could not find Yeo-Johnson transformed data (all_yeo_johnson.csv)")
        return
    
    # Load the one-hot encoded categorical data
    try:
        categorical_path = location.get_path('data/processed', 'categorical_encoded.csv')
        categorical_df = pd.read_csv(categorical_path, index_col='issuer_name')
        print(f"Loaded one-hot encoded categorical data from {categorical_path}")
        print(f"Found {len(categorical_df.columns)} one-hot encoded features")
        
    except FileNotFoundError:
        print("Error: Could not find one-hot encoded categorical data (categorical_encoded.csv)")
        print("Please run the feature engineering step first.")
        return
    
    # Ensure indices are aligned
    numerical_df.sort_index(ascending=True, inplace=True)
    categorical_df.sort_index(ascending=True, inplace=True)
    
    # Combine numerical and categorical data
    combined_df = pd.merge(
        numerical_df, 
        categorical_df, 
        left_index=True, 
        right_index=True, 
        how='inner'
    )
    
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Total features: {len(yeo_cols)} numerical + {len(categorical_df.columns)} categorical = {len(combined_df.columns)}")
    
    # Save the combined data as combined_df_for_ml_models.csv
    output_path = location.get_path('data', 'combined_df_for_ml_models.csv')
    combined_df.to_csv(output_path, index=True)
    print(f"ML model data saved to: {output_path}")

if __name__ == "__main__":
    # Initialize location
    base_dir = os.getcwd()    
    location = Location(base_dir)
    
    # Create ML model data
    create_ml_model_data(location)
    
    print("ML model data creation completed successfully.")