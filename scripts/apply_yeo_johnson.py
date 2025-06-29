#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Yeo-Johnson Transformation

This script applies the Yeo-Johnson transformation to stabilize variance
and make non-normal data more Gaussian-like. The transformation is applied
to numeric columns in the dataset and the transformed data is saved for
use in future modeling.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import pickle

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location

def gaussian(x, mean, std_dev):
    """
    Calculate the Gaussian (Normal) probability density function (PDF).

    Parameters:
    x (array-like): Values to evaluate the PDF.
    mean (float): Mean of the distribution.
    std_dev (float): Standard deviation of the distribution.

    Returns:
    array-like: Gaussian PDF values.
    """
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def plot_gaussian_distribution(data, output_path, dataset_name="Dataset"):
    """
    Plot the histogram of data and overlay a Gaussian distribution curve.

    Parameters:
    data (array-like): Input data to visualize.
    output_path (str): Path to save the figure.
    dataset_name (str): Name of the dataset for the plot title.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    median = np.median(data)

    # Create a range of x values for the Gaussian curve
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = gaussian(x, mean, std_dev)

    # Plot the histogram of the data
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')

    # Plot the Gaussian curve
    plt.plot(x, y, label='Gaussian Distribution', color='r', lw=2)

    # Mark statistical values
    plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(mean - std_dev, color='purple', linestyle=':', label=f'-1 Std Dev: {mean - std_dev:.2f}')
    plt.axvline(mean + std_dev, color='purple', linestyle=':', label=f'+1 Std Dev: {mean + std_dev:.2f}')
    plt.axvline(median, color='orange', linestyle='-.', label=f'Median: {median:.2f}')

    # Add labels, title, and legend
    plt.title(f'Distribution of {dataset_name} with Gaussian Fit')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.4)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved at: {output_path}")
    plt.close()

def plot_multiple_variables(df, variables, output_path):
    """
    Plot histograms and Gaussian distribution curves for multiple variables.

    Parameters:
    df (DataFrame): DataFrame containing the variables to plot.
    variables (list): List of column names to plot.
    output_path (str): Path to save the output plot.
    """
    num_vars = len(variables)
    rows = (num_vars + 1) // 2  # Number of rows, assuming 2 columns per row
    cols = 2

    plt.figure(figsize=(16, 4 * rows))

    for i, var in enumerate(variables):
        ax = plt.subplot(rows, cols, i + 1)
        data = df[var]
        mean = np.mean(data)
        std_dev = np.std(data)
        median = np.median(data)

        # Create a range of x values for the Gaussian curve
        x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
        y = gaussian(x, mean, std_dev)

        # Plot histogram and Gaussian curve
        ax.hist(data, bins=30, color='blue', edgecolor='black', density=True, alpha=0.6)
        ax.plot(x, y, label='Gaussian Fit', color='r', lw=2)

        # Mark statistical values
        ax.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='orange', linestyle='-.', label=f'Median: {median:.2f}')
        ax.axvline(mean - std_dev, color='purple', linestyle=':', label=f'-1 Std Dev: {mean - std_dev:.2f}')
        ax.axvline(mean + std_dev, color='purple', linestyle=':', label=f'+1 Std Dev: {mean + std_dev:.2f}')

        # Set labels and title
        ax.set_title(var.replace('_', ' ').title())
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved at: {output_path}")
    plt.close()

def save_list_to_pickle(data_list, file_path):
    """
    Save a list to a file using pickle.
    
    Args:
    data_list (list): The list to save.
    file_path (str): The path to save the list to.
    """
    with open(file_path, "wb") as file:
        pickle.dump(data_list, file)
    print(f"List saved to: {file_path}")

def interleave_with_yeo_joh(variables):
    """
    Interleave original variables with their Yeo-Johnson transformed counterparts.
    
    Args:
    variables (list): List of variable names.
    
    Returns:
    list: Interleaved list of variables.
    """
    # Filter out the 'yeo_joh_' variables
    yeo_joh_vars = [var for var in variables if var.startswith("yeo_joh_")]
    
    # Get the base variables (excluding 'yeo_joh_' prefixed ones)
    base_vars = [var for var in variables if not var.startswith("yeo_joh_")]
    
    # Create a dictionary to map base variables to their 'yeo_joh_' equivalents
    yeo_joh_map = {var.replace("yeo_joh_", ""): var for var in yeo_joh_vars}
    
    # Interleave the variables
    interleaved_list = []
    for var in base_vars:
        interleaved_list.append(var)
        if var in yeo_joh_map:
            interleaved_list.append(yeo_joh_map[var])

    return interleaved_list

def apply_yeo_johnson_transformation(input_file, location):
    """
    Apply Yeo-Johnson transformation to numeric columns.
    
    Args:
    input_file (str): Path to the input file.
    location (Location): Location object for file paths.
    """
    print("Loading data...")
    df = pd.read_csv(input_file, index_col='issuer_name')
    
    # Step 1: Extract ESG score if it exists
    print("Extracting ESG score...")
    try:
        score = df['esg_score']
        score_path = location.get_path('data/processed', 'score.csv')
        score.to_csv(score_path, index=True)
        print(f"ESG score saved to: {score_path}")
    except KeyError:
        print("ESG score column not found in the dataset.")
    
    # Step 2: Create lists of columns to exclude
    print("Preparing data for transformation...")
    # Find all columns with specific prefixes to exclude
    exclude_prefixes = ['iqr_out_', 'zscore_out_', 'iqr_mod_', 'z_mod_', 'yeo_joh_']
    forest_columns = ['forest_outlier', 'forest_outlier_tuned', 'anomaly_score', 'predicted_outlier']
    esg_columns = ['esg_score', 'z_mod_esg_score', 'iqr_mod_esg_score', 'zscore_out_esg_score', 'iqr_out_esg_score']
    
    # Combine all columns to exclude
    drop_columns = forest_columns + esg_columns
    for prefix in exclude_prefixes:
        drop_columns.extend([col for col in df.columns if col.startswith(prefix)])
    drop_columns = list(set(drop_columns))  # Remove duplicates
    
    # Create a copy of the dataframe without the columns to exclude
    filtered_df = df.drop(drop_columns, axis=1, errors='ignore')
    print(f"Processing {len(filtered_df.columns)} columns after filtering.")
    
    # Identify numeric columns for transformation
    numeric_columns = []
    for col in filtered_df.columns:
        try:
            # Convert to standard float if pandas extension type
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
            
            # If column contains mostly numeric data, keep it
            if filtered_df[col].notna().sum() > 0.5 * len(filtered_df):
                numeric_columns.append(col)
            else:
                print(f"Skipping column {col} - too many NaN values after conversion")
        except (ValueError, TypeError):
            print(f"Skipping non-numeric column: {col}")
            continue
    
    # Create a numeric-only dataframe
    numeric_df = filtered_df[numeric_columns].copy()
    print(f"Found {len(numeric_columns)} numeric columns for transformation.")
    
    # Initialize the PowerTransformer
    print("Applying Yeo-Johnson transformation...")
    pt = PowerTransformer(method='yeo-johnson')
    
    # Apply Yeo-Johnson transformation to each numeric column
    transformed_columns = []
    successful_pairs = []  # Track successful transformations for visualization
    
    for column in numeric_columns:
        try:
            # Fill NaN values with the median for transformation
            data = numeric_df[column].fillna(numeric_df[column].median()).values.reshape(-1, 1)
            
            # Transform the data and flatten the result
            transformed_data = pt.fit_transform(data).flatten()
            
            # Add the transformed column to the DataFrame with a new name
            transformed_column = f"yeo_joh_{column}"
            numeric_df[transformed_column] = transformed_data
            
            # Keep track of transformed columns
            transformed_columns.append(transformed_column)
            successful_pairs.append((column, transformed_column))
            print(f"Successfully transformed column: {column}")
        except Exception as e:
            print(f"Error transforming column {column}: {e}")
            continue
    
    # Create visualization directory if it doesn't exist
    vis_dir = location.get_path('visualizations', 'yeo_johnson')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create pickle directory if it doesn't exist
    pkl_dir = location.get_path('data/processed', 'pkl')
    os.makedirs(pkl_dir, exist_ok=True)
    
    # Plot original vs transformed distributions for selected variables
    print("Creating visualizations...")
    
    # Plot ESG score distribution if available
    try:
        if 'score' in locals() and not score.empty:
            plot_gaussian_distribution(
                score, 
                os.path.join(vis_dir, 'score.png'),
                dataset_name="original ESG score"
            )
    except:
        print("Could not generate ESG score visualization.")
    
    # Select a subset of successful pairs for visualization (up to 5 pairs)
    if successful_pairs:
        print("Creating comparison visualizations...")
        visualization_vars = []
        # Select up to 5 pairs of original/transformed columns
        sample_pairs = successful_pairs[:min(5, len(successful_pairs))]
        
        for orig, transformed in sample_pairs:
            visualization_vars.append(orig)
            visualization_vars.append(transformed)
        
        # Make sure we have the columns
        vis_columns = [col for col in visualization_vars if col in numeric_df.columns]
        if vis_columns:
            plot_multiple_variables(numeric_df, vis_columns, os.path.join(vis_dir, 'yeo_johnson_comparison.png'))
        else:
            print("No valid columns found for visualization.")
    else:
        print("No successful transformations to visualize.")
    
    # Optional: Add categorical data if available
    try:
        cat_file = location.get_path('data/processed', 'categorical_encoded.csv')
        cat_df = pd.read_csv(cat_file, index_col='issuer_name')
        cat_df = cat_df.convert_dtypes()
        cat_df.sort_index(ascending=True, inplace=True)
        
        # Prepare numerical dataframe
        numeric_df.sort_index(ascending=True, inplace=True)
        
        # Combine numerical and categorical data
        combined_df = pd.merge(numeric_df, cat_df, left_index=True, right_index=True, how='inner')
        combined_df = combined_df.convert_dtypes()
        
        # Save combined dataframe
        combined_path = location.get_path('data/processed', 'combined_yeo_johnson.csv')
        combined_df.to_csv(combined_path, index=True)
        print(f"Combined data saved to: {combined_path}")
        
        # Create and save column lists for future use
        print("Creating column lists for future reference...")
        
        # Get list of categorical columns
        cat_columns = cat_df.columns.tolist()
        
        # Define list of base columns (original numeric columns)
        base_columns = numeric_df.columns.tolist()
        base_columns = [col for col in base_columns if not col.startswith('yeo_joh_')]
        save_list_to_pickle(base_columns, os.path.join(pkl_dir, 'base_columns.pkl'))
        
        # Define Yeo-Johnson columns
        yeo_columns = transformed_columns
        yeo_base_columns = yeo_columns + cat_columns
        save_list_to_pickle(yeo_base_columns, os.path.join(pkl_dir, 'yeo_columns.pkl'))
        
    except FileNotFoundError:
        print("Categorical data not found. Saving only numerical data.")
        # Save only the numerical data with transformations
        numeric_path = location.get_path('data/processed', 'numerical_yeo_johnson.csv')
        numeric_df.to_csv(numeric_path, index=True)
        print(f"Numerical data saved to: {numeric_path}")
    
    # Save original plus transformed columns
    all_columns_path = location.get_path('data/processed', 'all_yeo_johnson.csv')
    numeric_df.to_csv(all_columns_path, index=True)
    print(f"All columns with Yeo-Johnson transformations saved to: {all_columns_path}")
    
    print("Yeo-Johnson transformation completed successfully.")
    return numeric_df

def main():
    # Initialize location
    base_dir = os.getcwd()
    location = Location(base_dir)
    
    # Define potential input files in order of preference
    potential_files = [
        'data/processed/outlier_all.csv',
        'data/processed/isolation_forest_outliers.csv',
        'data/processed/imputed_data.csv',
        'data/processed/numerical.csv',
        'data/processed/categorical.csv'
    ]
    
    # Try to find an existing input file from the list
    input_file = None
    for file_path in potential_files:
        try:
            potential_file = location.get_path('', file_path)
            if os.path.exists(potential_file):
                input_file = potential_file
                break
        except:
            continue
    
    # If no file was found, look for any CSV in the data directories
    if input_file is None:
        print("Preferred input files not found. Searching for alternatives...")
        for data_dir in ['data/processed', 'data/interim', 'data/raw']:
            try:
                dir_path = location.get_path('', data_dir)
                if os.path.exists(dir_path):
                    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                    if csv_files:
                        input_file = os.path.join(dir_path, csv_files[0])
                        print(f"Found alternative input file: {input_file}")
                        break
            except:
                continue
    
    # If still no file found, raise an error
    if input_file is None:
        raise FileNotFoundError("No suitable input files found in the data directories.")
    
    print(f"Using input file: {input_file}")
    
    # Apply Yeo-Johnson transformation
    apply_yeo_johnson_transformation(input_file, location)

if __name__ == "__main__":
    main()