#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update Yeo-Johnson Visualizations

This script creates additional visualizations for the Yeo-Johnson transformation
results, including a plot of the transformed ESG score and an updated comparison plot.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

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

def create_esg_score_visualizations(location):
    """
    Create visualizations for ESG scores with Yeo-Johnson transformation.
    
    Parameters:
    location (Location): Location object for handling file paths
    """
    print("Creating ESG score visualizations...")
    
    # Load the ESG score data
    score_path = location.get_path('data/processed', 'score.csv')
    try:
        score_df = pd.read_csv(score_path, index_col='issuer_name')
        score = score_df['esg_score']
        print(f"Loaded ESG score data from {score_path}")
    except FileNotFoundError:
        print(f"ESG score file not found at {score_path}. Trying to extract from outlier data...")
        
        # Try to get ESG score from outlier data
        outlier_path = location.get_path('data/processed', 'outlier_all.csv')
        try:
            outlier_df = pd.read_csv(outlier_path, index_col='issuer_name')
            score = outlier_df['esg_score']
            print(f"Extracted ESG score from {outlier_path}")
        except (FileNotFoundError, KeyError):
            print("Could not find ESG score data. Exiting.")
            return
    
    # Create visualization directory if it doesn't exist
    vis_dir = location.get_path('visualizations', 'yeo_johnson')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Apply Yeo-Johnson transformation to ESG score
    print("Applying Yeo-Johnson transformation to ESG score...")
    pt = PowerTransformer(method='yeo-johnson')
    transformed_score = pt.fit_transform(score.values.reshape(-1, 1)).flatten()
    
    # Create and save the visualization for original ESG score
    plot_gaussian_distribution(
        score,
        os.path.join(vis_dir, 'esg_score_original.png'),
        dataset_name="Original ESG Score"
    )
    
    # Create and save the visualization for Yeo-Johnson transformed ESG score
    plot_gaussian_distribution(
        transformed_score,
        os.path.join(vis_dir, 'esg_score_yeo_johnson.png'),
        dataset_name="Yeo-Johnson Transformed ESG Score"
    )
    
    # Create a visual comparison of original vs transformed ESG scores
    plt.figure(figsize=(12, 6))
    
    # Create subplots
    plt.subplot(1, 2, 1)
    plt.hist(score, bins=30, color='green', edgecolor='black', density=True, alpha=0.6)
    mean = np.mean(score)
    std_dev = np.std(score)
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = gaussian(x, mean, std_dev)
    plt.plot(x, y, color='red', lw=2)
    plt.axvline(mean, color='blue', linestyle='--')
    plt.title("Original ESG Score")
    plt.xlabel("Value")
    plt.ylabel("Density")
    
    plt.subplot(1, 2, 2)
    plt.hist(transformed_score, bins=30, color='blue', edgecolor='black', density=True, alpha=0.6)
    mean = np.mean(transformed_score)
    std_dev = np.std(transformed_score)
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = gaussian(x, mean, std_dev)
    plt.plot(x, y, color='red', lw=2)
    plt.axvline(mean, color='blue', linestyle='--')
    plt.title("Yeo-Johnson Transformed ESG Score")
    plt.xlabel("Value")
    plt.ylabel("Density")
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'esg_score_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"ESG score comparison saved to: {os.path.join(vis_dir, 'esg_score_comparison.png')}")
    plt.close()
    
    print("ESG score visualizations created successfully.")

def update_comparison_visualization(location):
    """
    Update the comparison visualization to show hist_capex_sales instead of net_income_usd
    and remove hist_fcf_yld.
    
    Parameters:
    location (Location): Location object for handling file paths
    """
    print("Updating comparison visualization...")
    
    # Load the numerical data with Yeo-Johnson transformations
    all_yeo_path = location.get_path('data/processed', 'all_yeo_johnson.csv')
    try:
        df = pd.read_csv(all_yeo_path, index_col='issuer_name')
        print(f"Loaded numerical data from {all_yeo_path}")
    except FileNotFoundError:
        print(f"Numerical data file not found at {all_yeo_path}. Trying to extract from combined data...")
        
        # Try to get data from combined file
        combined_path = location.get_path('data/processed', 'combined_yeo_johnson.csv')
        try:
            df = pd.read_csv(combined_path, index_col='issuer_name')
            print(f"Extracted numerical data from {combined_path}")
        except FileNotFoundError:
            print("Could not find numerical data with Yeo-Johnson transformations. Exiting.")
            return
    
    # Create visualization directory if it doesn't exist
    vis_dir = location.get_path('visualizations', 'yeo_johnson')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Define the variables to visualize (per requirements)
    # Replace net_income_usd with hist_capex_sales and remove hist_fcf_yld
    variables = [
        'market_cap_usd', 'yeo_joh_market_cap_usd',
        'hist_pe', 'yeo_joh_hist_pe',
        'hist_book_px', 'yeo_joh_hist_book_px',
        'hist_capex_sales', 'yeo_joh_hist_capex_sales',  # Added instead of net_income_usd
        'shares_outstanding', 'yeo_joh_shares_outstanding'
    ]
    
    # Create and save the updated comparison visualization
    plot_multiple_variables(
        df, 
        variables, 
        os.path.join(vis_dir, 'yeo_johnson_comparison.png')
    )
    
    print("Comparison visualization updated successfully.")

def main():
    """Main execution function."""
    # Initialize location
    base_dir = os.getcwd()
    location = Location(base_dir)
    
    # Create ESG score visualizations
    create_esg_score_visualizations(location)
    
    # Update comparison visualization
    update_comparison_visualization(location)
    
    print("Yeo-Johnson visualization updates completed successfully.")

if __name__ == "__main__":
    main()