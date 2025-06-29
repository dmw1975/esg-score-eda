#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering script for ESG Score analysis

This script handles one-hot encoding for categorical data and creates
visualizations of the feature expansion.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location


def create_feature_expansion_table(
    original_df: pd.DataFrame,
    encoded_df: pd.DataFrame,
    location: Location,
    subdirectory: str = "latex",
    file_name: str = "feature_expansion_table.tex"
) -> pd.DataFrame:
    """
    Create a LaTeX table showing the expansion of features after one-hot encoding
    and save it in a specified directory.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original DataFrame with categorical columns
    encoded_df : pd.DataFrame
        The one-hot encoded DataFrame
    location : Location
        Location object for handling file paths
    subdirectory : str, optional
        The subdirectory to save the LaTeX file to, by default "latex"
    file_name : str, optional
        The name of the LaTeX file, by default "feature_expansion_table.tex"

    Returns
    -------
    pd.DataFrame
        DataFrame containing the feature expansion information
    """
    # Initialize a list to store results
    expansion_data = []

    # Iterate through each column in the original dataframe
    for col in original_df.columns:
        if col.endswith('_na'):  # Skip NA indicator columns if they exist
            continue

        # Count unique categories in the original column
        original_categories = original_df[col].nunique()

        # Count encoded columns created from this original column
        encoded_cols = [enc_col for enc_col in encoded_df.columns if enc_col.startswith(f"{col}_")]
        encoded_features = len(encoded_cols)

        # Append the results
        expansion_data.append({
            'Variable': col,
            'Original Categories': original_categories,
            'Encoded Features': encoded_features
        })

    # Convert collected data into a DataFrame
    expansion_df = pd.DataFrame(expansion_data)

    # Add a Total row
    total_row = pd.DataFrame([{
        'Variable': 'Total',
        'Original Categories': expansion_df['Original Categories'].sum(),
        'Encoded Features': expansion_df['Encoded Features'].sum()
    }])

    expansion_df = pd.concat([expansion_df, total_row], ignore_index=True)

    # Convert the DataFrame to LaTeX format
    latex_code = expansion_df.to_latex(
        index=False,
        column_format='|l|r|r|',
        caption='Feature Space Expansion through One-Hot Encoding',
        label='tab:encoding_expansion',
        escape=False
    )

    # Define the output path
    output_path = location.get_path(subdirectory, file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the LaTeX code to a file
    with open(output_path, "w") as file:
        file.write(latex_code)

    return expansion_df


def create_visualizations(
    expansion_df: pd.DataFrame,
    subdirectory: str = "graphs/one_hot"
) -> pd.DataFrame:
    """
    Create visualizations of feature expansion and save them in the specified directory.

    Parameters
    ----------
    expansion_df : pd.DataFrame
        DataFrame containing the feature expansion information
    subdirectory : str, optional
        Directory to save the visualizations, by default "graphs/one_hot"

    Returns
    -------
    pd.DataFrame
        The top N variables by encoded features
    """
    # Create the directory if it doesn't exist
    os.makedirs(subdirectory, exist_ok=True)

    # Make a copy to avoid modifying the original
    df = expansion_df.copy()

    # Replace NaN values with 0
    df = df.fillna(0)

    # Select top N variables by encoded features for better visualization
    if len(df) <= 1:  # If only the Total row exists
        print("Not enough data for visualization")
        return df

    top_n = min(10, len(df) - 1)  # Use at most 10 variables or all if less
    top_vars = df.iloc[:-1].nlargest(top_n, 'Encoded Features')

    # Bar Chart - Original vs Encoded Features
    plt.figure(figsize=(12, 8))
    x = np.arange(len(top_vars))
    width = 0.35

    bar1 = plt.bar(x - width/2, top_vars['Original Categories'], width, label='Original Categories', color='#3498db')
    bar2 = plt.bar(x + width/2, top_vars['Encoded Features'], width, label='Encoded Features', color='#e74c3c')

    plt.xlabel('Categorical Variables', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Top Variables by Feature Expansion', fontsize=14)
    plt.xticks(x, top_vars['Variable'], rotation=45, ha='right', fontsize=10)
    plt.legend()

    # Add labels
    for bar in bar1 + bar2:
        plt.annotate(f'{int(bar.get_height())}', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    bar_chart_path = os.path.join(subdirectory, 'feature_expansion_top.png')
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved at: {bar_chart_path}")

    # Expansion Ratio Bar Chart
    expansion_ratio = top_vars.copy()
    mask = expansion_ratio['Original Categories'] > 0
    expansion_ratio.loc[mask, 'Expansion Ratio'] = (
        expansion_ratio.loc[mask, 'Encoded Features'] / 
        expansion_ratio.loc[mask, 'Original Categories']
    )
    expansion_ratio.loc[~mask, 'Expansion Ratio'] = 0

    plt.figure(figsize=(10, 8))
    bars = plt.barh(expansion_ratio['Variable'], expansion_ratio['Expansion Ratio'], color='#2ecc71')

    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', ha='left', va='center')

    plt.title('Expansion Ratio by Categorical Feature', fontsize=14)
    plt.xlabel('Ratio (Encoded Features / Original Categories)', fontsize=12)
    plt.ylabel('Categorical Variable', fontsize=12)
    plt.tight_layout()
    expansion_ratio_path = os.path.join(subdirectory, 'expansion_ratio_top.png')
    plt.savefig(expansion_ratio_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Expansion ratio chart saved at: {expansion_ratio_path}")

    return top_vars


def process_categorical_features(location: Location) -> None:
    """
    Process categorical features by applying one-hot encoding
    and creating visualizations

    Parameters
    ----------
    location : Location
        Location object for handling file paths
    """
    # Import categorical data
    subdirectory = 'data/processed'
    file_name = 'categorical.csv'
    full_path = location.get_path(subdirectory, file_name)
    df = pd.read_csv(full_path, index_col='issuer_name')

    # Drop the columns that aren't needed
    df = df.drop(columns=["isin", "esg_rating", "issuer_name.1"])

    # Apply one-hot encoding to the DataFrame
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Convert encoded columns to float32 for memory efficiency
    df_encoded = df_encoded.astype(np.float32)

    # Create feature expansion table
    expansion_table = create_feature_expansion_table(
        df, 
        df_encoded, 
        location, 
        subdirectory=subdirectory, 
        file_name="feature_expansion_table.tex"
    )

    # Create visualizations
    create_visualizations(expansion_table, subdirectory="visualizations/one_hot")

    # Save encoded data
    file_name = 'categorical_encoded.csv'
    full_path = location.get_path('data/processed', file_name)
    df_encoded.to_csv(full_path, index=True)
    print(f"Encoded categorical data saved to: {full_path}")


if __name__ == "__main__":
    # Initialize location
    base_dir = os.getcwd()    
    location = Location(base_dir)
    
    # Process categorical features
    process_categorical_features(location)
    
    print("Feature engineering completed successfully.")