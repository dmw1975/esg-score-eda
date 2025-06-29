#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Isolation Forest visualizations

This script generates isolation forest outlier detection visualizations
and saves them to the visualizations/outliers/isolation directory.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12  # Increase default font size

# Import the Location class from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location

def main():
    print("Generating Isolation Forest visualizations...")
    
    # Set up directories
    base_dir = os.getcwd()
    location = Location(base_dir)
    
    # Create the output directory
    os.makedirs(os.path.join(base_dir, "visualizations", "outliers", "isolation"), exist_ok=True)
    
    # Load the data
    # First try to find the outlier data file
    try:
        input_file = location.get_path("data/processed", "outliers_iqr_z.csv")
        df = pd.read_csv(input_file, index_col='issuer_name')
        print(f"Using data file: {input_file}")
    except FileNotFoundError:
        # Fall back to imputed data
        input_file = location.get_path("data/processed", "imputed_data.csv")
        df = pd.read_csv(input_file, index_col='issuer_name')
        print(f"Falling back to data file: {input_file}")
    
    # Filter to only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude IQR and Z-score columns and esg_score
    numeric_cols = [col for col in numeric_cols if not (col.startswith('zscore_out_') or col.startswith('iqr_out_') or col == 'esg_score')]
    print(f"Using {len(numeric_cols)} numeric columns for analysis (excluding esg_score)")
    
    # Create a clean dataframe with only numeric features
    data = df[numeric_cols].copy()
    
    # Load best parameters
    try:
        with open(os.path.join(base_dir, 'best_params_out_1.json'), 'r') as file:
            loaded_params = json.load(file)
    except FileNotFoundError:
        # Use default parameters if file not found
        loaded_params = {
            'contamination': 0.01,
            'n_estimators': 69,
            'random_state': 42
        }
        print("Warning: best_params_out_1.json not found, using default parameters")
    
    print(f"Using parameters: {loaded_params}")
    
    # Configure tuned Isolation Forest
    tuned_iso_forest = IsolationForest(**loaded_params)
    
    # Configure standard Isolation Forest
    standard_iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    
    # Fit both models and predict
    tuned_predictions = tuned_iso_forest.fit_predict(data)
    standard_predictions = standard_iso_forest.fit_predict(data)
    
    # Calculate anomaly scores
    tuned_scores = tuned_iso_forest.decision_function(data)
    
    # Create results dataframe
    results = pd.DataFrame({
        'tuned_outlier': tuned_predictions,
        'standard_outlier': standard_predictions,
        'anomaly_score': tuned_scores
    }, index=data.index)
    
    # Identify outliers
    outliers = results[results['tuned_outlier'] == -1]
    inliers = results[results['tuned_outlier'] == 1]
    
    # Get indices of outliers for both models
    std_outliers = list(results[results['standard_outlier'] == -1].index)
    tuned_outliers = list(results[results['tuned_outlier'] == -1].index)
    
    # Identify exclusive outliers
    exclusive_outliers = set(std_outliers) - set(tuned_outliers)
    
    # Get all unique indices
    all_indices = sorted(set(std_outliers + tuned_outliers))
    
    # Create comparison plot
    print("Creating outlier comparison plot...")
    plt.figure(figsize=(10, 16))
    
    # Scatter plot for standard outliers
    plt.scatter([1] * len(std_outliers), std_outliers, color='red', 
                label='Standard Isolation Forest (5%)', alpha=0.6, s=50)
    
    # Scatter plot for tuned outliers
    plt.scatter([2] * len(tuned_outliers), tuned_outliers, color='blue', 
                label='Tuned Isolation Forest (1%)', alpha=0.6, s=50)
    
    # Highlight exclusive outliers
    plt.scatter([1] * len(exclusive_outliers), list(exclusive_outliers), color='green', 
                label='Only in Standard', alpha=0.8, s=50)
    
    # Add labels and legend
    plt.xticks([1, 2], ['Standard (5%)', 'Tuned (1%)'], fontsize=12)
    plt.xlabel('Outlier Detection Method', fontsize=14)
    plt.ylabel('Company Name', fontsize=14)
    plt.title('Comparison of Outliers: Standard vs Tuned Isolation Forest', fontsize=16, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)
    plt.grid(True)
    
    # Show all y-axis labels
    plt.yticks(all_indices, all_indices, fontsize=8)  # Slightly larger font size for readability
    
    # Add padding
    plt.margins(0.05)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    output_path = location.get_path('visualizations/outliers/isolation', 'outlier_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()
    
    # Create anomaly score histogram
    print("Creating anomaly score histogram...")
    plt.figure(figsize=(12, 6))
    plt.hist(inliers['anomaly_score'], bins=30, color='green', alpha=0.7, label="Inlier Score Distribution")
    plt.hist(outliers['anomaly_score'], bins=30, color='blue', alpha=0.7, label="Outlier Score Distribution")
    plt.axvline(outliers['anomaly_score'].mean(), color='red', linestyle='--', label="Outlier Mean Score")
    plt.title("Distribution of Anomaly Scores", fontsize=18, fontweight='bold')  # Larger, bold title
    plt.xlabel("Anomaly Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=12)  # Larger axis tick labels
    plt.yticks(fontsize=12)  # Larger axis tick labels
    plt.legend(fontsize=12)  # Larger legend text
    plt.grid(True)
    
    # Save the histogram
    output_path = location.get_path('visualizations/outliers/isolation', 'anomaly_score_distribution.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved to {output_path}")
    plt.close()
    
    # Create feature importance plot
    print("Creating feature importance visualization...")
    
    # Get feature importances if available
    try:
        importances = tuned_iso_forest.feature_importances_
        
        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        importance_df = pd.DataFrame({
            'Feature': data.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Limit to top 20 features
        top_n = min(20, len(importance_df))
        top_features = importance_df.head(top_n)
        
        # Plot
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        plt.title('Feature Importance in Isolation Forest Model', fontsize=18, fontweight='bold')  # Larger, bold title
        plt.xlabel('Importance Score', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.yticks(fontsize=12)  # Larger feature names on y-axis
        plt.xticks(fontsize=12)  # Larger values on x-axis
        plt.tight_layout()
        
        # Save the feature importance plot
        output_path = location.get_path('visualizations/outliers/isolation', 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    except:
        print("Warning: Feature importances not available in this scikit-learn version")
    
    # Export outlier list with anomaly scores as CSV
    outlier_details = pd.concat([outliers, data.loc[outliers.index]], axis=1).sort_values('anomaly_score')
    output_path = location.get_path('visualizations/outliers/isolation', 'isolation_forest_outliers.csv')
    outlier_details.to_csv(output_path)
    print(f"Outlier list exported to {output_path}")
    
    print("Isolation Forest visualizations generation complete.")
    
if __name__ == "__main__":
    main()