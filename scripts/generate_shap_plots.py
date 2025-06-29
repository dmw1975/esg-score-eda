#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate SHAP (SHapley Additive exPlanations) visualizations for Isolation Forest

This script loads the Isolation Forest model results and generates SHAP visualizations
to explain which features are most important for outlier detection.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12  # Increase default font size for better readability

# Import the Location class from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location
from scripts.shap_compatibility import patch_numpy_for_shap

def main():
    print("Generating SHAP visualizations for Isolation Forest model...")
    
    # Patch NumPy for SHAP compatibility if needed
    patched = patch_numpy_for_shap()
    print(f"NumPy patched for SHAP compatibility: {patched}")
    
    # Now import SHAP (after potential patching)
    try:
        import shap
    except ImportError:
        print("Error: SHAP library not installed. Please install it with 'pip install shap'")
        return 1
    
    # Set up directories
    base_dir = os.getcwd()
    location = Location(base_dir)
    
    # Create the output directory for SHAP plots
    shap_dir = os.path.join(base_dir, "visualizations", "outliers", "shap")
    os.makedirs(shap_dir, exist_ok=True)
    
    # Load the data
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
    
    # Exclude outlier flag columns and the target variable (esg_score)
    features = [col for col in numeric_cols if not (
        col.startswith('zscore_out_') or 
        col.startswith('iqr_out_') or 
        col.startswith('forest_') or 
        col.startswith('anomaly_score') or
        col == 'esg_score'  # Exclude esg_score as it's the target variable
    )]
    
    print(f"Using {len(features)} numeric features for SHAP analysis")
    
    # Extract the features
    X = df[features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check if a previously trained model exists
    model_path = location.get_path("visualizations/outliers/isolation", "isolation_forest_model.pkl")
    
    if os.path.exists(model_path):
        print(f"Loading existing Isolation Forest model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("Creating new Isolation Forest model")
        # Create and fit a new Isolation Forest model
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        model.fit(X_scaled)
        
        # Save the model for future use
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to {model_path}")
    
    print("Creating SHAP explainer for the Isolation Forest model")
    try:
        # Create the SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_scaled)
        
        # Create and save the summary dot plot
        print("Generating summary dot plot...")
        plt.figure(figsize=(14, 12))  # Larger figure size
        shap.summary_plot(
            shap_values, 
            X_scaled,
            feature_names=features,
            show=False
        )
        plt.title('SHAP Summary: Feature Impact on Outlier Detection', fontsize=16, fontweight='bold')  # Add title
        plt.xlabel('SHAP value (impact on model output)', fontsize=14)  # Larger x-axis label
        # Make y-axis feature names larger
        plt.yticks(fontsize=12)  
        summary_dot_path = os.path.join(shap_dir, "shap_summary_dot.png")
        plt.tight_layout()
        plt.savefig(summary_dot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary dot plot saved to {summary_dot_path}")
        
        # Create and save the summary bar plot
        print("Generating summary bar plot...")
        plt.figure(figsize=(14, 12))  # Larger figure size
        shap.summary_plot(
            shap_values, 
            X_scaled,
            feature_names=features,
            plot_type="bar",
            show=False
        )
        plt.title('SHAP Feature Importance for Outlier Detection', fontsize=16, fontweight='bold')  # Add title
        plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=14)  # Larger x-axis label
        # Make y-axis feature names larger
        plt.yticks(fontsize=12)  
        summary_bar_path = os.path.join(shap_dir, "shap_summary_bar.png")
        plt.tight_layout()
        plt.savefig(summary_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary bar plot saved to {summary_bar_path}")
        
        # Create and save the decision plot if the dataset is small enough
        if X.shape[0] <= 1000:
            print("Generating decision plot...")
            plt.figure(figsize=(12, 12))
            shap.decision_plot(
                explainer.expected_value, 
                shap_values, 
                feature_names=features,
                show=False
            )
            decision_path = os.path.join(shap_dir, "shap_decision_plot.png")
            plt.tight_layout()
            plt.savefig(decision_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Decision plot saved to {decision_path}")
        else:
            print("Skipping decision plot due to large dataset size")
        
        print("SHAP visualizations completed successfully")
        return 0
        
    except Exception as e:
        print(f"Error generating SHAP visualizations: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())