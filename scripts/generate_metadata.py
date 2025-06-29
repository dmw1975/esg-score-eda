#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate metadata JSON files for ML pipeline data loading
"""

import os
import sys
import pandas as pd
import json

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location


def extract_column_information(location):
    """Extract column information from CSV files"""
    
    # Load the data
    linear_df = pd.read_csv(location.get_path('data/ml_output/raw', 'combined_df_for_linear_models.csv'), index_col='issuer_name')
    tree_df = pd.read_csv(location.get_path('data/ml_output/raw', 'combined_df_for_tree_models.csv'), index_col='issuer_name')
    
    # Get all columns
    linear_cols = list(linear_df.columns)
    tree_cols = list(tree_df.columns)
    
    # Define categorical features in tree model
    tree_categorical = [
        'gics_sector', 'gics_sub_ind', 'issuer_cntry_domicile_name', 'cntry_of_risk', 
        'top_1_shareholder_location', 'top_2_shareholder_location', 'top_3_shareholder_location'
    ]
    
    # Extract base numerical features (from tree model, excluding categoricals)
    tree_non_yeo = [col for col in tree_cols if not col.startswith('yeo_joh_')]
    base_numerical = sorted([col for col in tree_non_yeo if col not in tree_categorical])
    
    # Extract Yeo-Johnson features
    yeo_features = sorted([col for col in tree_cols if col.startswith('yeo_joh_')])
    
    # Extract one-hot encoded features from linear model
    one_hot_features = sorted([col for col in linear_cols if col not in base_numerical and col not in yeo_features])
    
    return {
        'base_numerical': base_numerical,
        'yeo_features': yeo_features,
        'one_hot_features': one_hot_features,
        'tree_categorical': tree_categorical,
        'linear_total': len(linear_cols),
        'tree_total': len(tree_cols)
    }


def create_linear_model_columns_json(location, column_info):
    """Create linear_model_columns.json"""
    
    linear_metadata = {
        "base_numerical_features": column_info['base_numerical'],
        "yeo_numerical_features": column_info['yeo_features'],
        "categorical_features": column_info['tree_categorical'],
        "one_hot_encoded_features": column_info['one_hot_features'],
        "total_features": column_info['linear_total'],
        "issuer_identifier": "issuer_name",
        "yeo_prefix": "yeo_joh_",
        "notes": {
            "target": "esg_score is in separate score.csv file",
            "random_feature": "Added dynamically in ml_project_refactored, not in source data",
            "datasets": "Models train on Base, Yeo, Base+Random, and Yeo+Random variants"
        }
    }
    
    output_path = location.get_path('data/ml_output/raw/metadata', 'linear_model_columns.json')
    with open(output_path, 'w') as f:
        json.dump(linear_metadata, f, indent=2)
    
    print(f"Created {output_path}")
    print(f"  Base numerical features: {len(column_info['base_numerical'])}")
    print(f"  Yeo numerical features: {len(column_info['yeo_features'])}")
    print(f"  Categorical features (original): {len(column_info['tree_categorical'])}")
    print(f"  One-hot encoded features: {len(column_info['one_hot_features'])}")
    print(f"  Total features: {column_info['linear_total']}")


def create_tree_model_columns_json(location, column_info):
    """Create tree_model_columns.json"""
    
    tree_metadata = {
        "base_numerical_features": column_info['base_numerical'],
        "yeo_numerical_features": column_info['yeo_features'],
        "categorical_features": column_info['tree_categorical'],
        "total_features": column_info['tree_total'],
        "issuer_identifier": "issuer_name",
        "yeo_prefix": "yeo_joh_",
        "notes": {
            "target": "esg_score is in separate score.csv file",
            "random_feature": "Added dynamically in ml_project_refactored, not in source data",
            "datasets": "Models train on Base, Yeo, Base+Random, and Yeo+Random variants"
        }
    }
    
    output_path = location.get_path('data/ml_output/raw/metadata', 'tree_model_columns.json')
    with open(output_path, 'w') as f:
        json.dump(tree_metadata, f, indent=2)
    
    print(f"\nCreated {output_path}")
    print(f"  Base numerical features: {len(column_info['base_numerical'])}")
    print(f"  Yeo numerical features: {len(column_info['yeo_features'])}")
    print(f"  Categorical features: {len(column_info['tree_categorical'])}")
    print(f"  Total features: {column_info['tree_total']}")


def create_yeo_johnson_mapping_json(location, column_info):
    """Create yeo_johnson_mapping.json"""
    
    # Create mapping from base to Yeo-Johnson columns
    mapping = {}
    for yeo_col in column_info['yeo_features']:
        base_col = yeo_col.replace('yeo_joh_', '')
        if base_col in column_info['base_numerical']:
            mapping[base_col] = yeo_col
    
    output_path = location.get_path('data/ml_output/raw/metadata', 'yeo_johnson_mapping.json')
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nCreated {output_path}")
    print(f"  Mapped {len(mapping)} numerical features to their Yeo-Johnson transformations")


def create_feature_groups_json(location, column_info):
    """Create feature_groups.json"""
    
    # Group features based on their names and meanings
    feature_groups = {
        "financial_metrics": [
            "market_cap_usd", "net_income_usd", "hist_pe", "hist_book_px",
            "hist_fcf_yld", "hist_ebitda_ev", "hist_gross_profit_usd",
            "hist_net_debt_usd", "hist_ev_usd", "hist_eps_usd"
        ],
        "performance_ratios": [
            "hist_roe", "hist_roic", "hist_roa", "hist_asset_turnover"
        ],
        "investment_metrics": [
            "hist_capex_sales", "hist_capex_depr", "hist_rd_exp_usd",
            "return_usd", "vol_total_usd"
        ],
        "governance_features": [
            "top_1_shareholder_percentage", "top_2_shareholder_percentage",
            "top_3_shareholder_percentage"
        ],
        "location_features": [
            "issuer_cntry_domicile_name", "cntry_of_risk",
            "top_1_shareholder_location", "top_2_shareholder_location",
            "top_3_shareholder_location"
        ],
        "industry_features": [
            "gics_sector", "gics_sub_ind"
        ]
    }
    
    # Add any remaining numerical features to a misc group
    all_grouped = []
    for group in feature_groups.values():
        all_grouped.extend(group)
    
    misc_features = [f for f in column_info['base_numerical'] if f not in all_grouped]
    if misc_features:
        feature_groups["other_metrics"] = sorted(misc_features)
    
    # Validate that all features in groups exist in the data
    all_features = column_info['base_numerical'] + column_info['tree_categorical']
    validated_groups = {}
    for group_name, features in feature_groups.items():
        validated_features = [f for f in features if f in all_features]
        if validated_features:
            validated_groups[group_name] = validated_features
    
    output_path = location.get_path('data/ml_output/raw/metadata', 'feature_groups.json')
    with open(output_path, 'w') as f:
        json.dump(validated_groups, f, indent=2)
    
    print(f"\nCreated {output_path}")
    for group_name, features in validated_groups.items():
        print(f"  {group_name}: {len(features)} features")


def main():
    """Main function to generate all metadata files"""
    
    # Initialize location - use parent directory if we're in scripts
    base_dir = os.getcwd()
    if os.path.basename(base_dir) == 'scripts':
        base_dir = os.path.dirname(base_dir)
    location = Location(base_dir)
    
    # Create metadata directory inside ml_output/raw
    metadata_dir = location.get_path('data/ml_output/raw', 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    print("Generating ML pipeline metadata files...")
    print("="*60)
    
    # Extract column information
    print("Extracting column information from CSV files...")
    column_info = extract_column_information(location)
    
    # Create all metadata files
    create_linear_model_columns_json(location, column_info)
    create_tree_model_columns_json(location, column_info)
    create_yeo_johnson_mapping_json(location, column_info)
    create_feature_groups_json(location, column_info)
    
    print("\n" + "="*60)
    print("Metadata generation completed successfully!")
    print(f"Files saved to: {metadata_dir}")


if __name__ == "__main__":
    main()