#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create Model-Specific Data Files for ESG Score analysis

This script creates two separate data files optimized for different model types:
1. combined_df_for_linear_models.csv - For Elastic Net (with BOTH original and transformed numericals + one-hot encoded categoricals)
2. combined_df_for_tree_models.csv - For XGBoost/LightGBM/CatBoost (with BOTH original and transformed numericals + native categoricals)

IMPORTANT: 
- Files are saved ONLY to data/ml_output/raw/ to avoid duplicates
- esg_score is EXCLUDED from both files as it's the target variable
"""

import os
import sys
import pandas as pd
import numpy as np

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location


def create_linear_model_data(location):
    """
    Create data file optimized for linear models (Elastic Net).
    Includes BOTH original AND Yeo-Johnson transformed numerical features,
    plus one-hot encoded categorical features.
    EXCLUDES esg_score (target variable).
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("\nCreating linear model data...")
    
    try:
        # 1. Load the original numerical data (non-transformed)
        numerical_path = location.get_path('data/processed', 'numerical.csv')
        original_numerical_df = pd.read_csv(numerical_path, index_col='issuer_name')
        
        # CRITICAL: Remove esg_score (target variable) if it exists
        if 'esg_score' in original_numerical_df.columns:
            print("Removing esg_score (target variable) from numerical features")
            original_numerical_df = original_numerical_df.drop(columns=['esg_score'])
        
        print(f"Loaded {len(original_numerical_df.columns)} original numerical features (excluding esg_score)")
        
        # 2. Load the Yeo-Johnson transformed numerical data
        yeo_path = location.get_path('data/processed', 'all_yeo_johnson.csv')
        yeo_df = pd.read_csv(yeo_path, index_col='issuer_name')
        
        # Keep only the yeo_joh_ prefixed columns
        yeo_cols = [col for col in yeo_df.columns if col.startswith('yeo_joh_')]
        
        # Remove yeo_joh_esg_score if it exists
        yeo_cols = [col for col in yeo_cols if col != 'yeo_joh_esg_score']
        
        yeo_df = yeo_df[yeo_cols]
        print(f"Loaded {len(yeo_cols)} Yeo-Johnson transformed features (excluding esg_score)")
        
        # 3. Load the one-hot encoded categorical data
        categorical_encoded_path = location.get_path('data/processed', 'categorical_encoded.csv')
        categorical_encoded_df = pd.read_csv(categorical_encoded_path, index_col='issuer_name')
        print(f"Loaded {len(categorical_encoded_df.columns)} one-hot encoded categorical features")
        
        # Verify we have the correct categorical columns (including shareholder locations)
        location_cols = [col for col in categorical_encoded_df.columns if 'shareholder_location' in col]
        print(f"Found {len(location_cols)} shareholder location one-hot encoded columns")
        
        # 4. Ensure indices are aligned
        original_numerical_df.sort_index(ascending=True, inplace=True)
        yeo_df.sort_index(ascending=True, inplace=True)
        categorical_encoded_df.sort_index(ascending=True, inplace=True)
        
        # 5. Combine all features
        # First combine original and transformed numerical features
        all_numerical_df = pd.merge(
            original_numerical_df, 
            yeo_df, 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        
        # Then add categorical features
        linear_df = pd.merge(
            all_numerical_df,
            categorical_encoded_df,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        print(f"\nCombined data shape: {linear_df.shape}")
        print(f"Total features: {len(original_numerical_df.columns)} original numerical + "
              f"{len(yeo_cols)} Yeo-Johnson + {len(categorical_encoded_df.columns)} one-hot encoded = "
              f"{len(linear_df.columns)}")
        
        # Verify we have both percentage columns
        percentage_cols = [col for col in linear_df.columns if 'shareholder_percentage' in col]
        print(f"\nShareholder percentage columns: {len(percentage_cols)}")
        for col in sorted(percentage_cols):
            print(f"  - {col}")
        
        # Double-check that esg_score is not present
        if 'esg_score' in linear_df.columns or 'yeo_joh_esg_score' in linear_df.columns:
            raise ValueError("ERROR: esg_score found in final dataset - target leakage!")
        
        # 6. Skip standardization - keep same scale as tree model data
        print("\nNote: Numerical features are NOT standardized (same scale as tree model data)")
        
        # 7. Save ONLY to ml_output directory (not to data/)
        output_path = location.get_path('data/ml_output/raw', 'combined_df_for_linear_models.csv')
        linear_df.to_csv(output_path, index=True)
        print(f"\nLinear model data saved to: {output_path}")
        print(f"Final shape: {linear_df.shape} (should NOT include esg_score)")
        
        return True
        
    except Exception as e:
        print(f"Error creating linear model data: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_tree_model_data(location):
    """
    Create data file optimized for tree-based models.
    Uses BOTH original AND Yeo-Johnson transformed numerical features plus native categorical features.
    EXCLUDES esg_score (target variable).
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("\nCreating tree model data...")
    
    try:
        # 1. Load the original numerical data (NOT transformed)
        numerical_path = location.get_path('data/processed', 'numerical.csv')
        numerical_df = pd.read_csv(numerical_path, index_col='issuer_name')
        
        # CRITICAL: Remove esg_score (target variable) if it exists
        if 'esg_score' in numerical_df.columns:
            print("Removing esg_score (target variable) from numerical features")
            numerical_df = numerical_df.drop(columns=['esg_score'])
        
        print(f"Loaded {len(numerical_df.columns)} original numerical features (excluding esg_score)")
        
        # Verify we have the original percentage columns
        percentage_cols = [col for col in numerical_df.columns if 'shareholder_percentage' in col]
        print(f"Found {len(percentage_cols)} shareholder percentage columns (original values)")
        
        # 2. Load the Yeo-Johnson transformed numerical data
        yeo_path = location.get_path('data/processed', 'all_yeo_johnson.csv')
        yeo_df = pd.read_csv(yeo_path, index_col='issuer_name')
        
        # Keep only the yeo_joh_ prefixed columns
        yeo_cols = [col for col in yeo_df.columns if col.startswith('yeo_joh_')]
        
        # Remove yeo_joh_esg_score if it exists
        yeo_cols = [col for col in yeo_cols if col != 'yeo_joh_esg_score']
        
        yeo_df = yeo_df[yeo_cols]
        print(f"Loaded {len(yeo_cols)} Yeo-Johnson transformed features (excluding esg_score)")
        
        # 3. Load the original categorical data (NOT one-hot encoded)
        categorical_path = location.get_path('data/processed', 'categorical.csv')
        categorical_df = pd.read_csv(categorical_path, index_col='issuer_name')
        
        # Drop columns that were excluded in the original pipeline
        cols_to_drop = ["isin", "esg_rating", "issuer_name.1"]
        categorical_df = categorical_df.drop(columns=[col for col in cols_to_drop if col in categorical_df.columns])
        print(f"Loaded {len(categorical_df.columns)} categorical features")
        
        # Verify we have the location columns
        location_cols = [col for col in categorical_df.columns if 'shareholder_location' in col]
        print(f"Found {len(location_cols)} shareholder location columns (native categorical)")
        
        # 4. Ensure indices are aligned and sorted
        numerical_df.sort_index(ascending=True, inplace=True)
        yeo_df.sort_index(ascending=True, inplace=True)
        categorical_df.sort_index(ascending=True, inplace=True)
        
        # 5. Combine all data: original numerical + Yeo-Johnson transformed + categorical
        # First combine numerical features (original + transformed)
        all_numerical_df = pd.merge(
            numerical_df,
            yeo_df,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Then add categorical features
        tree_df = pd.merge(
            all_numerical_df, 
            categorical_df, 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        
        # 6. Convert categorical columns to 'category' dtype for optimal tree model performance
        for col in categorical_df.columns:
            if col in tree_df.columns:
                tree_df[col] = tree_df[col].astype('category')
        
        print(f"\nCombined data shape: {tree_df.shape}")
        print(f"Total features: {len(numerical_df.columns)} original numerical + "
              f"{len(yeo_cols)} Yeo-Johnson + {len(categorical_df.columns)} categorical = {len(tree_df.columns)}")
        
        # Double-check that esg_score is not present
        if 'esg_score' in tree_df.columns or 'yeo_joh_esg_score' in tree_df.columns:
            raise ValueError("ERROR: esg_score found in final dataset - target leakage!")
        
        # 7. Save ONLY to ml_output directory (not to data/)
        output_path = location.get_path('data/ml_output/raw', 'combined_df_for_tree_models.csv')
        tree_df.to_csv(output_path, index=True)
        print(f"\nTree model data saved to: {output_path}")
        print(f"Final shape: {tree_df.shape} (should NOT include esg_score)")
        
        # Print memory comparison
        linear_path = location.get_path('data/ml_output/raw', 'combined_df_for_linear_models.csv')
        if os.path.exists(linear_path):
            linear_size = os.path.getsize(linear_path) / (1024 * 1024)  # MB
            tree_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"\nFile size comparison:")
            print(f"  Linear model file: {linear_size:.2f} MB")
            print(f"  Tree model file: {tree_size:.2f} MB")
            print(f"  Reduction: {(1 - tree_size/linear_size)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error creating tree model data: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_score_file(location):
    """
    Create score.csv file containing the original ESG scores from raw data.
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("\nCreating score.csv with original ESG scores...")
    
    try:
        # Load the raw data to get original ESG scores
        raw_data_path = location.get_path('data/raw', 'df_cleaned.csv')
        raw_df = pd.read_csv(raw_data_path)
        
        # Set issuer_name as index to match other files
        raw_df.set_index('issuer_name', inplace=True)
        
        # Extract only the esg_score column
        score_df = raw_df[['esg_score']].copy()
        
        # Verify the scores are in the expected range (0-10)
        print(f"ESG score range: {score_df['esg_score'].min():.2f} to {score_df['esg_score'].max():.2f}")
        print(f"ESG score mean: {score_df['esg_score'].mean():.2f}")
        print(f"Number of companies: {len(score_df)}")
        
        # Save to ml_output directory
        output_path = location.get_path('data/ml_output/raw', 'score.csv')
        score_df.to_csv(output_path, index=True)
        print(f"Score file saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating score file: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_zscore_file(location):
    """
    Create score_zscore.csv file containing z-score standardized ESG scores.
    
    Z-score standardization: (X - mean) / std_dev
    This transforms the scores to have mean=0 and std_dev=1
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("\nCreating score_zscore.csv with z-score standardized ESG scores...")
    
    try:
        # Load the raw data to get original ESG scores
        raw_data_path = location.get_path('data/raw', 'df_cleaned.csv')
        raw_df = pd.read_csv(raw_data_path)
        
        # Set issuer_name as index to match other files
        raw_df.set_index('issuer_name', inplace=True)
        
        # Extract only the esg_score column
        original_scores = raw_df['esg_score'].copy()
        
        # Calculate z-score standardization
        mean = original_scores.mean()
        std = original_scores.std()
        
        # Create DataFrame with z-score values
        zscore_df = pd.DataFrame(index=raw_df.index)
        zscore_df['esg_score_zscore'] = (original_scores - mean) / std
        
        # Verify the z-scores have mean≈0 and std≈1
        print(f"Original scores - Mean: {mean:.3f}, Std: {std:.3f}")
        print(f"Z-score range: {zscore_df['esg_score_zscore'].min():.3f} to {zscore_df['esg_score_zscore'].max():.3f}")
        print(f"Z-score mean: {zscore_df['esg_score_zscore'].mean():.6f} (should be ≈0)")
        print(f"Z-score std: {zscore_df['esg_score_zscore'].std():.6f} (should be ≈1)")
        print(f"Number of companies: {len(zscore_df)}")
        
        # Save to ml_output directory
        output_path = location.get_path('data/ml_output/raw', 'score_zscore.csv')
        zscore_df.to_csv(output_path, index=True)
        print(f"Z-score file saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating z-score file: {e}")
        import traceback
        traceback.print_exc()
        return False


def remove_duplicate_files(location):
    """
    Remove duplicate files from data/ directory that should only exist in data/ml_output/raw/
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
    """
    print("\nRemoving duplicate files from data/ directory...")
    
    files_to_remove = [
        'combined_df_for_linear_models.csv',
        'combined_df_for_tree_models.csv',
        'combined_df_for_ml_models.csv',  # obsolete file
        'score.csv'
    ]
    
    removed_count = 0
    for filename in files_to_remove:
        file_path = location.get_path('data', filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  Removed: {file_path}")
            removed_count += 1
    
    if removed_count == 0:
        print("  No duplicate files found")
    else:
        print(f"  Removed {removed_count} duplicate files")


def create_data_description(location):
    """
    Create a description file explaining the differences between the datasets.
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
    """
    description = """# Model-Specific Data Files

This directory contains ML-ready datasets optimized for different model types.

## CRITICAL: Target Variable Exclusion

**IMPORTANT**: The `esg_score` column is the TARGET variable and is NOT included in any feature files.
It is saved separately in `score.csv` for model evaluation.

## Files in data/ml_output/raw/

### 1. combined_df_for_linear_models.csv
- **Purpose**: Optimized for linear models (Elastic Net, Lasso, Ridge)
- **Features**: 
  - Original numerical features (26 - excluding esg_score)
  - Yeo-Johnson transformed numerical features (26)
  - One-hot encoded categorical features (336)
  - Numerical features are NOT standardized (same scale as tree model data)
- **Total**: 388 features (NO esg_score)
- **Use for**: Elastic Net, Lasso, Ridge, Linear SVM

### 2. combined_df_for_tree_models.csv
- **Purpose**: Optimized for tree-based models
- **Features**:
  - Original numerical features (26 - excluding esg_score)
  - Yeo-Johnson transformed numerical features (26)
  - Native categorical features (7)
  - Categorical features are stored as 'category' dtype
- **Total**: ~59 features (NO esg_score)
- **Benefits**: 
  - Includes both original and transformed features for feature importance analysis
  - Native categorical format for built-in tree model handling
  - No standardization (trees don't need it)
- **Use for**: XGBoost, LightGBM, CatBoost, Random Forest

### 3. score.csv
- **Purpose**: Target variable for model training and evaluation
- **Content**: Contains only the esg_score column with ORIGINAL values (0-10 scale)
- **Usage**: Load separately as y_train/y_test

### 4. score_zscore.csv
- **Purpose**: Z-score standardized target variable for models that benefit from normalized targets
- **Content**: Contains esg_score_zscore column with standardized values (mean=0, std=1)
- **Transformation**: (X - 6.779) / 1.898
- **Usage**: Alternative target for models that perform better with standardized targets

## Key Differences

1. **Shareholder Features**:
   - `top_*_shareholder_percentage`: QUANTITATIVE (numerical) - gets Yeo-Johnson transformed
   - `top_*_shareholder_location`: QUALITATIVE (categorical) - gets one-hot encoded

2. **Transformations**:
   - Linear models: Include both original + transformed numericals, NOT standardized
   - Tree models: Include both original + transformed numericals, NOT standardized

3. **Categorical Encoding**:
   - Linear models: One-hot encoded (necessary for linear algorithms)
   - Tree models: Native categorical format (trees can handle directly)

4. **Target Variable**:
   - NEVER included in feature files
   - Stored separately in score.csv

## Usage Examples

### For Elastic Net:
```python
# Load features
X = pd.read_csv('data/ml_output/raw/combined_df_for_linear_models.csv', index_col='issuer_name')
# Load target
y = pd.read_csv('data/ml_output/raw/score.csv', index_col='issuer_name')['esg_score']

# Data is NOT standardized - apply your own preprocessing if needed
# You can select either original or yeo_joh_ features based on your needs
```

### For XGBoost:
```python
# Load features
X = pd.read_csv('data/ml_output/raw/combined_df_for_tree_models.csv', index_col='issuer_name')
# Load target
y = pd.read_csv('data/ml_output/raw/score.csv', index_col='issuer_name')['esg_score']

# Identify categorical columns
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
# Use native categorical support in XGBoost
```

### For LightGBM:
```python
# Load features
X = pd.read_csv('data/ml_output/raw/combined_df_for_tree_models.csv', index_col='issuer_name')
# Load target
y = pd.read_csv('data/ml_output/raw/score.csv', index_col='issuer_name')['esg_score']

# Convert categorical columns to 'category' dtype if needed
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')
```

### For CatBoost:
```python
# Load features
X = pd.read_csv('data/ml_output/raw/combined_df_for_tree_models.csv', index_col='issuer_name')
# Load target
y = pd.read_csv('data/ml_output/raw/score.csv', index_col='issuer_name')['esg_score']

# Get categorical feature indices
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
cat_feature_indices = [X.columns.get_loc(col) for col in cat_features]
```

## Important Notes

1. Files are ONLY saved in `data/ml_output/raw/` to avoid duplicates
2. The obsolete `combined_df_for_ml_models.csv` should not be used
3. Always use the files from `data/ml_output/raw/` for ML training
4. NEVER include esg_score in your feature set - always load it separately
5. Tree models now have access to both original and transformed features for better performance
6. Two target options available:
   - `score.csv`: Original ESG scores (0-10 scale)
   - `score_zscore.csv`: Z-score standardized scores (mean=0, std=1)
"""
    
    output_path = location.get_path('data/ml_output', 'MODEL_DATA_README.md')
    with open(output_path, 'w') as f:
        f.write(description)
    print(f"\nData description saved to: {output_path}")


def main():
    """Main function to create model-specific data files."""
    # Initialize location
    base_dir = os.getcwd()    
    location = Location(base_dir)
    
    print("Creating model-specific data files...")
    print("="*60)
    
    # Ensure ml_output directories exist
    os.makedirs(location.get_path('data/ml_output/raw'), exist_ok=True)
    
    # Create linear model data
    linear_success = create_linear_model_data(location)
    
    # Create tree model data
    tree_success = create_tree_model_data(location)
    
    # Create score file with original ESG scores
    score_success = create_score_file(location)
    
    # Create z-score standardized file
    zscore_success = create_zscore_file(location)
    
    # Remove duplicate files from data/ directory
    if linear_success and tree_success and score_success and zscore_success:
        remove_duplicate_files(location)
        create_data_description(location)
        
        # Generate metadata files
        print("\nGenerating metadata files...")
        try:
            from generate_metadata import main as generate_metadata_main
            # Change to the scripts directory to run the metadata generation
            original_dir = os.getcwd()
            os.chdir(os.path.dirname(__file__))
            generate_metadata_main()
            os.chdir(original_dir)
        except Exception as e:
            print(f"Warning: Could not generate metadata files: {e}")
        
        print("\n" + "="*60)
        print("Model-specific data creation completed successfully!")
        print("Files saved to: data/ml_output/raw/")
        print("IMPORTANT: esg_score has been excluded from all feature files")
        print("         and saved separately in:")
        print("         - score.csv with ORIGINAL values (0-10 scale)")
        print("         - score_zscore.csv with Z-SCORE standardized values (mean=0, std=1)")
        print("="*60)
        return 0
    else:
        print("\nError: Some data files could not be created.")
        return 1


if __name__ == "__main__":
    sys.exit(main())