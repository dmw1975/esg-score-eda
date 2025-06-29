# Model-Specific Data Files

This directory contains three ML-ready datasets, each optimized for different model types:

## 1. combined_df_for_ml_models.csv
- **Purpose**: Original file, maintained for backward compatibility
- **Features**: Yeo-Johnson transformed numericals + one-hot encoded categoricals
- **Use for**: General purpose, existing notebooks/scripts

## 2. combined_df_for_linear_models.csv
- **Purpose**: Optimized for linear models (Elastic Net, Lasso, Ridge)
- **Features**: 
  - Standardized Yeo-Johnson transformed numerical features (including all shareholder percentages)
  - One-hot encoded categorical features
- **Preprocessing**: StandardScaler applied to numerical features
- **Use for**: Elastic Net, Lasso, Ridge, Linear SVM

## 3. combined_df_for_tree_models.csv
- **Purpose**: Optimized for tree-based models
- **Features**:
  - Original numerical features (NOT Yeo-Johnson transformed)
  - Native categorical features (not one-hot encoded)
- **Benefits**: 
  - Smaller file size (~48x reduction in categorical features)
  - Faster training for tree models
  - Better handling of categorical relationships
- **Use for**: XGBoost, LightGBM, CatBoost, Random Forest

## Usage Examples

### For Elastic Net:
```python
df = pd.read_csv('data/combined_df_for_linear_models.csv', index_col='issuer_name')
# Data is already scaled and one-hot encoded, ready for ElasticNet
```

### For XGBoost:
```python
df = pd.read_csv('data/combined_df_for_tree_models.csv', index_col='issuer_name')
# Identify categorical columns
cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Use native categorical support in XGBoost
```

### For LightGBM:
```python
df = pd.read_csv('data/combined_df_for_tree_models.csv', index_col='issuer_name')
# Convert categorical columns to 'category' dtype if needed
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')
```

### For CatBoost:
```python
df = pd.read_csv('data/combined_df_for_tree_models.csv', index_col='issuer_name')
# Get categorical feature indices
cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_feature_indices = [df.columns.get_loc(col) for col in cat_features]
```
