# Model-Specific Data Files

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
