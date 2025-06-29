# ESG Score EDA Pipeline Scripts

This directory contains the scripts that form the complete ESG Score data analysis pipeline. Each script performs a specific step in the process of transforming raw ESG data into ML-ready features.

## Pipeline Components

### 1. Missing Values Analysis

**Script:** `analyze_missing_values.py`

Analyzes patterns of missing data in ESG scores, with sector-specific analysis:
- Detects missing value patterns across sectors
- Visualizes missing data distributions
- Applies appropriate imputation strategies
- Outputs: `data/processed/imputed_data.csv`

### 2. Outlier Detection

**Script:** `detect_outliers.py`

Implements statistical outlier detection methods:
- IQR (Interquartile Range) method
- Z-score method
- Adds outlier flags to identified data points
- Creates detailed outlier reports
- Outputs: `data/processed/outliers_mean_processed.csv`

### 3. Isolation Forest Outlier Detection

**Script:** `detect_isolation_forest_outliers.py`

Uses machine learning for advanced outlier detection:
- Applies the Isolation Forest algorithm
- Optimizes model parameters with Optuna
- Compares standard and tuned models
- Generates comprehensive visualizations
- Outputs: `data/processed/outlier_all.csv`

### 4. Outlier Visualization

**Script:** `generate_outlier_comparison_plots.py`

Creates visual comparisons of outlier detection methods:
- Generates boxplots of original vs. processed data
- Shows impact of different outlier detection approaches
- Provides visual validation of detection algorithms
- Outputs: Visualizations in `visualizations/outliers/`

### 5. Outlier Imputation

**Script:** `impute_outliers.py`

Replaces detected outliers with appropriate values:
- Imputes outliers from all detection methods
- Shows before/after distributions
- Creates special handling for isolation forest outliers
- Outputs: `data/processed/outlier_forest_imputed.csv`

### 6. Variable Transformation

**Script:** `apply_yeo_johnson.py`

Normalizes data distributions:
- Applies Yeo-Johnson transformations
- Creates comparison visualizations
- Transforms skewed variables to more normal distributions
- Outputs: `data/processed/combined_yeo_johnson.csv`

### 7. Feature Engineering

**Script:** `feature_engineering.py`

Prepares categorical features:
- Performs one-hot encoding
- Visualizes feature expansion
- Creates clean, ML-ready categorical features
- Outputs: `data/processed/categorical_encoded.csv`

### 8. ML Data Creation

**Script:** `create_ml_model_data.py`

Creates the final dataset for ML modeling:
- Combines all processed data
- Ensures compatibility with ML libraries
- Creates a single, clean dataset for modeling
- Outputs: `data/combined_df_for_ml_models.csv`

### 9. Score Generation

**Script:** `generate_score.py`

Creates a simplified score.csv file for evaluation:
- Extracts ESG scores from the ML-ready dataset
- Creates a compact file with just issuer names and scores
- Ensures compatibility with evaluation scripts
- Outputs: `data/score.csv`

## Using the Pipeline

### Running with the Main Pipeline Script

The recommended way to run these scripts is through the main pipeline script:

```bash
# Run the entire pipeline
python main.py --all

# Or run specific components
python main.py --missing-values
python main.py --outliers-iqr-z
python main.py --outliers-forest
python main.py --outlier-plots
python main.py --impute-outliers
python main.py --yeo-johnson
python main.py --feature-eng
```

### Running Individual Scripts

You can also run the scripts individually in sequence:

```bash
python scripts/analyze_missing_values.py
python scripts/detect_outliers.py
python scripts/detect_isolation_forest_outliers.py
python scripts/generate_outlier_comparison_plots.py
python scripts/impute_outliers.py
python scripts/apply_yeo_johnson.py
python scripts/feature_engineering.py
python scripts/create_ml_model_data.py
python scripts/generate_score.py
```

## Visualization Output

All visualizations are saved in the `visualizations/` directory:

- `visualizations/missing_values/`: Missing value patterns and analysis
- `visualizations/outliers/`: Outlier detection and comparison visualizations
- `visualizations/outliers/isolation/`: Isolation Forest specific visualizations
- `visualizations/outliers/boxplots/`: Boxplots comparing original and processed data
- `visualizations/outliers/shap/`: SHAP explanations for outlier detection
- `visualizations/yeo_johnson/`: Transformation visualizations
- `visualizations/one_hot/`: Feature engineering visualizations
- `visualizations/performance/`: Pipeline performance metrics

## Additional Utility Scripts

- `shap_compatibility.py`: Helper for SHAP visualizations with newer NumPy versions

## Documentation

Each script contains detailed docstrings explaining parameters, functions, and outputs. For implementation details, refer to the source code and comments within each script.