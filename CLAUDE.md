# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ESG (Environmental, Social, and Governance) Score Exploratory Data Analysis (EDA) pipeline. The project focuses on comprehensive data analysis, cleaning, outlier detection, feature engineering, and preparation of ESG score data for machine learning models.

## Repository Structure

```
esg-score-eda/
├── data/                           # Data directory
│   ├── raw/                        # Raw input data
│   │   └── df_cleaned.csv         # Source data file
│   ├── processed/                  # Processed datasets
│   │   ├── imputed_data.csv       # Data with imputed missing values
│   │   ├── outlier_all.csv        # Data with outlier flags
│   │   ├── isolation_forest_outliers.csv
│   │   ├── categorical.csv        # Categorical features
│   │   ├── numerical.csv          # Numerical features
│   │   └── pkl/                   # Pickled data structures
│   ├── interim/                    # Intermediate processing files
│   ├── combined_df_for_ml_models.csv  # Final ML-ready dataset
│   ├── combined_df_for_linear_models.csv  # Optimized for linear models
│   ├── combined_df_for_tree_models.csv    # Optimized for tree models
│   └── score.csv                  # Extracted ESG scores
│
├── notebooks/                      # Jupyter notebooks for interactive analysis
│   ├── 1_eda_v1.2.ipynb          # Exploratory data analysis
│   ├── 2_outlier_IQR_Z_v1.4.ipynb # IQR and Z-score outlier detection
│   ├── 3_outlier_Isolation_Forest_v1.6.ipynb # Isolation Forest analysis
│   ├── 4_outlier_processing_v1.2.ipynb # Outlier imputation
│   ├── 5_one_hot_V1.0.ipynb      # One-hot encoding analysis
│   └── 6_yeo_jo_v1.0.ipynb       # Yeo-Johnson transformation
│
├── scripts/                        # Pipeline processing scripts
│   ├── analyze_missing_values.py   # Missing value analysis
│   ├── detect_outliers.py         # Statistical outlier detection
│   ├── detect_isolation_forest_outliers.py # ML-based outlier detection
│   ├── generate_outlier_comparison_plots.py # Visualization
│   ├── impute_outliers.py         # Outlier imputation
│   ├── apply_yeo_johnson.py       # Data transformation
│   ├── feature_engineering.py     # Feature encoding
│   ├── create_ml_model_data.py    # Final dataset creation
│   ├── generate_score.py          # Score extraction
│   └── create_model_specific_data.py # Model-specific data preparation
│
├── src/                           # Core modules
│   └── analysis/                  # Analysis utilities
│       ├── feature_engineering.py
│       ├── missing_values.py
│       └── outlier_detection.py
│
├── visualizations/                # Generated plots and figures
│   ├── missing_values/            # Missing value visualizations
│   ├── outliers/                  # Outlier detection results
│   ├── yeo_johnson/               # Transformation visualizations
│   ├── one_hot/                   # Feature encoding visualizations
│   └── performance/               # Pipeline performance metrics
│
├── main.py                        # Main pipeline orchestrator
├── location.py                    # Path management utility
├── requirements.txt               # Python dependencies
└── setup.sh                       # Setup script
```

## Build/Test Commands

```bash
# Run entire pipeline
python main.py --all

# Run specific components
python main.py --missing-values       # Missing values analysis
python main.py --outliers-iqr-z       # IQR and Z-score outlier detection
python main.py --outliers-forest      # Isolation Forest outlier detection
python main.py --outlier-plots        # Outlier comparison plots
python main.py --impute-outliers      # Outlier imputation
python main.py --yeo-johnson          # Yeo-Johnson transformation
python main.py --feature-eng          # Feature engineering
python main.py --model-specific-data  # Create model-specific data files

# Performance tracking
python main.py --all --no-time-tracking  # Disable execution time tracking
```

## Pipeline Overview

1. **Missing Values Analysis**: Analyzes patterns in missing data, especially by sector
2. **Outlier Detection**: Uses multiple methods (IQR, Z-score, Isolation Forest)
3. **Outlier Imputation**: Replaces detected outliers with appropriate values
4. **Data Transformation**: Applies Yeo-Johnson transformation for normalization
5. **Feature Engineering**: One-hot encoding and feature preparation
6. **Final Data Creation**: Combines all processed data into ML-ready format
7. **Model-Specific Data**: Creates optimized datasets for linear vs tree models

## Key Features

- **Sector-specific analysis**: Special handling for different business sectors
- **Multiple outlier detection methods**: Statistical and ML-based approaches
- **Comprehensive visualization**: Automatic generation of analysis plots
- **Performance tracking**: Execution time monitoring and visualization
- **Flexible pipeline**: Run all steps or individual components
- **ML-ready outputs**: Prepared datasets for different model types

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- pandas==2.0.3
- numpy==1.26.4
- scikit-learn==1.4.0
- matplotlib==3.8.2
- seaborn==0.13.0
- shap>=0.44.0 (for model explainability)
- optuna==3.5.0 (for hyperparameter optimization)

## Data Files

- **Input**: `data/raw/df_cleaned.csv` - Primary source data
- **Output**: `data/combined_df_for_ml_models.csv` - Final ML-ready dataset
- **Score file**: `data/score.csv` - Extracted ESG scores for evaluation

## Development Best Practices (CRITICAL - Going Forward)

### File Management
1. **NO STANDALONE SCRIPTS**: Everything must integrate into the pipeline
   - Don't create diagnostic scripts that require manual execution
   - Don't create fix scripts that generate patches
   - All functionality must run automatically via main.py

2. **DIRECT MODIFICATIONS ONLY**: Change files in place
   - Use version control for safety, not create new files
   - Modify pipeline components directly
   - No intermediate fix generators

3. **CLEAN AS YOU GO**: Delete temporary files immediately
   - Remove diagnostic scripts after debugging
   - Don't leave abandoned attempts
   - No accumulation of test files

4. **FOLLOW ARCHITECTURE**: Respect project structure
   - Scripts go in scripts/ directory
   - Core modules in src/analysis/
   - Outputs in appropriate data/ subdirectories
   - Visualizations in visualizations/

### Integration Requirements
1. **Automatic Execution**: All fixes must run without manual intervention
2. **Pipeline Integration**: New functionality must be called by main.py
3. **No Manual Steps**: Everything automated through command-line flags
4. **Proper Imports**: New modules must be imported where needed

### Problem-Solving Approach
1. **Understand First**: Use logging/debugging instead of creating test scripts
2. **Fix In Place**: Modify existing files rather than creating new ones
3. **Test Through Pipeline**: Use main.py to test changes
4. **Document Changes**: Update this file when making significant changes

### What NOT to Do
- ❌ Create `test_*.py` files in root directory for debugging
- ❌ Create `fix_*.py` scripts that generate patches
- ❌ Create `verify_*.py` scripts for manual checking
- ❌ Leave temporary files after fixing issues
- ❌ Create workarounds instead of permanent solutions

### What TO Do Instead
- ✅ Use logging for understanding issues
- ✅ Modify source files directly
- ✅ Ensure all changes integrate with main.py
- ✅ Clean up any temporary work immediately
- ✅ Follow the established project structure

## Code Style Guidelines
- **Imports**: Standard library first, third-party packages second, local modules last
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Docstrings**: Include descriptions for all functions and classes
- **Comments**: Explain complex logic, avoid obvious comments
- **Line Length**: Keep lines under 100 characters when possible
- **Error Handling**: Use try/except for anticipated errors, with descriptive messages

## Pipeline Best Practices
1. **Data Validation**: Check for NaN values and data types before processing
2. **Reproducibility**: Use consistent random seeds where applicable
3. **Memory Efficiency**: Process large datasets in chunks when possible
4. **Error Recovery**: Pipeline should continue with next step if one fails
5. **Logging**: Each script should log its progress and any issues
6. **Visualization**: Save all plots to appropriate directories

## Avoiding Common Pitfalls
1. **Path Management**: Use location.py for consistent file paths
2. **Data Dependencies**: Ensure required input files exist before processing
3. **Output Validation**: Check that output files are created successfully
4. **Version Compatibility**: Maintain compatibility with specified package versions
5. **Memory Usage**: Monitor memory usage for large datasets
   - No intermediate fix generators

3. **CLEAN AS YOU GO**: Delete temporary files immediately
   - Remove diagnostic scripts after debugging
   - Don't leave abandoned attempts
   - No accumulation of test files

4. **FOLLOW ARCHITECTURE**: Respect project structure
   - Test files go in proper test directories
   - No root-level utility scripts
   - Maintain module organization

### Integration Requirements
1. **Automatic Execution**: All fixes must run without manual intervention
2. **Pipeline Integration**: New functionality must be called by main.py
3. **No Manual Steps**: Everything automated through command-line flags
4. **Proper Imports**: New modules must be imported where needed

### Problem-Solving Approach
1. **Understand First**: Use debugger/logging instead of creating test scripts
2. **Fix In Place**: Modify existing files rather than creating new ones
3. **Test Properly**: Use the existing test framework
4. **Document Changes**: Update this file when making significant changes

### What NOT to Do
- ❌ Create `test_*.py` files in root directory for debugging
- ❌ Create `fix_*.py` scripts that generate patches
- ❌ Create `verify_*.py` scripts for manual checking
- ❌ Leave temporary files after fixing issues
- ❌ Create workarounds instead of permanent solutions

### What TO Do Instead
- ✅ Use logging and debugger for understanding issues
- ✅ Modify source files directly
- ✅ Ensure all changes integrate with main.py
- ✅ Clean up any temporary work immediately
- ✅ Follow the established project structure