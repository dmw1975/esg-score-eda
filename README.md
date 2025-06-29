# ESG Score EDA Pipeline

A comprehensive, production-ready pipeline for ESG (Environmental, Social, and Governance) score exploratory data analysis, outlier detection, and machine learning data preparation.

## 🚨 Important: Data Requirements

**This repository requires a data file that is NOT included:** 
- You must provide your own ESG data file: `data/raw/df_cleaned.csv`
- This file should contain your ESG scores and related features
- The pipeline expects CSV format with appropriate column headers
- Example columns: company identifiers, ESG scores, sector information, financial metrics

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Quick Start Guide](#quick-start-guide)
- [Project Structure](#project-structure)
- [Detailed Usage](#detailed-usage)
- [Pipeline Components](#pipeline-components)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This pipeline automates the complete ESG data analysis workflow:
- **Data Analysis**: Missing values, outliers, distributions
- **Data Cleaning**: Smart imputation, outlier handling
- **Feature Engineering**: Transformations, encoding, scaling
- **ML Preparation**: Separate datasets optimized for linear vs tree-based models
- **Visualization**: Comprehensive plots and reports at each stage

### Key Features
- 🔍 Multi-method outlier detection (IQR, Z-score, Isolation Forest)
- 📊 Sector-specific missing value analysis
- 🏗️ Modern Python architecture with type hints
- ⚡ Performance optimized with parallel processing
- 📈 SHAP explainability for outlier detection
- 🎯 Model-specific data preparation

## Prerequisites

- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM recommended
- **OS**: Windows, macOS, or Linux
- **Data**: Your own `df_cleaned.csv` file with ESG data

## Installation & Setup

### Method 1: Quick Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/esg-score-eda.git
cd esg-score-eda

# Run comprehensive setup
bash setup.sh

# Or setup with options:
bash setup.sh --help           # Show all options
bash setup.sh --env            # Only create virtual environment
bash setup.sh --directories    # Only create directory structure
bash setup.sh --run           # Setup and run pipeline
bash setup.sh --sample        # Generate sample data (if available)
```

### Method 2: Using Activation Script

```bash
# Clone and enter directory
git clone https://github.com/yourusername/esg-score-eda.git
cd esg-score-eda

# Activate environment (creates venv if needed, installs dependencies)
source activate_env.sh

# On Windows:
# bash activate_env.sh
```

### Method 3: Manual Setup

```bash
# Clone repository
git clone https://github.com/yourusername/esg-score-eda.git
cd esg-score-eda

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install as editable package for development
pip install -e .

# Create directory structure
mkdir -p data/{raw,processed,interim}
mkdir -p visualizations/{missing_values,outliers,yeo_johnson,one_hot,performance}
```

## Quick Start Guide

### Step 1: Prepare Your Data

Place your ESG data file at:
```
data/raw/df_cleaned.csv
```

Expected data format:
- CSV file with headers
- Should include ESG scores (environmental, social, governance)
- Sector/industry classification recommended
- Company identifiers and financial metrics helpful

### Step 2: Activate Environment

```bash
source activate_env.sh  # Linux/macOS
# or
activate_env.sh         # Windows
```

### Step 3: Run the Pipeline

```bash
# Run complete pipeline
python main.py --all

# Or run specific steps
python main.py --missing-values    # Analyze missing data
python main.py --outliers-iqr-z    # Detect outliers
python main.py --feature-eng       # Engineer features
```

### Step 4: Find Your Results

- **ML-ready data**: `data/combined_df_for_ml_models.csv`
- **Model-specific data**: 
  - `data/combined_df_for_linear_models.csv` (scaled, one-hot encoded)
  - `data/combined_df_for_tree_models.csv` (native categoricals)
- **Visualizations**: `visualizations/` directory
- **Metadata**: `data/ml_output/raw/metadata/` (feature mappings)

## Project Structure

```
esg-score-eda/
├── data/                          # Data directory (git-ignored)
│   ├── raw/                       # Input data
│   │   └── df_cleaned.csv        # YOUR DATA FILE (not provided)
│   ├── processed/                 # Intermediate processing files
│   ├── interim/                   # Temporary files
│   └── ml_output/                 # ML-ready outputs
│       └── raw/
│           ├── combined_df_for_linear_models.csv
│           ├── combined_df_for_tree_models.csv
│           └── metadata/          # Feature mappings
│
├── src/                           # Source code package
│   └── esg_eda/                   # Main package
│       ├── analysis/              # Core analysis modules
│       ├── core/                  # Base classes and config
│       ├── pipeline/              # Pipeline orchestration
│       ├── utils/                 # Utilities
│       └── visualization/         # Plotting functions
│
├── scripts/                       # Individual pipeline scripts
│   ├── analyze_missing_values.py
│   ├── detect_outliers.py
│   ├── detect_isolation_forest_outliers.py
│   ├── impute_outliers.py
│   ├── apply_yeo_johnson.py
│   ├── feature_engineering.py
│   └── create_model_specific_data.py
│
├── notebooks/                     # Jupyter notebooks (exploration)
├── visualizations/                # Generated plots (git-ignored)
├── tests/                         # Unit and integration tests
│   ├── conftest.py               # Test configuration
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── main.py                        # Main CLI entry point
├── location.py                    # Path management utility
├── activate_env.sh                # Quick environment activation
├── setup.sh                       # Comprehensive setup script
├── setup.py                       # Package setup (legacy support)
├── pyproject.toml                 # Modern Python packaging config
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies (optional)
├── .gitignore                     # Git ignore rules
├── CLAUDE.md                      # AI assistant instructions
└── README.md                      # This file
```

### Key Files Explained

- **main.py**: Command-line interface for running the pipeline
- **location.py**: Centralized path management (creates directories as needed)
- **activate_env.sh**: Quick script to activate virtual environment and check dependencies
- **setup.sh**: Comprehensive setup with options for environment, directories, and running
- **pyproject.toml**: Modern Python packaging configuration with tool settings
- **CLAUDE.md**: Instructions for AI assistants (helps with maintenance)

## Detailed Usage

### Running the Complete Pipeline

```bash
# Activate environment first
source activate_env.sh

# Run all steps with default settings
python main.py --all

# Run without performance tracking
python main.py --all --no-time-tracking

# Run with custom data path
python main.py --all --input data/raw/my_data.csv
```

### Running Individual Components

```bash
# 1. Analyze missing values
python main.py --missing-values

# 2. Detect outliers using IQR and Z-score
python main.py --outliers-iqr-z

# 3. Detect outliers using Isolation Forest
python main.py --outliers-forest

# 4. Generate outlier comparison plots
python main.py --outlier-plots

# 5. Impute outliers
python main.py --impute-outliers

# 6. Apply Yeo-Johnson transformation
python main.py --yeo-johnson

# 7. Perform feature engineering
python main.py --feature-eng

# 8. Create model-specific datasets
python main.py --model-specific-data
```

### Using as a Python Package

```python
from src.analysis.missing_values import MissingValuesAnalyzer
from src.analysis.outlier_detection import OutlierDetector
from src.analysis.feature_engineering import FeatureEngineer
import pandas as pd

# Load your data
df = pd.read_csv('data/raw/df_cleaned.csv')

# Analyze missing values
analyzer = MissingValuesAnalyzer()
missing_report = analyzer.analyze_missing_values(df)

# Detect outliers
detector = OutlierDetector()
outliers = detector.detect_outliers_iqr(df, numerical_columns)

# Engineer features
engineer = FeatureEngineer()
df_engineered = engineer.engineer_features(df)
```

## Pipeline Components

### 1. Missing Values Analysis (`analyze_missing_values.py`)
- Identifies missing data patterns
- Tests sector-specific associations
- Compares imputation strategies
- Generates heatmaps and distribution plots

### 2. Outlier Detection
#### Statistical Methods (`detect_outliers.py`)
- **IQR Method**: Uses 1.5 * IQR by default
- **Z-Score Method**: Flags values beyond 3 standard deviations

#### Machine Learning (`detect_isolation_forest_outliers.py`)
- Isolation Forest with hyperparameter tuning
- SHAP explanations for outlier predictions
- Handles high-dimensional data well

### 3. Data Imputation (`impute_outliers.py`)
- Replaces outliers with median values
- Preserves data distribution
- Tracks imputation locations

### 4. Feature Transformation (`apply_yeo_johnson.py`)
- Normalizes skewed distributions
- Improves linear model performance
- Generates before/after visualizations

### 5. Feature Engineering (`feature_engineering.py`)
- One-hot encoding for categoricals
- Handles high-cardinality features
- Creates interaction features

### 6. Model-Specific Preparation (`create_model_specific_data.py`)
- **Linear models**: Scaled features, one-hot encoding
- **Tree models**: Native categoricals, no scaling needed
- Maintains feature mappings

## Output Files

### Primary Outputs

| File | Description | Location |
|------|-------------|----------|
| `combined_df_for_ml_models.csv` | Complete ML-ready dataset | `data/` |
| `combined_df_for_linear_models.csv` | Optimized for linear models | `data/ml_output/raw/` |
| `combined_df_for_tree_models.csv` | Optimized for tree models | `data/ml_output/raw/` |
| `score.csv` | Extracted ESG scores | `data/ml_output/raw/` |

### Metadata Files

| File | Description | Location |
|------|-------------|----------|
| `feature_groups.json` | Categorical/numerical feature lists | `data/ml_output/raw/metadata/` |
| `linear_model_columns.json` | Column names for linear models | `data/ml_output/raw/metadata/` |
| `tree_model_columns.json` | Column names for tree models | `data/ml_output/raw/metadata/` |
| `yeo_johnson_mapping.json` | Transformation mappings | `data/ml_output/raw/metadata/` |

### Intermediate Files

- `data/processed/imputed_data.csv` - Data after missing value imputation
- `data/processed/outlier_all.csv` - Combined outlier detection results
- `data/processed/isolation_forest_outliers.csv` - ML-based outlier flags

### Visualizations

Generated in `visualizations/` directory:
- `missing_values/` - Missing data patterns and heatmaps
- `outliers/` - Outlier detection comparisons
- `yeo_johnson/` - Transformation effects
- `performance/` - Pipeline execution metrics

## Configuration

### Environment Variables

Create a `.env` file for custom settings:

```bash
# Data paths
ESG_DATA_RAW=data/raw
ESG_DATA_PROCESSED=data/processed

# Pipeline settings
ESG_PIPELINE_IQR_MULTIPLIER=1.5
ESG_PIPELINE_Z_SCORE_THRESHOLD=3.0
ESG_PIPELINE_CHUNK_SIZE=10000

# Logging
ESG_LOG_LEVEL=INFO
ESG_DEBUG=false
```

### Python Configuration

```python
# In your scripts
from src.core.config import Settings

settings = Settings()
settings.pipeline.iqr_multiplier = 2.0
settings.visualization.dpi = 300
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run specific test file
pytest tests/unit/test_missing_values.py

# Run with verbose output
pytest -v
```

### Test Structure

- `tests/conftest.py` - Shared fixtures and test data generators
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - End-to-end pipeline tests
- `tests/fixtures/` - Test data files

### Writing Tests

```python
# Example test
def test_outlier_detection(sample_data):
    detector = OutlierDetector()
    result = detector.detect_outliers_iqr(sample_data, ['score'])
    assert 'score_outlier' in result.columns
    assert result['score_outlier'].sum() > 0
```

## Troubleshooting

### Common Issues

**Data file not found**
```
FileNotFoundError: data/raw/df_cleaned.csv
```
Solution: Ensure you've placed your ESG data file in `data/raw/df_cleaned.csv`

**Import errors**
```
ImportError: No module named 'src'
```
Solution: Install package in development mode:
```bash
pip install -e .
```

**Memory errors with large datasets**
```python
# Reduce chunk size in processing
python main.py --all --chunk-size 5000
```

**Missing dependencies**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Getting Help

1. Check error messages in log files
2. Run with debug mode: `export ESG_DEBUG=true`
3. Check the [Issues](https://github.com/yourusername/esg-score-eda/issues) page
4. Review test examples in `tests/` directory

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatters
black src/ scripts/
isort src/ scripts/

# Run linters
flake8 src/
mypy src/

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with scikit-learn, pandas, numpy, and matplotlib
- SHAP for model interpretability
- Isolation Forest implementation from scikit-learn
- Optuna for hyperparameter optimization