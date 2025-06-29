# ESG Score EDA Pipeline

A comprehensive, production-ready pipeline for ESG (Environmental, Social, and Governance) score data analysis. This pipeline automates the entire workflow from raw data to ML-ready features, including advanced outlier detection, missing value imputation, and feature engineering.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Output Files](#output-files)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Key Features

### 🔍 Advanced Data Analysis
- **Missing Values Analysis**: Sector-specific patterns detection with multiple imputation strategies
- **Multi-Method Outlier Detection**: Statistical (IQR, Z-score) and ML-based (Isolation Forest) approaches
- **Feature Engineering**: Automated encoding, transformation, and feature creation

### 📊 Comprehensive Visualizations
- Missing value heatmaps and distribution analysis
- Outlier detection comparisons and SHAP explanations
- Before/after transformation plots
- Pipeline performance metrics

### 🏗️ Production-Ready Architecture
- Modern Python package structure (`src/esg_eda`)
- Type hints and comprehensive documentation
- Pydantic v2 configuration management
- Modular design for easy customization
- Cross-platform compatibility

### ⚡ Performance Optimized
- Parallel processing support
- Memory-efficient chunk processing
- Execution time tracking
- Configurable pipeline stages

## Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended for large datasets

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/esg-score-eda.git
cd esg-score-eda

# Option 1: Use the activation script (Recommended)
source activate_env.sh

# Option 2: Manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install package in development mode (optional)
pip install -e .
```

### Using the Activation Script

The `activate_env.sh` script automatically:
- Creates a virtual environment if it doesn't exist
- Activates the virtual environment
- Installs all dependencies from requirements.txt
- Displays helpful information about running the pipeline

```bash
# Activate the environment
source activate_env.sh

# When done, deactivate with:
deactivate
```

## Quick Start

### 1. Activate the Environment
```bash
source activate_env.sh
```

### 2. Prepare Your Data
Place your ESG data file at: `data/raw/df_cleaned.csv`

### 3. Run the Complete Pipeline
```bash
python main.py --all
```

### 4. Find Your Results
- Processed data: `data/combined_df_for_ml_models.csv`
- Visualizations: `visualizations/` directory
- Model-specific data: `data/combined_df_for_linear_models.csv` and `data/combined_df_for_tree_models.csv`

## Project Structure

```
esg-score-eda/
├── src/esg_eda/               # Main package
│   ├── __init__.py           
│   ├── analysis/              # Analysis modules
│   │   ├── missing_values.py  # Missing value analysis
│   │   ├── outlier_detection.py # Outlier detection
│   │   └── feature_engineering.py # Feature transformations
│   ├── core/                  # Core functionality
│   │   ├── config.py          # Pydantic configuration
│   │   ├── base.py            # Base classes
│   │   └── exceptions.py      # Custom exceptions
│   ├── pipeline/              # Pipeline orchestration
│   │   └── orchestrator.py    # Main pipeline class
│   ├── utils/                 # Utilities
│   │   └── logging.py         # Logging configuration
│   └── visualization/         # Visualization tools
├── scripts/                   # Pipeline scripts (legacy)
├── data/                      # Data directory
│   ├── raw/                   # Input data
│   ├── processed/             # Intermediate files
│   └── ml_output/             # ML-ready outputs
├── visualizations/            # Generated plots
├── tests/                     # Unit tests
├── activate_env.sh           # Environment activation script
├── main.py                    # CLI entry point
├── pyproject.toml            # Modern Python packaging
├── requirements.txt          # Dependencies
├── claude.md                 # AI assistant instructions
└── README.md                 # This file
```

## Usage Guide

### Command Line Interface

First, ensure your environment is activated:
```bash
source activate_env.sh
```

The pipeline can be run via the main CLI with various options:

```bash
# Run complete pipeline
python main.py --all

# Run specific components
python main.py --missing-values      # Missing values analysis only
python main.py --outliers-iqr-z      # IQR and Z-score outlier detection
python main.py --outliers-forest     # Isolation Forest outlier detection
python main.py --outlier-plots       # Generate comparison plots
python main.py --impute-outliers     # Impute detected outliers
python main.py --yeo-johnson         # Apply Yeo-Johnson transformation
python main.py --feature-eng         # Feature engineering
python main.py --model-specific-data # Create model-specific datasets

# Additional options
python main.py --all --no-time-tracking  # Disable performance tracking
```

### Python API

Use the package programmatically:

```python
from esg_eda import ESGPipeline, MissingValuesAnalyzer, OutlierDetector

# Run complete pipeline
pipeline = ESGPipeline(
    input_path="data/raw/df_cleaned.csv",
    output_dir="output"
)
results = pipeline.run_full_pipeline()

# Or use individual components
import pandas as pd
data = pd.read_csv("data/raw/df_cleaned.csv")

# Missing values analysis
analyzer = MissingValuesAnalyzer(data)
missing_results = analyzer.analyze()
analyzer.visualize("visualizations/missing")

# Outlier detection
detector = OutlierDetector(data)
outlier_results = detector.detect_all_methods()
detector.visualize("visualizations/outliers")
```

## Pipeline Components

### 1. Missing Values Analysis
- Analyzes patterns in missing data
- Tests sector-specific associations
- Evaluates multiple imputation strategies
- Generates comprehensive visualizations

### 2. Outlier Detection
- **IQR Method**: Statistical approach using interquartile range
- **Z-Score Method**: Identifies extreme values based on standard deviations
- **Isolation Forest**: ML-based anomaly detection with hyperparameter tuning
- **SHAP Analysis**: Explains why points are considered outliers

### 3. Feature Engineering
- **Categorical Encoding**: Smart one-hot encoding with cardinality limits
- **Numerical Transformation**: Yeo-Johnson transformation for skewed features
- **Feature Creation**: Automated feature interactions and derived features
- **Model-Specific Preparation**: Separate datasets for linear vs tree models

## Configuration

The pipeline uses Pydantic v2 for configuration management. Settings can be customized via:

### Environment Variables
```bash
export ESG_PIPELINE_IQR_MULTIPLIER=1.5
export ESG_PIPELINE_Z_SCORE_THRESHOLD=3.0
export ESG_LOG_LEVEL=INFO
```

### Configuration File (.env)
```ini
ESG_DATA_RAW_DATA=data/raw
ESG_DATA_PROCESSED_DATA=data/processed
ESG_PIPELINE_CHUNK_SIZE=10000
ESG_DEBUG=false
```

### Python Configuration
```python
from esg_eda import get_settings

settings = get_settings()
settings.pipeline.iqr_multiplier = 2.0
settings.pipeline.z_score_threshold = 2.5
```

## API Reference

### Core Classes

#### `ESGPipeline`
Main orchestrator for the complete pipeline.

```python
pipeline = ESGPipeline(input_path="data.csv", output_dir="results")
pipeline.run_full_pipeline()
```

#### `MissingValuesAnalyzer`
Analyzes and handles missing values.

```python
analyzer = MissingValuesAnalyzer(data, sector_column='gics_sector')
results = analyzer.analyze()
imputed_data = analyzer.impute(strategy='sector_mean')
```

#### `OutlierDetector`
Detects outliers using multiple methods.

```python
detector = OutlierDetector(data)
results = detector.detect_all_methods(iqr_threshold=1.5, z_threshold=3.0)
processed = detector.process_outliers(method='median')
```

#### `FeatureEngineer`
Handles feature transformation and engineering.

```python
engineer = FeatureEngineer(data)
engineer.identify_features(exclude_cols=['id'])
encoded = engineer.encode_categorical(max_categories=50)
normalized = engineer.normalize_numerical(method='standard')
```

## Output Files

### Primary Outputs
| File | Description |
|------|-------------|
| `data/raw/combined_df_for_ml_models.csv` | Complete ML-ready dataset |
| `data/raw/combined_df_for_linear_models.csv` | Optimized for linear models (scaled, one-hot) |
| `data/raw/combined_df_for_tree_models.csv` | Optimized for tree models (native categoricals) |
| `data/score.csv` | Extracted ESG scores |
| `data//raw/metadata/linear_model_columns.json` | Column names for linear models|
| `data//raw/metadata/tree__model_columns.json` | Column names for tree models|
| `data//raw/metadata/yeo_johnson_mapping.json` | Column names for Yeo-Johnson tranformed features|

### Visualizations
- **Missing Values**: Heatmaps, sector patterns, imputation comparisons
- **Outliers**: Detection comparisons, SHAP explanations, distribution plots
- **Transformations**: Before/after distributions, QQ plots
- **Performance**: Execution time breakdowns, memory usage

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/esg_eda

# Run specific test file
pytest tests/test_missing_values.py
```

### Code Quality
```bash
# Format code
black src/ scripts/

# Sort imports
isort src/ scripts/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Building Documentation
```bash
# Generate API documentation
sphinx-build -b html docs/source docs/build
```

## Best Practices

### Data Preparation
1. Ensure your input data has proper column names (no spaces, special characters)
2. Include a sector/category column for better analysis
3. Remove or handle infinite values before processing

### Memory Management
- For datasets > 1GB, consider using `chunk_size` parameter
- Monitor memory usage with system tools
- Use `--no-time-tracking` to reduce overhead

### Reproducibility
- Set random seeds via configuration
- Save intermediate results for debugging
- Use version control for your data files

## Troubleshooting

### Common Issues

**ImportError: No module named 'esg_eda'**
```bash
pip install -e .  # Install package in development mode
```

**MemoryError during processing**
```python
# Increase chunk size in configuration
settings.pipeline.chunk_size = 5000
```

**SHAP visualization errors**
- Update to latest SHAP version: `pip install shap --upgrade`
- Check NumPy compatibility



## Acknowledgments

- Built with scikit-learn, pandas, and the Python data science ecosystem
- SHAP for model interpretability
- Optuna for hyperparameter optimization
- Pydantic for configuration management

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{esg_score_eda,
  title = {ESG Score EDA Pipeline},
  author = {ESG EDA Team},
  year = {2024},
  url = {https://github.com/yourusername/esg-score-eda}
}
```