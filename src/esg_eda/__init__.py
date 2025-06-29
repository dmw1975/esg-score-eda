"""ESG Score EDA - A Modern Pipeline for ESG Data Analysis

This package provides a comprehensive pipeline for exploratory data analysis,
outlier detection, and feature engineering for ESG (Environmental, Social, 
and Governance) score data.
"""

__version__ = "0.2.0"
__author__ = "ESG EDA Team"
__email__ = "contact@example.com"

# Import key classes for easier access
from .analysis.missing_values import MissingValuesAnalyzer
from .analysis.outlier_detection import OutlierDetector
from .analysis.feature_engineering import FeatureEngineer
from .pipeline.orchestrator import ESGPipeline
from .core.config import get_settings

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "MissingValuesAnalyzer",
    "OutlierDetector",
    "FeatureEngineer",
    "ESGPipeline",
    "get_settings",
]