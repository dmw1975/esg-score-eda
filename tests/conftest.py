"""Shared fixtures and configuration for pytest."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Generator

from src.esg_eda.core import Settings


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """Create test settings with temporary directories."""
    settings = Settings(
        data=dict(
            base_dir=temp_dir,
            raw_data=temp_dir / "data" / "raw",
            processed_data=temp_dir / "data" / "processed",
            interim_data=temp_dir / "data" / "interim",
        ),
        visualization=dict(
            base_dir=temp_dir / "visualizations"
        ),
        logging=dict(
            log_dir=temp_dir / "logs"
        ),
        debug=True
    )
    settings.create_directories()
    return settings


@pytest.fixture
def sample_esg_data() -> pd.DataFrame:
    """Create sample ESG data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create base data
    data = {
        'company_id': range(n_samples),
        'gics_sector': np.random.choice(
            ['Energy', 'Materials', 'Industrials', 'Consumer Discretionary', 
             'Consumer Staples', 'Health Care', 'Financials', 'IT', 
             'Communication Services', 'Utilities', 'Real Estate'],
            n_samples
        ),
        'market_cap': np.random.lognormal(20, 2, n_samples),
        'revenue': np.random.lognormal(18, 2, n_samples),
        'employees': np.random.lognormal(8, 1.5, n_samples).astype(int),
    }
    
    # Add ESG scores with some missing values
    for category in ['environmental', 'social', 'governance']:
        scores = np.random.uniform(0, 100, n_samples)
        # Introduce missing values
        missing_mask = np.random.random(n_samples) < 0.1
        scores[missing_mask] = np.nan
        data[f'{category}_score'] = scores
    
    # Add some numerical features with outliers
    data['carbon_emissions'] = np.random.gamma(2, 2, n_samples) * 1000
    # Add some extreme outliers
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data['carbon_emissions'][outlier_indices] *= 10
    
    # Add categorical features
    data['rating'] = np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'], n_samples)
    data['country'] = np.random.choice(['USA', 'UK', 'Germany', 'France', 'Japan'], n_samples)
    
    df = pd.DataFrame(data)
    
    # Add some sector-specific missing patterns
    # IT companies have more missing environmental data
    it_mask = df['gics_sector'] == 'IT'
    it_missing = np.random.random(it_mask.sum()) < 0.3
    df.loc[it_mask, 'environmental_score'][it_missing] = np.nan
    
    return df


@pytest.fixture
def small_sample_data() -> pd.DataFrame:
    """Create a small sample dataset for quick tests."""
    data = {
        'company_id': [1, 2, 3, 4, 5],
        'gics_sector': ['IT', 'Energy', 'IT', 'Financials', 'Energy'],
        'environmental_score': [80.5, 65.3, np.nan, 72.1, 68.9],
        'social_score': [75.2, np.nan, 82.1, 79.5, 71.3],
        'governance_score': [88.1, 81.2, 85.5, np.nan, 83.7],
        'carbon_emissions': [1000, 5000, 800, 2000, 50000],  # Last one is outlier
        'rating': ['AA', 'BBB', 'A', 'AA', 'BB']
    }
    return pd.DataFrame(data)


@pytest.fixture
def numerical_columns() -> list:
    """List of numerical columns in test data."""
    return [
        'market_cap', 'revenue', 'employees', 
        'environmental_score', 'social_score', 'governance_score',
        'carbon_emissions'
    ]


@pytest.fixture
def categorical_columns() -> list:
    """List of categorical columns in test data."""
    return ['gics_sector', 'rating', 'country']


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )