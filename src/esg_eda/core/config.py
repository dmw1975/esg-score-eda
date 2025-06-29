"""Configuration management for ESG EDA pipeline using Pydantic."""

from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class DataPaths(BaseSettings):
    """Data path configuration."""
    
    base_dir: Path = Field(default=Path.cwd(), description="Base directory for the project")
    raw_data: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_data: Path = Field(default=Path("data/processed"), description="Processed data directory")
    interim_data: Path = Field(default=Path("data/interim"), description="Interim data directory")
    external_data: Path = Field(default=Path("data/external"), description="External data directory")
    
    # Specific data files
    input_file: str = Field(default="df_cleaned.csv", description="Main input file name")
    score_file: str = Field(default="score.csv", description="Score file name")
    
    @field_validator("raw_data", "processed_data", "interim_data", "external_data", mode='before')
    def resolve_paths(cls, v, info):
        """Resolve relative paths to absolute paths."""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute() and info.data.get("base_dir"):
            return info.data["base_dir"] / v
        return v
    
    model_config = {"env_prefix": "ESG_DATA_"}


class VisualizationPaths(BaseSettings):
    """Visualization path configuration."""
    
    base_dir: Path = Field(default=Path("visualizations"), description="Base visualization directory")
    missing_values: Path = Field(default=Path("missing_values"), description="Missing values visualizations")
    outliers: Path = Field(default=Path("outliers"), description="Outlier visualizations")
    features: Path = Field(default=Path("features"), description="Feature engineering visualizations")
    performance: Path = Field(default=Path("performance"), description="Performance visualizations")
    
    @field_validator("missing_values", "outliers", "features", "performance", mode='before')
    def resolve_paths(cls, v, info):
        """Resolve relative paths to absolute paths."""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute() and info.data.get("base_dir"):
            return info.data["base_dir"] / v
        return v
    
    model_config = {"env_prefix": "ESG_VIS_"}


class PipelineConfig(BaseSettings):
    """Pipeline configuration settings."""
    
    # Missing value handling
    missing_value_threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Threshold for dropping columns with missing values"
    )
    imputation_method: str = Field(
        default="sector_mean",
        description="Default imputation method",
        pattern="^(global_mean|sector_mean|sector_median|knn|mice)$"
    )
    
    # Outlier detection
    iqr_multiplier: float = Field(default=1.5, ge=0.0, description="IQR multiplier for outlier detection")
    z_score_threshold: float = Field(default=3.0, ge=0.0, description="Z-score threshold for outliers")
    isolation_forest_contamination: float = Field(
        default=0.1, 
        ge=0.0, 
        le=0.5,
        description="Contamination parameter for Isolation Forest"
    )
    
    # Feature engineering
    one_hot_max_categories: int = Field(
        default=50, 
        ge=1,
        description="Maximum categories for one-hot encoding"
    )
    yeo_johnson_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Skewness threshold for Yeo-Johnson transformation"
    )
    
    # Performance
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all cores)")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    chunk_size: int = Field(default=10000, ge=100, description="Chunk size for processing large datasets")
    
    model_config = {"env_prefix": "ESG_PIPELINE_"}


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format"
    )
    file_output: bool = Field(default=True, description="Enable file output")
    console_output: bool = Field(default=True, description="Enable console output")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    
    model_config = {"env_prefix": "ESG_LOG_"}


class Settings(BaseSettings):
    """Main settings class combining all configurations."""
    
    # Sub-configurations
    data: DataPaths = Field(default_factory=DataPaths)
    visualization: VisualizationPaths = Field(default_factory=VisualizationPaths)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # General settings
    project_name: str = Field(default="ESG Score EDA", description="Project name")
    version: str = Field(default="0.2.0", description="Project version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Model-specific data settings
    create_linear_model_data: bool = Field(
        default=True,
        description="Create optimized data for linear models"
    )
    create_tree_model_data: bool = Field(
        default=True,
        description="Create optimized data for tree-based models"
    )
    
    model_config = {
        "env_prefix": "ESG_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        # Data directories
        for attr_name in ["raw_data", "processed_data", "interim_data", "external_data"]:
            path = getattr(self.data, attr_name)
            path.mkdir(parents=True, exist_ok=True)
        
        # Visualization directories
        for attr_name in ["missing_values", "outliers", "features", "performance"]:
            path = getattr(self.visualization, attr_name)
            path.mkdir(parents=True, exist_ok=True)
        
        # Log directory
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_input_path(self) -> Path:
        """Get the full path to the input file."""
        return self.data.raw_data / self.data.input_file
    
    def get_score_path(self) -> Path:
        """Get the full path to the score file."""
        return self.data.processed_data / self.data.score_file
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict(exclude_unset=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.create_directories()
    return settings