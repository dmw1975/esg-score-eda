"""Base classes for pipeline components."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import time
from datetime import datetime

import pandas as pd
import numpy as np

from .config import Settings, get_settings
from .exceptions import DataValidationError, PipelineError


class DataValidator:
    """Validates data according to specified rules."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 1) -> None:
        """Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            
        Raises:
            DataValidationError: If validation fails
        """
        if df is None:
            raise DataValidationError("DataFrame is None")
        
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Expected pandas DataFrame, got {type(df).__name__}"
            )
        
        if len(df) < min_rows:
            raise DataValidationError(
                f"DataFrame has {len(df)} rows, minimum required is {min_rows}"
            )
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise DataValidationError(
                    f"Missing required columns: {missing_columns}"
                )
    
    def validate_numeric_columns(self, df: pd.DataFrame, 
                                columns: List[str]) -> None:
        """Validate that specified columns are numeric.
        
        Args:
            df: DataFrame to validate
            columns: List of column names that should be numeric
            
        Raises:
            DataValidationError: If any column is not numeric
        """
        for col in columns:
            if col not in df.columns:
                raise DataValidationError(f"Column '{col}' not found in DataFrame")
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataValidationError(
                    f"Column '{col}' is not numeric, got {df[col].dtype}"
                )
    
    def check_missing_values(self, df: pd.DataFrame, 
                            threshold: Optional[float] = None) -> Dict[str, float]:
        """Check missing values in DataFrame.
        
        Args:
            df: DataFrame to check
            threshold: Maximum allowed missing percentage (0-1)
            
        Returns:
            Dictionary of column names to missing percentages
            
        Raises:
            DataValidationError: If any column exceeds threshold
        """
        threshold = threshold or self.settings.pipeline.missing_value_threshold
        missing_percentages = df.isnull().mean()
        
        # Check if any column exceeds threshold
        high_missing = missing_percentages[missing_percentages > threshold]
        if len(high_missing) > 0:
            self.logger.warning(
                f"Columns with missing values above threshold ({threshold}): "
                f"{high_missing.to_dict()}"
            )
        
        return missing_percentages.to_dict()


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, name: str, settings: Optional[Settings] = None):
        self.name = name
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.validator = DataValidator(self.settings)
        self._execution_time: Optional[float] = None
        self._start_time: Optional[float] = None
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the pipeline step.
        
        This method must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> None:
        """Validate inputs before processing.
        
        This method must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def validate_outputs(self, result: Any) -> None:
        """Validate outputs after processing.
        
        This method must be implemented by subclasses.
        """
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the pipeline step with validation and timing.
        
        This is the main entry point that handles:
        - Input validation
        - Execution timing
        - Output validation
        - Error handling
        """
        self.logger.info(f"Starting {self.name}")
        self._start_time = time.time()
        
        try:
            # Validate inputs
            self.validate_inputs(*args, **kwargs)
            
            # Run the step
            result = self.run(*args, **kwargs)
            
            # Validate outputs
            self.validate_outputs(result)
            
            # Record execution time
            self._execution_time = time.time() - self._start_time
            self.logger.info(
                f"Completed {self.name} in {self._execution_time:.2f} seconds"
            )
            
            return result
            
        except Exception as e:
            self._execution_time = time.time() - self._start_time
            self.logger.error(
                f"Failed {self.name} after {self._execution_time:.2f} seconds: {str(e)}"
            )
            
            # Re-raise as PipelineError if not already
            if not isinstance(e, PipelineError):
                raise PipelineError(
                    f"Pipeline step '{self.name}' failed: {str(e)}",
                    step_name=self.name,
                    original_error=str(e),
                    error_type=type(e).__name__
                )
            raise
    
    def get_execution_time(self) -> Optional[float]:
        """Get the execution time of the last run."""
        return self._execution_time
    
    def save_dataframe(self, df: pd.DataFrame, path: Path, **kwargs) -> None:
        """Save a DataFrame with logging."""
        self.logger.info(f"Saving DataFrame to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, **kwargs)
        self.logger.info(f"Saved {len(df)} rows to {path}")
    
    def load_dataframe(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load a DataFrame with validation."""
        self.logger.info(f"Loading DataFrame from {path}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        df = pd.read_csv(path, **kwargs)
        self.logger.info(f"Loaded {len(df)} rows from {path}")
        return df


class TransformerStep(PipelineStep):
    """Base class for data transformation steps."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TransformerStep":
        """Fit the transformer to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the data in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def run(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
            fit: bool = True) -> pd.DataFrame:
        """Execute the transformation step."""
        if fit:
            return self.fit_transform(X, y)
        else:
            return self.transform(X)


class VisualizationStep(PipelineStep):
    """Base class for visualization steps."""
    
    def __init__(self, name: str, settings: Optional[Settings] = None):
        super().__init__(name, settings)
        self.figures: List[Any] = []
    
    @abstractmethod
    def create_visualizations(self, df: pd.DataFrame, **kwargs) -> List[Path]:
        """Create visualizations and return paths to saved files."""
        pass
    
    def run(self, df: pd.DataFrame, **kwargs) -> List[Path]:
        """Execute the visualization step."""
        return self.create_visualizations(df, **kwargs)
    
    def save_figure(self, fig: Any, path: Path, dpi: int = 300, **kwargs) -> None:
        """Save a figure with logging."""
        self.logger.info(f"Saving figure to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
        self.logger.info(f"Saved figure to {path}")
        self.figures.append(fig)


class Pipeline:
    """Orchestrates multiple pipeline steps."""
    
    def __init__(self, steps: List[PipelineStep], settings: Optional[Settings] = None):
        self.steps = steps
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.execution_times: Dict[str, float] = {}
        self.results: Dict[str, Any] = {}
    
    def run(self, initial_data: Any, **kwargs) -> Dict[str, Any]:
        """Run all pipeline steps in sequence.
        
        Args:
            initial_data: Initial data to pass to first step
            **kwargs: Additional arguments passed to all steps
            
        Returns:
            Dictionary of step names to results
        """
        self.logger.info(f"Starting pipeline with {len(self.steps)} steps")
        start_time = time.time()
        
        data = initial_data
        for step in self.steps:
            try:
                # Execute step
                result = step.execute(data, **kwargs)
                
                # Store result and timing
                self.results[step.name] = result
                self.execution_times[step.name] = step.get_execution_time()
                
                # Use result as input for next step if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    data = result
                    
            except Exception as e:
                self.logger.error(f"Pipeline failed at step '{step.name}': {str(e)}")
                raise
        
        total_time = time.time() - start_time
        self.execution_times['total'] = total_time
        self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        
        return self.results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            'steps': len(self.steps),
            'completed': len(self.results),
            'execution_times': self.execution_times,
            'total_time': self.execution_times.get('total', 0),
            'timestamp': datetime.now().isoformat()
        }