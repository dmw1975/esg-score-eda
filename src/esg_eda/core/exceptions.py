"""Custom exceptions for the ESG EDA pipeline."""

from typing import Optional, Dict, Any


class ESGEDAError(Exception):
    """Base exception for all ESG EDA errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(ESGEDAError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, column: Optional[str] = None, 
                 validation_type: Optional[str] = None, **kwargs):
        details = {"column": column, "validation_type": validation_type}
        details.update(kwargs)
        super().__init__(message, details)


class PipelineError(ESGEDAError):
    """Raised when a pipeline step fails."""
    
    def __init__(self, message: str, step_name: Optional[str] = None, 
                 error_type: Optional[str] = None, **kwargs):
        details = {"step_name": step_name, "error_type": error_type}
        details.update(kwargs)
        super().__init__(message, details)


class ConfigurationError(ESGEDAError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 expected_type: Optional[str] = None, **kwargs):
        details = {"config_key": config_key, "expected_type": expected_type}
        details.update(kwargs)
        super().__init__(message, details)


class FileNotFoundError(ESGEDAError):
    """Raised when a required file is not found."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        details = {"file_path": file_path}
        details.update(kwargs)
        super().__init__(message, details)


class DataTypeError(DataValidationError):
    """Raised when data type is incorrect."""
    
    def __init__(self, message: str, column: str, expected_type: str, 
                 actual_type: str, **kwargs):
        super().__init__(
            message, 
            column=column,
            validation_type="data_type",
            expected_type=expected_type,
            actual_type=actual_type,
            **kwargs
        )


class MissingDataError(DataValidationError):
    """Raised when required data is missing."""
    
    def __init__(self, message: str, column: str, missing_percentage: float, **kwargs):
        super().__init__(
            message,
            column=column,
            validation_type="missing_data",
            missing_percentage=missing_percentage,
            **kwargs
        )


class OutlierDetectionError(PipelineError):
    """Raised when outlier detection fails."""
    
    def __init__(self, message: str, method: str, **kwargs):
        super().__init__(
            message,
            step_name="outlier_detection",
            error_type="detection_failure",
            method=method,
            **kwargs
        )