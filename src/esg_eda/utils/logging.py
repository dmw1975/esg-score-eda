"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
from rich.console import Console

from ..core import Settings, get_settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console_output: Optional[bool] = None,
    file_output: Optional[bool] = None,
    settings: Optional[Settings] = None
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console_output: Enable console output
        file_output: Enable file output
        settings: Settings object (if not provided, uses get_settings())
    """
    settings = settings or get_settings()
    
    # Use provided values or fall back to settings
    level = level or settings.logging.level
    console_output = console_output if console_output is not None else settings.logging.console_output
    file_output = file_output if file_output is not None else settings.logging.file_output
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler with rich formatting
    if console_output:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=settings.debug
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_file is None:
            log_file = settings.logging.log_dir / "esg_eda.log"
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(settings.logging.format)
        )
        root_logger.addHandler(file_handler)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, console={console_output}, file={file_output}")
    if file_output:
        logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary logging configuration changes."""
    
    def __init__(self, level: Optional[str] = None, suppress: bool = False):
        self.level = level
        self.suppress = suppress
        self.original_level = None
        self.original_handlers = None
    
    def __enter__(self):
        """Enter the context."""
        root_logger = logging.getLogger()
        
        if self.level:
            self.original_level = root_logger.level
            root_logger.setLevel(self.level)
        
        if self.suppress:
            self.original_handlers = root_logger.handlers[:]
            root_logger.handlers = []
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        root_logger = logging.getLogger()
        
        if self.original_level is not None:
            root_logger.setLevel(self.original_level)
        
        if self.original_handlers is not None:
            root_logger.handlers = self.original_handlers