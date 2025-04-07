#!/usr/bin/env python3
"""
Logger Setup for CadEval

This module provides a utility for configuring consistent logging across
the CadEval project components.
"""

import os
import sys
import logging
import logging.handlers
from typing import Optional, Union, List

# Default log directory within project
DEFAULT_LOG_DIR = "logs"

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_level: Optional[int] = None
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Name of the logger (usually __name__ from the calling module)
        level: Logging level for the console handler
        log_file: Path to the log file (if None, file logging is disabled)
        console: Whether to log to console
        log_format: Format string for log messages
        file_level: Logging level for the file handler (if None, uses level)
    
    Returns:
        A configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(min(level, file_level or level))  # Use the lowest level
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Clear any existing handlers safely
    for handler in logger.handlers[:]:
        # Optionally close handler before removing?
        # try: handler.close() 
        # except Exception: pass
        logger.removeHandler(handler)
    # logger.handlers = [] # Avoid direct assignment
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if a log file is specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level or level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def configure_project_logging(
    log_dir: str = DEFAULT_LOG_DIR,
    components: Optional[List[str]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Configure logging for all components of the CadEval project.
    
    Args:
        log_dir: Directory for log files (created if it doesn't exist)
        components: List of component names to configure loggers for
                   (if None, sets up only root logger)
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_format: Format string for log messages
    """
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_log_file = os.path.join(log_dir, "cadeval.log")
    setup_logger(
        name="",  # Root logger
        level=console_level,
        log_file=root_log_file,
        console=True,
        log_format=log_format,
        file_level=file_level
    )
    
    # Configure component loggers if specified
    if components:
        for component in components:
            component_log_file = os.path.join(log_dir, f"{component}.log")
            setup_logger(
                name=component,
                level=console_level,
                log_file=component_log_file,
                console=False,  # Don't add another console handler
                log_format=log_format,
                file_level=file_level
            )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name, creating it if it doesn't exist.
    
    Args:
        name: Name of the logger (usually __name__ from the calling module)
        
    Returns:
        A logger instance
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    # Example usage when run directly
    
    # First set up the project logging
    configure_project_logging(
        components=["config", "llm_api", "render", "geometry"]
    )
    
    # Get a logger for this module
    logger = get_logger("logger_test")
    
    # Log some test messages
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    print("\nLogging configuration successful!")
    print(f"Check the log files in the '{DEFAULT_LOG_DIR}' directory.") 