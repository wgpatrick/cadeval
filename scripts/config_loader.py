#!/usr/bin/env python3
"""
Configuration Loader for CadEval

This module provides utilities for loading and accessing the YAML configuration
file used throughout the CadEval project.
"""

import os
import sys
import yaml
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.logger_setup import get_logger, setup_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Global configuration instance (singleton)
_config_instance = None

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class Config:
    """
    Configuration loader and validator for CadEval.
    
    This class handles loading the YAML configuration file and provides
    methods to access configuration values.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Config object.
        
        Args:
            config_path: Path to the configuration YAML file.
        
        Raises:
            ConfigError: If the configuration file can't be loaded or is invalid.
        """
        self.config_path = config_path
        self.config_data = None
        
        # Load the configuration file
        self._load_config()
        
        # Validate required config sections
        self._validate_config()
        
        logger.info(f"Configuration loaded successfully from {config_path}")
    
    def _load_config(self) -> None:
        """
        Load the configuration from the YAML file.
        
        Raises:
            ConfigError: If the file can't be found or parsed.
        """
        try:
            with open(self.config_path, 'r') as config_file:
                self.config_data = yaml.safe_load(config_file)
        except FileNotFoundError:
            error_msg = f"Configuration file not found: {self.config_path}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML configuration: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    def _validate_config(self) -> None:
        """
        Validate the configuration to ensure all required sections are present.
        
        Raises:
            ConfigError: If any required section is missing.
        """
        # Define required sections
        required_sections = ['openscad', 'directories', 'llm', 'geometry_check']
        
        # Check if each required section exists
        for section in required_sections:
            if section not in self.config_data:
                error_msg = f"Missing required configuration section: {section}"
                logger.error(error_msg)
                raise ConfigError(error_msg)
        
        # Validate specific required keys within sections
        openscad_required = ['executable_path', 'minimum_version', 'render_timeout_seconds']
        for key in openscad_required:
            if key not in self.config_data['openscad']:
                error_msg = f"Missing required key in openscad section: {key}"
                logger.error(error_msg)
                raise ConfigError(error_msg)
        
        # Directories should have 'output' defined
        if 'output' not in self.config_data['directories']:
            error_msg = "Missing 'output' directory in directories section"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        # LLM section should have models defined
        if 'models' not in self.config_data['llm'] or not self.config_data['llm']['models']:
            error_msg = "No LLM models defined in configuration"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        # Validate geometry check has required keys
        if 'bounding_box_tolerance_mm' not in self.config_data['geometry_check']:
            error_msg = "Missing bounding_box_tolerance_mm in geometry_check section"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        logger.debug("Configuration validation successful")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value
                     (e.g., 'openscad.executable_path')
            default: Default value to return if key doesn't exist
        
        Returns:
            The configuration value, or the default if not found
        """
        if not self.config_data:
            error_msg = "Configuration not loaded"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Configuration key not found: {key_path}, using default: {default}")
            return default
    
    def get_required(self, key_path: str) -> Any:
        """
        Get a required configuration value.
        
        Args:
            key_path: Dot-separated path to the configuration value
        
        Returns:
            The configuration value
            
        Raises:
            ConfigError: If the key doesn't exist
        """
        value = self.get(key_path)
        if value is None:
            error_msg = f"Required configuration key not found: {key_path}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        return value
    
    def resolve_path(self, key_path: str, must_exist: bool = False) -> str:
        """
        Resolve a path from the configuration.
        
        Args:
            key_path: Path to the configuration key containing the path
            must_exist: If True, raises an error if the path doesn't exist
        
        Returns:
            The resolved absolute path
            
        Raises:
            ConfigError: If path is invalid or doesn't exist when must_exist is True
        """
        path = self.get_required(key_path)
        
        # If the path is not absolute, make it relative to the config file directory
        if not os.path.isabs(path):
            config_dir = os.path.dirname(os.path.abspath(self.config_path))
            path = os.path.join(config_dir, path)
        
        # Normalize the path
        path = os.path.normpath(path)
        
        # Check if the path exists if required
        if must_exist and not os.path.exists(path):
            error_msg = f"Path does not exist: {path} (from {key_path})"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        return path


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get a global Config instance (singleton pattern).
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        A Config instance
    """
    global _config_instance
    
    if _config_instance is None or _config_instance.config_path != config_path:
        logger.debug(f"Creating new Config instance with path: {config_path}")
        _config_instance = Config(config_path)
    
    return _config_instance


if __name__ == "__main__":
    # Set up logging for this script when run directly
    setup_logger(__name__, log_file="logs/config_loader.log")
    
    try:
        # Load the configuration
        config = get_config()
        
        # Print some configuration values
        print("Configuration loaded successfully!")
        print(f"OpenSCAD executable: {config.get('openscad.executable_path')}")
        print(f"Output directory: {config.get('directories.output')}")
        print(f"LLM model: {config.get('llm.models')[0]['name']}")
        print(f"Bounding box tolerance: {config.get('geometry_check.bounding_box_tolerance_mm')} mm")
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}")
        sys.exit(1) 