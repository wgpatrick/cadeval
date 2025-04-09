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
from dotenv import load_dotenv
import logging

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.logger_setup import get_logger, setup_logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

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
            if not isinstance(self.config_data, dict):
                logger.error(f"Configuration file {self.config_path} does not contain a valid YAML dictionary.")
                raise ConfigError(f"Configuration file {self.config_path} is invalid: Root element must be a dictionary.")
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
        if not isinstance(self.config_data, dict):
            # This check might be redundant if _load_config already raises, but good for safety
            raise ConfigError("Configuration data is not a dictionary.")

        required_sections = ['openscad', 'directories', 'llm', 'geometry_check', 'evaluation', 'prompts']
        for section in required_sections:
            if section not in self.config_data:
                raise ConfigError(f"Missing required configuration section: '{section}'")
            if not isinstance(self.config_data[section], dict):
                # Allow prompts to be validated more specifically below
                if section != 'prompts':
                    raise ConfigError(f"Configuration section '{section}' must be a dictionary.")

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
        
        # Validate 'prompts' section
        prompts_section = self.config_data.get('prompts')
        if not isinstance(prompts_section, dict):
            raise ConfigError("Configuration section 'prompts' must be a dictionary.")
        if 'default' not in prompts_section:
            raise ConfigError("Missing required key 'default' in 'prompts' section.")
        # Optional: Validate that prompts are strings?
        for key, value in prompts_section.items():
            if not isinstance(value, str):
                logger.warning(f"Value for prompt key '{key}' in 'prompts' section is not a string. Found type: {type(value)}")
                # Decide if this should be a warning or a ConfigError based on requirements
                # raise ConfigError(f"Prompt value for key '{key}' must be a string.")

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
    
    def get_api_key(self, provider: str) -> str:
        """
        Get the API key for a specified LLM provider from environment variables.
        
        Args:
            provider: The LLM provider name (e.g., 'openai', 'anthropic', 'google')
            
        Returns:
            The API key as a string
            
        Raises:
            ConfigError: If the API key is not found or is empty
        """
        # Map provider names to environment variable names
        provider_env_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY'
        }
        
        if provider.lower() not in provider_env_map:
            error_msg = f"Unknown LLM provider: {provider}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        env_var_name = provider_env_map[provider.lower()]
        api_key = os.getenv(env_var_name)
        
        if not api_key or api_key == f"your_{provider.lower()}_api_key_here":
            error_msg = f"API key for {provider} not found in environment variables"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        return api_key

    def get_prompt(self, key: str) -> str:
        """
        Retrieves a prompt string from the 'prompts' section by its key.

        Args:
            key (str): The key of the prompt to retrieve (e.g., 'default', 'concise').

        Returns:
            str or None: The prompt string if the key exists and the value is a string,
                         otherwise None. Returns None if the 'prompts' section doesn't exist.
        """
        prompts_section = self.config_data.get('prompts')
        if isinstance(prompts_section, dict):
            prompt_value = prompts_section.get(key)
            # Ensure the retrieved value is actually a string before returning
            if isinstance(prompt_value, str):
                return prompt_value
            elif prompt_value is not None:
                logger.warning(f"Prompt key '{key}' found but value is not a string (type: {type(prompt_value)}). Returning None.")
                return None
            else:
                # Key not found within the prompts dict
                logger.debug(f"Prompt key '{key}' not found in 'prompts' section.")
                return None
        else:
            # Prompts section doesn't exist or is not a dict
            logger.warning(f"Attempted to get prompt key '{key}' but 'prompts' section is missing or not a dictionary.")
            return None


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
        
        # Test API key access (will fail with template values)
        try:
            print("\nTesting API key access:")
            for provider in ['openai', 'anthropic', 'google']:
                try:
                    # Just check if we can access, don't print actual keys
                    api_key = config.get_api_key(provider)
                    masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
                    print(f"  {provider.capitalize()} API key found: {masked_key}")
                except ConfigError as e:
                    print(f"  {provider.capitalize()} API key: Not configured")
        except Exception as e:
            print(f"Error testing API keys: {e}")
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}")
        sys.exit(1) 