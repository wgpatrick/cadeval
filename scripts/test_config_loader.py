#!/usr/bin/env python3
"""
Unit tests for the config_loader module.
"""

import os
import sys
import yaml
import tempfile
import unittest
import logging
from unittest.mock import patch

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from scripts.config_loader import Config, ConfigError, get_config

# Disable logging for tests
logging.disable(logging.CRITICAL)


class TestConfigLoader(unittest.TestCase):
    """Test cases for the config_loader module."""
    
    def setUp(self):
        """Set up test environment with sample configuration files."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid configuration file
        valid_config = {
            "openscad": {
                "executable_path": "/usr/bin/openscad",
                "minimum_version": "2021.01",
                "render_timeout_seconds": 120
            },
            "directories": {
                "tasks": "./tasks",
                "reference": "./reference",
                "generated_outputs": "./outputs",
                "results": "./results",
                "output": "./outputs"  # Required by current config_loader
            },
            "llm": {
                "models": [
                    {
                        "name": "test-model",
                        "provider": "test",
                        "temperature": 0.5,
                        "max_tokens": 1000
                    }
                ]
            },
            "geometry_check": {
                "bounding_box_tolerance_mm": 0.5
            }
        }
        
        self.valid_config_path = os.path.join(self.temp_dir, "valid_config.yaml")
        with open(self.valid_config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        # Create an invalid YAML file
        self.invalid_config_path = os.path.join(self.temp_dir, "invalid_config.yaml")
        with open(self.invalid_config_path, 'w') as f:
            f.write("openscad: {executable_path: '/usr/bin/openscad', incomplete_json")
        
        # Create an incomplete configuration file (missing required sections/keys)
        incomplete_config = {
            "openscad": {
                "executable_path": "/usr/bin/openscad",
                "minimum_version": "2021.01",
                "render_timeout_seconds": 120
            },
            # Missing 'directories' section
            "llm": {
                "models": []
            },
            "geometry_check": {
                "bounding_box_tolerance_mm": 0.5
            }
        }
        
        self.incomplete_config_path = os.path.join(self.temp_dir, "incomplete_config.yaml")
        with open(self.incomplete_config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        # Path to a non-existent file
        self.nonexistent_path = os.path.join(self.temp_dir, "nonexistent.yaml")
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config = Config(self.valid_config_path)
        self.assertIsNotNone(config.config_data)
        self.assertEqual(config.config_path, self.valid_config_path)
        self.assertEqual(config.get("openscad.executable_path"), "/usr/bin/openscad")
        self.assertEqual(config.get("directories.output"), "./outputs")
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        with self.assertRaises(ConfigError) as context:
            Config(self.nonexistent_path)
        self.assertIn("not found", str(context.exception))
    
    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        with self.assertRaises(ConfigError) as context:
            Config(self.invalid_config_path)
        self.assertIn("parsing YAML", str(context.exception))
    
    def test_load_incomplete_config(self):
        """Test loading an incomplete configuration file."""
        with self.assertRaises(ConfigError) as context:
            Config(self.incomplete_config_path)
        self.assertIn("Missing required configuration section", str(context.exception))
    
    def test_get_config_values(self):
        """Test retrieving configuration values with dot notation."""
        config = Config(self.valid_config_path)
        
        # Test getting existing values
        self.assertEqual(config.get("openscad.executable_path"), "/usr/bin/openscad")
        self.assertEqual(config.get("directories.output"), "./outputs")
        self.assertEqual(config.get("geometry_check.bounding_box_tolerance_mm"), 0.5)
        
        # Test getting nested values
        model = config.get("llm.models")[0]
        self.assertEqual(model["name"], "test-model")
        self.assertEqual(model["provider"], "test")
        
        # Test getting non-existent values with default
        self.assertEqual(config.get("nonexistent.key", "default_value"), "default_value")
        self.assertIsNone(config.get("nonexistent.key"))
    
    def test_get_required_config(self):
        """Test retrieving required configuration values."""
        config = Config(self.valid_config_path)
        
        # Test getting existing required values
        self.assertEqual(config.get_required("openscad.executable_path"), "/usr/bin/openscad")
        
        # Test getting non-existent required values
        with self.assertRaises(ConfigError) as context:
            config.get_required("nonexistent.key")
        self.assertIn("not found", str(context.exception))
    
    def test_singleton_get_config(self):
        """Test the singleton get_config function."""
        # Get config instance with valid path
        config1 = get_config(self.valid_config_path)
        self.assertIsNotNone(config1)
        
        # Get another instance with the same path - should be the same object
        config2 = get_config(self.valid_config_path)
        self.assertIs(config1, config2)
        
        # Try with the incomplete config - should raise an error
        with self.assertRaises(ConfigError):
            get_config(self.incomplete_config_path)


if __name__ == "__main__":
    unittest.main() 