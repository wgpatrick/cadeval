#!/usr/bin/env python3
"""
Unit tests for the config_loader module.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import yaml
import pytest

# Add the parent directory to the path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the MODULE instead of the specific classes
from scripts import config_loader

# Assume config_loader is importable relative to the tests directory
# or the project root is added to PYTHONPATH when running pytest
# from scripts.config_loader import Config, ConfigError # Keep original commented out

# --- Test Fixtures ---

@pytest.fixture
def valid_config_dict():
    """Provides a dictionary representing a valid config with prompts."""
    return {
        'prompts': {
            'default': "Default prompt: {description}",
            'concise': "Concise prompt: {description}"
        },
        'openscad': {
            'executable_path': '/path/to/openscad',
            'minimum_version': '2021.01',
            'render_timeout_seconds': 120
        },
        'directories': {
            'output': './results'
        },
        'llm': {
            'models': [{'name': 'test-model', 'provider': 'test'}]
        },
        'geometry_check': {
            'bounding_box_tolerance_mm': 1.0
        },
        'evaluation': { # Added evaluation section for completeness based on previous config edits
            'num_replicates': 2,
            'tasks_per_model': [],
            'models_per_task': []
        }
    }

@pytest.fixture
def config_missing_default_prompt_dict(valid_config_dict):
    """Config dict missing the default prompt key."""
    config = valid_config_dict.copy()
    config['prompts'] = valid_config_dict['prompts'].copy() # Ensure deep copy for nested dict
    del config['prompts']['default']
    return config

@pytest.fixture
def config_prompts_not_dict_dict(valid_config_dict):
    """Config dict where prompts section is not a dictionary."""
    config = valid_config_dict.copy()
    config['prompts'] = ["list", "not", "dict"]
    return config

@pytest.fixture
def config_missing_prompts_section_dict(valid_config_dict):
    """Config dict missing the entire prompts section."""
    config = valid_config_dict.copy()
    del config['prompts']
    return config

@pytest.fixture
def mock_config_file(tmp_path):
    """Factory fixture to create a temporary config file with given content."""
    def _create_file(config_content_dict):
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_content_dict, f)
        return str(config_path)
    return _create_file

# --- Test Cases ---

def test_config_load_success_with_prompts(mock_config_file, valid_config_dict):
    """Test Case 1: Successfully loads config with a valid prompts section."""
    config_path = mock_config_file(valid_config_dict)
    try:
        # Instantiate using the module reference
        config = config_loader.Config(config_path)
        assert isinstance(config.config_data.get('prompts'), dict)
        # The validation logic itself will be tested separately
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during config loading: {e}")


# These tests target the validation logic which needs to be implemented.
# Uncomment `@pytest.mark.xfail` if running tests before implementation.

# @pytest.mark.xfail(reason="Validation logic for prompts not yet implemented")
def test_validate_prompts_success(mock_config_file, valid_config_dict):
    """Test Case 2: Validation passes with a correct prompts section."""
    config_path = mock_config_file(valid_config_dict)
    try:
        # Instantiate using the module reference
        config_loader.Config(config_path)
    except config_loader.ConfigError as e:
        pytest.fail(f"Validation failed unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during validation: {e}")

# @pytest.mark.xfail(reason="Validation logic for prompts not yet implemented")
def test_validate_prompts_missing_default_raises(mock_config_file, config_missing_default_prompt_dict):
    """Test Case 3: Validation fails if default prompt key is missing."""
    config_path = mock_config_file(config_missing_default_prompt_dict)
    with pytest.raises(config_loader.ConfigError, match=r"Missing required key 'default' in 'prompts' section"):
        # Instantiate using the module reference
        config_loader.Config(config_path)

# @pytest.mark.xfail(reason="Validation logic for prompts not yet implemented")
def test_validate_prompts_not_dict_raises(mock_config_file, config_prompts_not_dict_dict):
    """Test Case 4: Validation fails if prompts section is not a dict."""
    config_path = mock_config_file(config_prompts_not_dict_dict)
    with pytest.raises(config_loader.ConfigError, match=r"Configuration section 'prompts' must be a dictionary"):
        # Instantiate using the module reference
        config_loader.Config(config_path)

# @pytest.mark.xfail(reason="Validation logic for missing prompts section not yet implemented")
def test_validate_prompts_section_missing_raises(mock_config_file, config_missing_prompts_section_dict):
    """Test Case 8: Validation should fail if the 'prompts' section is required and missing."""
    config_path = mock_config_file(config_missing_prompts_section_dict)
    # Assuming 'prompts' section becomes mandatory due to 'default' being required
    with pytest.raises(config_loader.ConfigError, match=r"Missing required configuration section: 'prompts'"):
        # Instantiate using the module reference
        config_loader.Config(config_path)


# These tests target the get_prompt method which needs to be implemented.
# Uncomment `@pytest.mark.xfail` if running tests before implementation.

# @pytest.mark.xfail(reason="get_prompt method not yet implemented")
def test_get_prompt_default(mock_config_file, valid_config_dict):
    """Test get_prompt returns the default prompt."""
    config_path = mock_config_file(valid_config_dict)
    # Instantiate using the module reference
    config = config_loader.Config(config_path)
    
    # Check if get_prompt method exists before calling
    if not hasattr(config, 'get_prompt'):
        pytest.fail("Config object does not have the 'get_prompt' method.")
    
    assert config.get_prompt("default") == "Default prompt: {description}"

def test_get_prompt_concise(mock_config_file, valid_config_dict):
    """Test get_prompt returns the concise prompt."""
    config_path = mock_config_file(valid_config_dict)
    # Instantiate using the module reference
    config = config_loader.Config(config_path)
    
    # Check if get_prompt method exists before calling
    if not hasattr(config, 'get_prompt'):
        pytest.fail("Config object does not have the 'get_prompt' method.")
    
    assert config.get_prompt("concise") == "Concise prompt: {description}"

def test_get_prompt_nonexistent(mock_config_file, valid_config_dict):
    """Test get_prompt returns None for a non-existent key."""
    config_path = mock_config_file(valid_config_dict)
    # Instantiate using the module reference
    config = config_loader.Config(config_path)
    
    # Check if get_prompt method exists before calling
    if not hasattr(config, 'get_prompt'):
        pytest.fail("Config object does not have the 'get_prompt' method.")
    
    assert config.get_prompt("non_existent") is None

def test_get_prompt_missing_section(mock_config_file, config_missing_prompts_section_dict):
    """Test get_prompt returns None when the prompts section is missing."""
    config_path = mock_config_file(config_missing_prompts_section_dict)
    
    # Patch _validate_config using the module reference
    with patch.object(config_loader.Config, '_validate_config', return_value=None):
        # Instantiate using the module reference
        config = config_loader.Config(config_path)
    
    # Check if get_prompt method exists before calling
    if not hasattr(config, 'get_prompt'):
        pytest.fail("Config object does not have the 'get_prompt' method.")
    
    assert config.get_prompt("default") is None
