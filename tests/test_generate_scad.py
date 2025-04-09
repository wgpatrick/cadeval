#!/usr/bin/env python3
"""
Unit tests for the generate_scad module.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call
import pytest

# Add the parent directory to the path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the functions we want to test
from scripts.generate_scad import generate_scad_for_task, extract_scad_code
from scripts.config_loader import Config

# Disable logging for tests to keep output clean
import logging
logging.disable(logging.CRITICAL)


class TestGenerateScad:
    """Test cases for generate_scad.py."""
    
    def test_generate_scad_for_task_uses_prompt_key(self, monkeypatch):
        """Test that generate_scad_for_task uses config.get_prompt with correct prompt_key."""
        # Create mock objects
        mock_config = MagicMock(spec=Config)
        mock_llm_client = MagicMock()
        mock_client_creator = MagicMock(return_value=mock_llm_client)
        
        # Configure the mock to return a specific prompt template
        mock_config.get_prompt = MagicMock(return_value="Test prompt template: {description}")
        
        # Set up the mock to return successful generation
        mock_llm_client.generate_text.return_value = "cube([10, 10, 10]);"
        
        # Patch necessary functions
        monkeypatch.setattr("scripts.generate_scad.config", mock_config)
        monkeypatch.setattr("scripts.generate_scad.create_llm_client", mock_client_creator)
        
        # Create mock task and model config
        task = {
            "task_id": "test_task",
            "description": "Create a simple cube."
        }
        model_config = {
            "provider": "mock_provider",
            "name": "mock_model",
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        # Call the function with a specific prompt_key
        output_dir = "test_output"
        prompt_key = "test_prompt"
        result = generate_scad_for_task(
            task, 
            model_config, 
            output_dir=output_dir, 
            prompt_key=prompt_key, 
            replicate_id=1
        )
        
        # Verify the function called get_prompt with the correct key
        mock_config.get_prompt.assert_called_once_with(prompt_key)
        
        # Verify the LLM client was called with the formatted prompt
        expected_prompt = "Test prompt template: Create a simple cube."
        mock_llm_client.generate_text.assert_called_once_with(expected_prompt)
        
        # Check that the result includes the correct prompt_key and prompt_used
        assert result["prompt_key"] == prompt_key
        assert result["prompt_used"] == expected_prompt
        assert result["success"] is True
        
    def test_generate_scad_handles_missing_prompt_key(self, monkeypatch):
        """Test that generate_scad_for_task handles a missing prompt key gracefully."""
        # Create mock objects
        mock_config = MagicMock(spec=Config)
        
        # Configure the mock to return None for non-existent prompt key
        mock_config.get_prompt = MagicMock(return_value=None)
        
        # Patch necessary functions
        monkeypatch.setattr("scripts.generate_scad.config", mock_config)
        
        # Create mock task and model config
        task = {
            "task_id": "test_task",
            "description": "Create a simple cube."
        }
        model_config = {
            "provider": "mock_provider",
            "name": "mock_model"
        }
        
        # Call the function with a non-existent prompt_key
        result = generate_scad_for_task(
            task, 
            model_config, 
            output_dir="test_output", 
            prompt_key="non_existent_key"
        )
        
        # Verify the function called get_prompt with the correct key
        mock_config.get_prompt.assert_called_once_with("non_existent_key")
        
        # Check that the result includes an error about the missing prompt key
        assert result["success"] is False
        assert "not found in configuration" in result["error"]
        
    def test_extract_scad_code_from_markdown(self):
        """Test extracting code from a markdown code block."""
        response = """
Here's the OpenSCAD code for a simple cube:

```openscad
// Simple cube model
cube([10, 10, 10]);
```

I hope this helps!
"""
        code = extract_scad_code(response)
        assert code.strip() == "// Simple cube model\ncube([10, 10, 10]);"
        
    def test_extract_scad_code_from_raw_code(self):
        """Test extracting code when it's not in a code block."""
        response = """// Simple model
cube([10, 10, 10]);"""
        
        code = extract_scad_code(response)
        assert code.strip() == "// Simple model\ncube([10, 10, 10]);" 