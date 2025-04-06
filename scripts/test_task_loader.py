#!/usr/bin/env python3
"""
Unit tests for the task_loader module.
"""

import os
import sys
import unittest
import tempfile
import shutil
import yaml
import json
import pytest
from unittest.mock import patch, mock_open
from pydantic import ValidationError

# Add the parent directory to the path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Adjusted import relative to project root
from scripts.task_loader import load_tasks, TaskLoadError, load_schema, validate_task
from scripts.config_loader import Config, get_config # Assuming get_config loads schema path

# Disable logging for tests to keep output clean
import logging
logging.disable(logging.CRITICAL)


class TestTaskLoader(unittest.TestCase):
    """Test cases for the task_loader module."""
    
    def setUp(self):
        """Set up test environment with sample task files."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid task file
        self.valid_task = {
            "task_id": "test_task1",
            "description": "A simple test task for unit testing.",
            "reference_stl": "./reference/test_task1.stl",
            "requirements": {
                "bounding_box": [10.0, 10.0, 10.0],
                "topology_requirements": {
                    "expected_component_count": 1
                }
            }
        }
        
        self.valid_task_path = os.path.join(self.temp_dir, "test_task1.yaml")
        with open(self.valid_task_path, 'w') as f:
            yaml.dump(self.valid_task, f)
        
        # Create another valid task file (to test multiple task loading)
        self.valid_task2 = {
            "task_id": "test_task2",
            "description": "A second test task for unit testing.",
            "reference_stl": "./reference/test_task2.stl",
            "requirements": {
                "bounding_box": [20.0, 20.0, 5.0]
                # No topology_requirements to test optional fields
            }
        }
        
        self.valid_task2_path = os.path.join(self.temp_dir, "test_task2.yaml")
        with open(self.valid_task2_path, 'w') as f:
            yaml.dump(self.valid_task2, f)
            
        # Create an invalid YAML file
        self.invalid_yaml_path = os.path.join(self.temp_dir, "invalid_task.yaml")
        with open(self.invalid_yaml_path, 'w') as f:
            f.write("task_id: 'invalid'\ndescription: 'This is not valid YAML")
        
        # Create a task file missing the task_id field
        self.missing_id_task = {
            "description": "A task missing the task_id field.",
            "reference_stl": "./reference/missing_id.stl",
            "requirements": {
                "bounding_box": [10.0, 10.0, 10.0]
            }
        }
        
        self.missing_id_path = os.path.join(self.temp_dir, "missing_id.yaml")
        with open(self.missing_id_path, 'w') as f:
            yaml.dump(self.missing_id_task, f)
            
        # Create an empty task file
        self.empty_task_path = os.path.join(self.temp_dir, "empty.yaml")
        with open(self.empty_task_path, 'w') as f:
            f.write("")
        
        # Create a non-YAML file that should be ignored
        self.non_yaml_path = os.path.join(self.temp_dir, "not_a_task.txt")
        with open(self.non_yaml_path, 'w') as f:
            f.write("This is not a YAML file")
        
        # Create an empty directory for testing empty directory handling
        self.empty_dir = os.path.join(self.temp_dir, "empty_dir")
        os.makedirs(self.empty_dir)
        
        # Path to a non-existent directory
        self.nonexistent_dir = os.path.join(self.temp_dir, "does_not_exist")
    
    def tearDown(self):
        """Clean up temporary files and directories."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_tasks(self):
        """Test loading valid task files."""
        tasks = load_tasks(self.temp_dir)
        
        # Filter out tasks from files we created that should be skipped
        valid_tasks = [task for task in tasks if task["task_id"] in ["test_task1", "test_task2"]]
        
        # Should have loaded both valid task files
        self.assertEqual(len(valid_tasks), 2)
        
        # Verify the tasks were loaded correctly
        task1 = next((task for task in valid_tasks if task["task_id"] == "test_task1"), None)
        task2 = next((task for task in valid_tasks if task["task_id"] == "test_task2"), None)
        
        self.assertIsNotNone(task1)
        self.assertIsNotNone(task2)
        
        # Check the content of task1
        self.assertEqual(task1["description"], "A simple test task for unit testing.")
        self.assertEqual(task1["reference_stl"], "./reference/test_task1.stl")
        self.assertEqual(task1["requirements"]["bounding_box"], [10.0, 10.0, 10.0])
        self.assertEqual(task1["requirements"]["topology_requirements"]["expected_component_count"], 1)
        
        # Check the content of task2
        self.assertEqual(task2["description"], "A second test task for unit testing.")
        self.assertEqual(task2["reference_stl"], "./reference/test_task2.stl")
        self.assertEqual(task2["requirements"]["bounding_box"], [20.0, 20.0, 5.0])
        # task2 should not have topology_requirements
        self.assertNotIn("topology_requirements", task2["requirements"])
    
    def test_empty_directory(self):
        """Test handling of an empty directory (no task files)."""
        tasks = load_tasks(self.empty_dir)
        
        # Should return an empty list, not None or raise an exception
        self.assertEqual(tasks, [])
    
    def test_nonexistent_directory(self):
        """Test handling of a non-existent directory."""
        with self.assertRaises(TaskLoadError) as context:
            load_tasks(self.nonexistent_dir)
        
        # Error message should mention the directory
        self.assertIn("directory not found", str(context.exception).lower())
        self.assertIn(self.nonexistent_dir, str(context.exception))
    
    def test_invalid_yaml(self):
        """Test handling of a file with invalid YAML syntax."""
        # Create a directory with only the invalid YAML file
        invalid_dir = os.path.join(self.temp_dir, "invalid_dir")
        os.makedirs(invalid_dir)
        
        invalid_file = os.path.join(invalid_dir, "invalid.yaml")
        shutil.copy(self.invalid_yaml_path, invalid_file)
        
        # Should not raise an exception, but return an empty list
        tasks = load_tasks(invalid_dir)
        self.assertEqual(tasks, [])
    
    def test_missing_task_id(self):
        """Test handling of a task file with missing task_id."""
        # Create a directory with only the file missing task_id
        missing_id_dir = os.path.join(self.temp_dir, "missing_id_dir")
        os.makedirs(missing_id_dir)
        
        missing_id_file = os.path.join(missing_id_dir, "missing_id.yaml")
        shutil.copy(self.missing_id_path, missing_id_file)
        
        # Should not raise an exception, but return an empty list
        tasks = load_tasks(missing_id_dir)
        self.assertEqual(tasks, [])
    
    def test_empty_task_file(self):
        """Test handling of an empty task file."""
        # Create a directory with only the empty task file
        empty_file_dir = os.path.join(self.temp_dir, "empty_file_dir")
        os.makedirs(empty_file_dir)
        
        empty_file = os.path.join(empty_file_dir, "empty.yaml")
        shutil.copy(self.empty_task_path, empty_file)
        
        # Should not raise an exception, but return an empty list
        tasks = load_tasks(empty_file_dir)
        self.assertEqual(tasks, [])
    
    def test_mixed_files(self):
        """Test loading from a directory with a mix of valid and invalid files."""
        # Create a directory with a mix of file types
        mixed_dir = os.path.join(self.temp_dir, "mixed_dir")
        os.makedirs(mixed_dir)
        
        # Copy one valid and various invalid files
        shutil.copy(self.valid_task_path, os.path.join(mixed_dir, "valid.yaml"))
        shutil.copy(self.invalid_yaml_path, os.path.join(mixed_dir, "invalid.yaml"))
        shutil.copy(self.missing_id_path, os.path.join(mixed_dir, "missing_id.yaml"))
        shutil.copy(self.empty_task_path, os.path.join(mixed_dir, "empty.yaml"))
        shutil.copy(self.non_yaml_path, os.path.join(mixed_dir, "not_yaml.txt"))
        
        # Should only load the valid task
        tasks = load_tasks(mixed_dir)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "test_task1")
    
    def test_schema_validation_valid(self):
        """Test schema validation with a valid task."""
        # Create a test schema
        schema = {
            "type": "object",
            "required": ["task_id", "description", "reference_stl", "requirements"],
            "properties": {
                "task_id": {"type": "string"},
                "description": {"type": "string"},
                "reference_stl": {"type": "string"},
                "requirements": {
                    "type": "object",
                    "required": ["bounding_box"],
                    "properties": {
                        "bounding_box": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        # Valid task should pass validation
        self.assertTrue(validate_task(self.valid_task, schema))
    
    def test_schema_validation_invalid(self):
        """Test schema validation with invalid tasks."""
        # Create a test schema
        schema = {
            "type": "object",
            "required": ["task_id", "description", "reference_stl", "requirements"],
            "properties": {
                "task_id": {"type": "string"},
                "description": {"type": "string"},
                "reference_stl": {"type": "string"},
                "requirements": {
                    "type": "object",
                    "required": ["bounding_box"],
                    "properties": {
                        "bounding_box": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        # Missing required field
        invalid_task1 = {
            "task_id": "invalid1",
            "description": "Missing reference_stl field",
            "requirements": {
                "bounding_box": [10.0, 10.0, 10.0]
            }
        }
        self.assertFalse(validate_task(invalid_task1, schema))
        
        # Wrong type for bounding_box (string instead of array)
        invalid_task2 = {
            "task_id": "invalid2",
            "description": "Wrong type for bounding_box",
            "reference_stl": "./reference/invalid2.stl",
            "requirements": {
                "bounding_box": "10x10x10"
            }
        }
        self.assertFalse(validate_task(invalid_task2, schema))
        
        # Wrong number of items in bounding_box
        invalid_task3 = {
            "task_id": "invalid3",
            "description": "Wrong number of items in bounding_box",
            "reference_stl": "./reference/invalid3.stl",
            "requirements": {
                "bounding_box": [10.0, 10.0]  # Only 2 dimensions
            }
        }
        self.assertFalse(validate_task(invalid_task3, schema))
    
    def test_load_tasks_with_validation(self):
        """Test loading tasks with schema validation."""
        # Create a directory for this test
        validation_dir = os.path.join(self.temp_dir, "validation_dir")
        os.makedirs(validation_dir)
        
        # Create a schema file
        schema_path = os.path.join(self.temp_dir, "test_schema.json")
        schema = {
            "type": "object",
            "required": ["task_id", "description", "reference_stl", "requirements"],
            "properties": {
                "task_id": {"type": "string"},
                "description": {"type": "string"},
                "reference_stl": {"type": "string"},
                "requirements": {
                    "type": "object",
                    "required": ["bounding_box"],
                    "properties": {
                        "bounding_box": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "number"}
                        }
                    }
                }
            }
        }
        with open(schema_path, 'w') as f:
            json.dump(schema, f)
        
        # Create a valid task file
        valid_path = os.path.join(validation_dir, "valid.yaml")
        shutil.copy(self.valid_task_path, valid_path)
        
        # Create an invalid task file (missing reference_stl)
        invalid_task = {
            "task_id": "invalid",
            "description": "Missing reference_stl field",
            "requirements": {
                "bounding_box": [10.0, 10.0, 10.0]
            }
        }
        invalid_path = os.path.join(validation_dir, "invalid.yaml")
        with open(invalid_path, 'w') as f:
            yaml.dump(invalid_task, f)
        
        # Load tasks with validation
        tasks = load_tasks(validation_dir, validate=True, schema_path=schema_path)
        
        # Should only load the valid task
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "test_task1")
        
        # Load without validation should load both
        tasks_no_validation = load_tasks(validation_dir, validate=False)
        self.assertEqual(len(tasks_no_validation), 2)


if __name__ == "__main__":
    unittest.main() 