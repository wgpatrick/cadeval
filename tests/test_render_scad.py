#!/usr/bin/env python3
"""
Integration tests for the render_scad module.

Requires a valid OpenSCAD installation and configuration.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest import mock

# Add the parent directory to the path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure scripts directory is also in path if needed directly
scripts_dir = os.path.join(parent_dir, 'scripts')
if scripts_dir not in sys.path:
     sys.path.insert(0, scripts_dir)

from scripts.render_scad import render_scad_file, render_all_scad, validate_openscad_config, RenderError
from scripts.config_loader import get_config, Config, ConfigError

# Disable logging during tests unless debugging
import logging
# logging.basicConfig(level=logging.DEBUG) # Uncomment for debug logs
logging.disable(logging.CRITICAL)


@unittest.skipIf(
    not os.path.exists(get_config().get('openscad.executable_path', '')),
    "OpenSCAD executable not found, skipping render tests."
)
class TestRenderScad(unittest.TestCase):
    """Test cases for OpenSCAD rendering script."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources once for the class."""
        cls.config = get_config() # Load main config
        # Perform validation once to ensure OpenSCAD is usable
        try:
            validate_openscad_config(cls.config)
        except RenderError as e:
             raise unittest.SkipTest(f"OpenSCAD validation failed: {e}. Skipping tests.")
        
        # Define paths to test SCAD files (relative to project root)
        cls.test_data_dir = os.path.join(parent_dir, "tests", "test_data", "scad")
        cls.good_scad = os.path.join(cls.test_data_dir, "good_cube.scad")
        cls.bad_scad = os.path.join(cls.test_data_dir, "bad_syntax.scad")
        cls.long_scad = os.path.join(cls.test_data_dir, "long_render.scad")
        cls.nonexistent_scad = os.path.join(cls.test_data_dir, "does_not_exist.scad")

    def setUp(self):
        """Set up a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="cadeval_render_test_")

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    # --- Test Cases for render_scad_file --- 

    def test_render_good_file(self):
        """Test rendering a known-good SCAD file."""
        result = render_scad_file(self.good_scad, self.temp_dir, self.config)
        
        self.assertEqual(result["status"], "Success")
        self.assertEqual(result["return_code"], 0)
        self.assertIsNotNone(result["stl_path"])
        self.assertTrue(os.path.exists(result["stl_path"]))
        self.assertGreater(os.path.getsize(result["stl_path"]), 0)
        # Check for summary file - might depend on OpenSCAD version supporting it
        if result["summary_path"]:
             self.assertTrue(os.path.exists(result["summary_path"]))
             self.assertGreater(os.path.getsize(result["summary_path"]), 0)
        self.assertIsNone(result["error"])

    def test_render_bad_syntax(self):
        """Test rendering a SCAD file with syntax errors."""
        result = render_scad_file(self.bad_scad, self.temp_dir, self.config)
        
        self.assertEqual(result["status"], "Compile Error")
        self.assertNotEqual(result["return_code"], 0)
        self.assertIsNone(result["stl_path"])
        # STL file should not exist or be empty
        output_stl_path = os.path.join(self.temp_dir, "bad_syntax.stl")
        self.assertFalse(os.path.exists(output_stl_path) and os.path.getsize(output_stl_path) > 0)
        self.assertIsNotNone(result["error"])
        self.assertIsNotNone(result["stderr"])
        # Check for common error indicators in stderr
        self.assertIn("ERROR:", result["stderr"].upper())

    # --- Placeholder for more tests (timeout, file not found, etc.) --- 
    
    @mock.patch('scripts.render_scad.get_config') # Mock get_config used within render_scad_file
    def test_render_timeout(self, mock_get_config):
        """Test rendering a potentially long-running file with a short timeout."""
        # Create a mock config object that returns a short timeout
        mock_config_instance = mock.MagicMock(spec=Config)
        # Set up side effect for config.get
        def mock_config_get_side_effect(key, default=None):
            if key == 'openscad.executable_path':
                 return self.config.get_required('openscad.executable_path') # Use real path
            if key == 'openscad.render_timeout_seconds':
                 return 0.1 # Very short timeout
            # Return real values for other keys if needed, or defaults
            return self.config.get(key, default)
        
        mock_config_instance.get.side_effect = mock_config_get_side_effect
        mock_config_instance.get_required.side_effect = lambda key: mock_config_get_side_effect(key, None) # Simple required mock
        # Ensure config_path attribute exists for relative path resolution if needed
        mock_config_instance.config_path = self.config.config_path 

        # Make the mocked get_config return our specific mock instance
        mock_get_config.return_value = mock_config_instance
        
        # Run render with the long-running file and the mocked config
        result = render_scad_file(self.long_scad, self.temp_dir, mock_config_instance)
        
        self.assertEqual(result["status"], "Timeout")
        self.assertIsNone(result["stl_path"])
        self.assertIsNone(result["summary_path"])
        self.assertIsNotNone(result["error"])
        self.assertIn("timed out after 0.1 seconds", result["error"])
        # Duration might be slightly over 0.1 due to process overhead
        self.assertGreaterEqual(result.get("duration", 0), 0.1)
        
    def test_render_scad_not_found(self):
        """Test rendering a non-existent SCAD file."""
        result = render_scad_file(self.nonexistent_scad, self.temp_dir, self.config)
        
        # OpenSCAD should return an error code when the input file doesn't exist
        self.assertEqual(result["status"], "Compile Error") # Or potentially "Failed" depending on exact exit
        self.assertNotEqual(result["return_code"], 0)
        self.assertIsNone(result["stl_path"])
        self.assertIsNone(result["summary_path"])
        self.assertIsNotNone(result["error"])
        self.assertIsNotNone(result["stderr"])
        # Check stderr for common file not found indicators
        stderr_upper = result["stderr"].upper()
        # Temporarily print stderr to see the actual message
        # print(f"\nDEBUG: stderr for non-existent file:\n{result['stderr']}\n") 
        # print(f"DEBUG: error message for non-existent file:\n{result['error']}\n")
        
        # Assert that stderr is empty or whitespace for this error type
        self.assertTrue(result['stderr'].strip() == "")
        # Assert that the error message mentions the exit code
        self.assertIn("exit code 1", result.get("error", ""))
        
    @mock.patch('scripts.render_scad.get_config')
    def test_openscad_executable_not_found(self, mock_get_config):
        """Test rendering when OpenSCAD executable path is invalid."""
        # Mock config to return an invalid executable path
        mock_config_instance = mock.MagicMock(spec=Config)
        invalid_path = "/invalid/path/to/nonexistent_openscad_executable"
        
        def mock_config_get_side_effect(key, default=None):
            if key == 'openscad.executable_path':
                 return invalid_path
            # Use real config for other values if needed
            return self.config.get(key, default)
            
        mock_config_instance.get.side_effect = mock_config_get_side_effect
        mock_config_instance.get_required.side_effect = lambda key: mock_config_get_side_effect(key, None)
        mock_config_instance.config_path = self.config.config_path
        mock_get_config.return_value = mock_config_instance
        
        # Run render with the good file and the mocked config
        # We expect this to fail *before* subprocess due to the helper function check, 
        # but the test covers the subprocess FileNotFoundError case as well.
        result = render_scad_file(self.good_scad, self.temp_dir, mock_config_instance)
        
        # The error should be caught during command construction or execution
        self.assertEqual(result["status"], "Failed")
        self.assertIsNone(result["stl_path"])
        self.assertIsNone(result["summary_path"])
        self.assertIsNotNone(result["error"])
        # Check if the error message contains the invalid path
        self.assertIn(invalid_path, result["error"])
        self.assertIn("not found", result["error"].lower())

# --- Placeholder for TestRenderAllScad class --- 

class TestCommandConstruction(unittest.TestCase):
    """Test cases for OpenSCAD command construction logic."""
    
    def test_command_basic(self):
        """Test basic command construction with default config values."""
        # Mock config object
        mock_config = mock.MagicMock(spec=Config)
        
        # Mock get_required method
        mock_config.get_required = mock.MagicMock(side_effect=lambda key: {
            'openscad.executable_path': '/path/to/openscad'
        }.get(key)) # Must mock the method itself
        
        # Mock get method
        mock_config.get = mock.MagicMock(side_effect = lambda key, default=None: {
            'openscad.executable_path': '/path/to/openscad',
            'openscad.export_format': 'asciistl',
            'openscad.backend': None, # Default backend
            'openscad.summary_options': 'all'
        }.get(key, default))
        
        mock_config.config_path = '/path/to/config.yaml' # Needed for potential relative path logic
        
        # Inputs
        scad_path = 'input/model.scad'
        stl_path = 'output/model.stl'
        summary_path = 'output/model_summary.json'
        
        # Call the helper function (need to import it)
        from scripts.render_scad import _build_openscad_command 
        command = _build_openscad_command(scad_path, stl_path, summary_path, mock_config)
        
        # Expected command
        expected = [
            '/path/to/openscad', '-q', 
            '--export-format', 'asciistl',
            '--summary', 'all',
            '--summary-file', summary_path,
            '-o', stl_path,
            scad_path
        ]
        
        self.assertEqual(command, expected)
        
    def test_command_with_backend(self):
        """Test command construction when backend is specified."""
        mock_config = mock.MagicMock(spec=Config)
        
        # Mock get_required method
        mock_config.get_required = mock.MagicMock(side_effect=lambda key: {
            'openscad.executable_path': '/usr/bin/openscad'
        }.get(key)) # Must mock the method itself
        
        # Mock get method
        mock_config.get = mock.MagicMock(side_effect = lambda key, default=None: {
            'openscad.executable_path': '/usr/bin/openscad',
            'openscad.export_format': 'binstl',
            'openscad.backend': 'Manifold', # Specific backend
            'openscad.summary_options': 'geometry'
        }.get(key, default))
        
        mock_config.config_path = '/fake/config.yaml'
        
        scad_path = 'a.scad'
        stl_path = 'b.stl'
        summary_path = 'c.json'
        
        from scripts.render_scad import _build_openscad_command 
        command = _build_openscad_command(scad_path, stl_path, summary_path, mock_config)
        
        expected = [
            '/usr/bin/openscad', '-q', 
            '--export-format', 'binstl',
            '--backend', 'Manifold', # Backend should be included
            '--summary', 'geometry',
            '--summary-file', summary_path,
            '-o', stl_path,
            scad_path
        ]
        
        self.assertEqual(command, expected)

# --- Placeholder for TestRenderAllScad class --- 

if __name__ == "__main__":
    unittest.main() 