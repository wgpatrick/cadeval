#!/usr/bin/env python3
"""
Unit and Integration tests for the geometry_check module.

Requires trimesh and open3d libraries.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest import mock
import numpy as np # Required for similarity checks

# Add the parent directory to the path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure scripts directory is also in path if needed directly
scripts_dir = os.path.join(parent_dir, 'scripts')
if scripts_dir not in sys.path:
     sys.path.insert(0, scripts_dir)

# Mock config loader early if geometry_check imports it at module level
# sys.modules['scripts.config_loader'] = mock.MagicMock()

# Try importing the module under test
try:
    from scripts.geometry_check import (
        perform_geometry_checks, 
        check_render_success,
        check_watertight,
        check_single_component,
        check_bounding_box,
        check_similarity,
        GeometryCheckError
    )
    from scripts.config_loader import get_config, Config
except ImportError as e:
    print(f"Failed to import geometry_check or dependencies: {e}")
    print("Ensure trimesh, open3d, and other script dependencies are installed.")
    sys.exit(1)

# Disable logging during tests unless debugging
import logging
# logging.basicConfig(level=logging.DEBUG) # Uncomment for debug logs
logging.disable(logging.CRITICAL)


# Helper function to create mock rendering info
def create_mock_rendering_info(status="Success", summary_path=None):
    return {
        "status": status,
        "summary_path": summary_path
    }

# Helper function to create mock task requirements
def create_mock_task_requirements(bbox=[10, 10, 10], expected_components=1):
    req = {
        "bounding_box": bbox
    }
    if expected_components is not None:
        req['topology_requirements'] = {'expected_component_count': expected_components}
    return req


class TestGeometryChecks(unittest.TestCase):
    """Test cases for geometry checking functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources once for the class."""
        cls.config = get_config() # Load real config for tolerance etc.
        
        # Define paths to test STL files (relative to project root)
        cls.test_data_dir = os.path.join(parent_dir, "tests", "test_data", "stl")
        cls.ref_cube_stl = os.path.join(cls.test_data_dir, "ref_cube.stl")
        cls.gen_cube_ident_stl = os.path.join(cls.test_data_dir, "gen_cube_identical.stl")
        cls.gen_cube_slight_diff_stl = os.path.join(cls.test_data_dir, "gen_cube_slight_diff.stl")
        cls.gen_sphere_diff_stl = os.path.join(cls.test_data_dir, "gen_sphere_diff.stl")
        cls.gen_non_watertight_stl = os.path.join(cls.test_data_dir, "gen_non_watertight.stl")
        cls.gen_multi_comp_stl = os.path.join(cls.test_data_dir, "gen_multi_component.stl")
        
        # Corresponding summary paths (assuming they exist)
        cls.ref_cube_summary = os.path.join(cls.test_data_dir, "ref_cube_summary.json")
        cls.gen_cube_ident_summary = os.path.join(cls.test_data_dir, "gen_cube_identical_summary.json")

        # Basic check that reference file exists
        if not os.path.exists(cls.ref_cube_stl):
             raise FileNotFoundError(f"Reference test STL not found: {cls.ref_cube_stl}")

    def setUp(self):
        """Set up per-test resources if needed."""
        # Example: Copy files to a temp location if tests modify them
        pass 

    def tearDown(self):
        """Clean up per-test resources if needed."""
        pass 

    # --- Tests for Individual Check Functions --- 

    def test_check_render_success(self):
        """Test Check 1: Render Success logic."""
        self.assertTrue(check_render_success("Success"))
        self.assertFalse(check_render_success("Compile Error"))
        self.assertFalse(check_render_success("Timeout"))
        self.assertFalse(check_render_success("Failed"))
        self.assertFalse(check_render_success("Unknown"))

    def test_check_watertight_good(self):
        """Test Check 2: Watertight check on a good cube."""
        is_watertight, error = check_watertight(self.ref_cube_stl)
        self.assertTrue(is_watertight)
        self.assertIsNone(error)

    def test_check_watertight_bad(self):
        """Test Check 2: Watertight check on a non-watertight model."""
        # The non-watertight model might still load, but is_watertight should be false
        is_watertight, error = check_watertight(self.gen_non_watertight_stl)
        self.assertFalse(is_watertight)
        self.assertIsNone(error) # Expect loading to succeed

    def test_check_watertight_invalid_stl(self):
         """Test Check 2: Watertight check on a non-existent file."""
         # Create a path to a non-existent file
         non_existent_path = os.path.join(self.test_data_dir, "no_such_file.stl")
         is_watertight, error = check_watertight(non_existent_path)
         self.assertIsNone(is_watertight)
         self.assertIsNotNone(error)
         self.assertIn("Failed to load mesh", error)

    # --- Add tests for check_single_component, check_bounding_box, check_similarity --- 
    
    # --- Tests for perform_geometry_checks (Orchestrator) --- 
    
    # def test_perform_checks_success_path(self):
    #     """Test the main orchestrator function on an identical model."""
    #     pass
        
    # def test_perform_checks_render_failed(self):
    #     """Test orchestrator when rendering failed."""
    #     pass

if __name__ == "__main__":
    unittest.main() 