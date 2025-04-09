#!/usr/bin/env python3
"""
Unit tests for the process_results.py script.
"""

import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock
import logging
import numpy as np # For potential np.isinf checks if needed
import yaml # Added import
import statistics # Added import

# Add project root to sys.path to allow importing process_results
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions to be tested
try:
    from scripts.process_results import (
        process_data_for_dashboard,
    )
    from scripts.config_loader import Config # For threshold checks
except ImportError as e:
    print(f"Failed to import process_results or dependencies: {e}")
    sys.exit(1)

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Mock Config class for tests
class MockConfig:
    def __init__(self, config_dict):
        self._config = config_dict

    def get(self, key, default=None):
        # Basic implementation for testing, assumes keys exist or default is used
        parts = key.split('.')
        value = self._config
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
            
    def get_required(self, key):
        # Basic implementation, assumes keys exist
        parts = key.split('.')
        value = self._config
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
             raise KeyError(f"Missing required key: {key}")


# Sample data mimicking results.json entries, including new fields
MOCK_RESULTS_DATA = [
    {
        "task_id": "task1",
        "model_name": "model_A",
        "output_scad_path": "/path/t1_mA.scad",
        "output_stl_path": "/path/t1_mA.stl",
        "generation_error": None,
        "render_status": "Success",
        "check_error": None,
        "checks": {
            "check_render_successful": True,
            "check_is_watertight": True,
            "check_is_single_component": True,
            "check_bounding_box_accurate": True,
            "check_volume_passed": True, # New
            "check_hausdorff_passed": True # New
        },
        "geometric_similarity_distance": 0.15,
        "icp_fitness_score": 0.99,
        "hausdorff_95p_distance": 0.3, # Added based on 99p for test
        "hausdorff_99p_distance": 0.3, # New
        "reference_volume_mm3": 1000.0, # New
        "generated_volume_mm3": 1005.0, # New
        "reference_bbox_mm": [10.0, 10.0, 10.0], # New
        "generated_bbox_aligned_mm": [10.1, 10.0, 9.9] # New
    },
    {
        "task_id": "task2",
        "model_name": "model_A",
        "output_scad_path": "/path/t2_mA.scad",
        "output_stl_path": "/path/t2_mA.stl",
        "generation_error": None,
        "render_status": "Success",
        "check_error": "Volume mismatch",
        "checks": {
            "check_render_successful": True,
            "check_is_watertight": True,
            "check_is_single_component": True,
            "check_bounding_box_accurate": True,
            "check_volume_passed": False, # New - Failed
            "check_hausdorff_passed": False # New - Failed
        },
        "geometric_similarity_distance": 0.25,
        "icp_fitness_score": 0.98,
        "hausdorff_95p_distance": 0.6, # Added based on 99p for test
        "hausdorff_99p_distance": 0.6, # New - Above threshold
        "reference_volume_mm3": 500.0, # New
        "generated_volume_mm3": 555.0, # New - > 1% diff
        "reference_bbox_mm": [8.0, 8.0, 8.0], # New
        "generated_bbox_aligned_mm": [8.1, 7.9, 8.0] # New
    },
    {
        "task_id": "task3",
        "model_name": "model_B",
        "output_scad_path": "/path/t3_mB.scad",
        "output_stl_path": None, # Render Failed
        "generation_error": None,
        "render_status": "Timeout",
        "check_error": "Geometry checks skipped due to render failure",
        "checks": { # Checks might be partially populated or all None/False
             "check_render_successful": False
        },
        "geometric_similarity_distance": None,
        "icp_fitness_score": None,
        "hausdorff_95p_distance": None, # Added None explicitly
        "hausdorff_99p_distance": None, # New
        "reference_volume_mm3": None, # New
        "generated_volume_mm3": None, # New
        "reference_bbox_mm": None, # New
        "generated_bbox_aligned_mm": None # New
    }
]

class TestProcessResults(unittest.TestCase):
    """Tests for the process_results script functions."""

    def setUp(self):
        """Set up test resources."""
        # Load actual config for thresholds, etc.
        config_path = os.path.join(project_root, 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                actual_config_data = yaml.safe_load(f)
            if actual_config_data is None:
                 raise ValueError("config.yaml is empty or invalid")
            self.mock_config = MockConfig(actual_config_data)
        except FileNotFoundError:
            self.fail(f"Test setup failed: config.yaml not found at {config_path}")
        except Exception as e:
            self.fail(f"Test setup failed: Error loading config.yaml: {e}")

    def test_process_dashboard_data_new_fields(self):
        """Test processing of result entries into dashboard format with new fields."""
        # Call the new processing function once
        dashboard_data_list = process_data_for_dashboard(MOCK_RESULTS_DATA, self.mock_config)

        # Check we got the expected number of entries processed
        self.assertEqual(len(dashboard_data_list), len(MOCK_RESULTS_DATA))

        # --- Test with the first mock entry (success case) ---
        processed = dashboard_data_list[0]

        # Check existence and values of key processed fields
        self.assertIn('haus_95p_dist', processed) # Check for new key
        # Assuming 4 decimal places from fmt function
        self.assertEqual(processed['haus_95p_dist'], '0.3000') # Value from MOCK_RESULTS_DATA[0]['hausdorff_99p_distance'] assuming 95p==99p in this mock for simplicity

        self.assertIn('haus_99p_dist', processed)
        self.assertEqual(processed['haus_99p_dist'], '0.3000')

        self.assertIn('ref_vol', processed)
        self.assertEqual(processed['ref_vol'], '1000.00') # Assuming 2 decimal places
        self.assertIn('gen_vol', processed)
        self.assertEqual(processed['gen_vol'], '1005.00') # Assuming 2 decimal places

        # Check boolean check statuses derived by process_data_for_dashboard
        self.assertIn('check_volume_passed', processed)
        self.assertTrue(processed.get('check_volume_passed'))
        self.assertIn('check_hausdorff_passed', processed)
        self.assertTrue(processed.get('check_hausdorff_passed')) # Should be true as 0.3 <= 0.5 threshold

        # Check overall status
        self.assertTrue(processed.get('overall_passed'), "Expected overall success for mock entry 0")


        # --- Test with the second mock entry (failure case) ---
        processed_fail = dashboard_data_list[1]

        self.assertIn('haus_95p_dist', processed_fail)
        self.assertEqual(processed_fail['haus_95p_dist'], '0.6000') # Assuming 95p == 99p in mock

        self.assertIn('haus_99p_dist', processed_fail)
        self.assertEqual(processed_fail['haus_99p_dist'], '0.6000')

        self.assertIn('check_volume_passed', processed_fail)
        self.assertFalse(processed_fail.get('check_volume_passed')) # 555 vs 500 > 1% threshold
        self.assertIn('check_hausdorff_passed', processed_fail)
        # Update assertion: 0.6 <= actual threshold of 1.0, so this should PASS
        self.assertTrue(processed_fail.get('check_hausdorff_passed'))

        # Check overall passed - should still be False due to volume check failure
        self.assertFalse(processed_fail.get('overall_passed'), "Expected overall failure for mock entry 1")

        self.assertEqual(processed_fail['ref_vol'], "500.00")
        self.assertEqual(processed_fail['gen_vol'], "555.00")

        # --- Test with the third mock entry (render fail case) ---
        processed_render_fail = dashboard_data_list[2]

        self.assertIn('haus_95p_dist', processed_render_fail)
        self.assertEqual(processed_render_fail['haus_95p_dist'], 'N/A') # Should be N/A as checks didn't run
        self.assertIn('haus_99p_dist', processed_render_fail)
        self.assertEqual(processed_render_fail['haus_99p_dist'], 'N/A')

        self.assertIsNone(processed_render_fail.get('check_volume_passed')) # Check status should be None
        self.assertIsNone(processed_render_fail.get('check_hausdorff_passed')) # Check status should be None

        self.assertFalse(processed_render_fail.get('overall_passed'), "Expected overall failure for mock entry 2 (render fail)")


if __name__ == '__main__':
    unittest.main() 