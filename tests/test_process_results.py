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
        process_individual_result,
        calculate_meta_statistics
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

    def test_process_individual_result_new_fields(self):
        """Test processing of a single result entry with new fields."""
        # Test with the first mock entry (success case)
        entry = MOCK_RESULTS_DATA[0]

        processed = process_individual_result(entry, self.mock_config)

        # Check existence and values of new raw data fields
        self.assertIn('hausdorff_99p_distance', processed)
        self.assertEqual(processed['hausdorff_99p_distance'], 0.3)

        self.assertIn('reference_volume_mm3', processed)
        self.assertEqual(processed['reference_volume_mm3'], 1000.0)
        self.assertIn('generated_volume_mm3', processed)
        self.assertEqual(processed['generated_volume_mm3'], 1005.0)

        self.assertIn('reference_bbox_mm', processed)
        self.assertEqual(processed['reference_bbox_mm'], [10.0, 10.0, 10.0])
        self.assertIn('generated_bbox_aligned_mm', processed)
        self.assertEqual(processed['generated_bbox_aligned_mm'], [10.1, 10.0, 9.9])

        # Check boolean check statuses derived from 'checks'
        self.assertIn('individual_check_statuses', processed)
        self.assertTrue(processed['individual_check_statuses'].get('check_volume_passed'))
        self.assertTrue(processed['individual_check_statuses'].get('check_hausdorff_passed'))
        
        # Check derived overall flags (ensure they reflect new checks if logic includes them)
        # Assuming overall_pipeline_success requires all checks including new ones
        self.assertTrue(processed.get('all_geometry_checks_passed'), "Expected all checks passed for mock entry 0")
        self.assertTrue(processed.get('chamfer_check_passed'), "Chamfer check should pass for mock entry 0")
        self.assertTrue(processed.get('overall_pipeline_success'), "Expected overall success for mock entry 0")

        # Check formatted detail fields
        self.assertIn('hausdorff_99p_distance_detail', processed)
        self.assertEqual(processed['hausdorff_99p_distance_detail'], "0.3000") # Assuming 4 decimal places

        self.assertIn('volume_reference_detail', processed)
        self.assertEqual(processed['volume_reference_detail'], "1000.00") # Assuming 2 decimal places
        self.assertIn('volume_generated_detail', processed)
        self.assertEqual(processed['volume_generated_detail'], "1005.00") # Assuming 2 decimal places

        self.assertIn('bbox_reference_detail', processed)
        self.assertEqual(processed['bbox_reference_detail'], "10.00, 10.00, 10.00") # Assuming formatting
        self.assertIn('bbox_generated_aligned_detail', processed)
        self.assertEqual(processed['bbox_generated_aligned_detail'], "10.10, 10.00, 9.90") # Assuming formatting
        
        # Test with the second mock entry (failure case)
        entry_fail = MOCK_RESULTS_DATA[1]
        processed_fail = process_individual_result(entry_fail, self.mock_config)
        
        self.assertFalse(processed_fail['individual_check_statuses'].get('check_volume_passed'))
        self.assertFalse(processed_fail['individual_check_statuses'].get('check_hausdorff_passed'))
        self.assertFalse(processed_fail.get('all_geometry_checks_passed'), "Expected all checks failed for mock entry 1")
        self.assertFalse(processed_fail.get('overall_pipeline_success'), "Expected overall failure for mock entry 1")
        self.assertEqual(processed_fail['hausdorff_99p_distance_detail'], "0.6000")
        self.assertEqual(processed_fail['volume_reference_detail'], "500.00")
        self.assertEqual(processed_fail['volume_generated_detail'], "555.00")

    def test_calculate_meta_statistics_new_aggs(self):
        """Test calculation of meta statistics including new aggregates."""
        # Process all mock data first
        processed_data = [process_individual_result(entry, self.mock_config) for entry in MOCK_RESULTS_DATA]

        meta_stats = calculate_meta_statistics(processed_data)

        # --- Check stats for model_A ---
        self.assertIn("model_A", meta_stats)
        stats_a = meta_stats["model_A"]

        # Counts
        self.assertEqual(stats_a['total_tasks'], 2)
        self.assertEqual(stats_a['scad_gen_success_count'], 2)
        self.assertEqual(stats_a['render_success_count'], 2) # Based on check_render_successful
        self.assertEqual(stats_a['geo_check_run_success_count'], 2) # Both checks ran (no major error)
        self.assertEqual(stats_a['all_geo_checks_passed_count'], 1) # Only task 1 passed all
        self.assertEqual(stats_a['overall_pipeline_success_count'], 1) # Only task 1 fully succeeded

        # New check counts
        self.assertEqual(stats_a['volume_passed_count'], 1)
        self.assertEqual(stats_a['hausdorff_passed_count'], 1)

        # Rates (relative to checks run where applicable)
        # volume_check_pass_rate_rel = 1 / 2 * 100 = 50.0
        self.assertAlmostEqual(stats_a['volume_check_pass_rate_rel'], 50.0)
        # hausdorff_check_pass_rate_rel = 1 / 2 * 100 = 50.0
        self.assertAlmostEqual(stats_a['hausdorff_check_pass_rate_rel'], 50.0)
        # all_geo_checks_passed_rate_rel = 1 / 2 * 100 = 50.0
        self.assertAlmostEqual(stats_a['all_geo_checks_passed_rate_rel'], 50.0)
        # overall_pipeline_success_rate = 1 / 2 * 100 = 50.0
        self.assertAlmostEqual(stats_a['overall_pipeline_success_rate'], 50.0)


        # Averages/Medians/Stdev for new metrics (calculated on runs where check occurred)
        # Hausdorff: [0.3, 0.6] -> avg=0.45, median=0.45, stdev=sqrt(((0.3-0.45)^2 + (0.6-0.45)^2)/1) = sqrt(0.0225 + 0.0225) = sqrt(0.045) approx 0.212
        self.assertAlmostEqual(stats_a['average_hausdorff_99p_distance'], 0.45)
        self.assertAlmostEqual(stats_a['median_hausdorff_99p_distance'], 0.45)
        self.assertAlmostEqual(stats_a['stdev_hausdorff_99p_distance'], statistics.stdev([0.3, 0.6]))

        # Volume Diff %:
        # Task 1: |1005 - 1000| / 1000 * 100 = 0.5%
        # Task 2: |555 - 500| / 500 * 100 = 55 / 500 * 100 = 11.0%
        # Values: [0.5, 11.0] -> avg=5.75, median=5.75, stdev=sqrt(((0.5-5.75)^2 + (11.0-5.75)^2)/1) = sqrt((-5.25)^2 + (5.25)^2) = sqrt(27.5625 * 2) = sqrt(55.125) approx 7.4246
        self.assertAlmostEqual(stats_a['average_volume_diff_percent'], 5.75)
        self.assertAlmostEqual(stats_a['median_volume_diff_percent'], 5.75)
        self.assertAlmostEqual(stats_a['stdev_volume_diff_percent'], statistics.stdev([0.5, 11.0]))

        # Check Chamfer average (existing metric)
        # Chamfer: [0.15, 0.25] -> avg=0.20, median=0.20, stdev=sqrt(((0.15-0.2)^2 + (0.25-0.2)^2)/1) = sqrt(0.0025 + 0.0025) = sqrt(0.005) approx 0.0707
        self.assertAlmostEqual(stats_a['average_chamfer_distance'], 0.20)
        self.assertAlmostEqual(stats_a['median_chamfer_distance'], 0.20)
        self.assertAlmostEqual(stats_a['stdev_chamfer_distance'], statistics.stdev([0.15, 0.25]))


        # --- Check stats for model_B ---
        self.assertIn("model_B", meta_stats)
        stats_b = meta_stats["model_B"]
        self.assertEqual(stats_b['total_tasks'], 1)
        self.assertEqual(stats_b['scad_gen_success_count'], 1)
        self.assertEqual(stats_b['render_success_count'], 0) # Failed render
        self.assertEqual(stats_b['geo_check_run_success_count'], 0) # Checks likely skipped
        self.assertEqual(stats_b['all_geo_checks_passed_count'], 0)
        self.assertEqual(stats_b['overall_pipeline_success_count'], 0)
        self.assertEqual(stats_b['volume_passed_count'], 0) # New
        self.assertEqual(stats_b['hausdorff_passed_count'], 0) # New

        # Rates should be 0 or None
        self.assertEqual(stats_b['volume_check_pass_rate_rel'], 0)
        self.assertEqual(stats_b['hausdorff_check_pass_rate_rel'], 0)
        self.assertEqual(stats_b['overall_pipeline_success_rate'], 0)

        # Averages should be None as no successful checks ran
        self.assertIsNone(stats_b['average_hausdorff_99p_distance'])
        self.assertIsNone(stats_b['average_volume_diff_percent'])
        self.assertIsNone(stats_b['average_chamfer_distance'])


if __name__ == '__main__':
    unittest.main() 