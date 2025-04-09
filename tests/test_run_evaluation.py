#!/usr/bin/env python3
"""
Unit tests for the run_evaluation.py script.
"""

import unittest
import argparse
import os
import sys
from unittest.mock import patch, MagicMock, call, create_autospec
import logging
import io # Import io for StringIO
import pytest # Import pytest

# Add project root to sys.path to allow importing run_evaluation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the function to be tested
# We need to import the script carefully to avoid running its main block if any
# Using importlib might be safer, but direct import is simpler for now
# Assuming run_evaluation.py can be imported without side effects at module level
from scripts.run_evaluation import parse_arguments

# Import the main function itself and Config class for mocking
from scripts.run_evaluation import main
from scripts.config_loader import Config, ConfigError

# --- Helper function for creating mock config --- Start ---
def create_mock_config(config_values):
    """Creates a mock Config object that handles get and get_required."""
    # NO MOCKING of ConfigError here. We will use the actual one.

    # Create a proper mock Config that inherits behaviors
    mock_config = MagicMock(spec=Config)
    
    def mock_get(key, default=None):
        # Split the key by dots to handle nested access
        parts = key.split('.')
        current = config_values
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    
    def mock_get_required(key):
        # Split the key by dots to handle nested access
        parts = key.split('.')
        current = config_values
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Raise the ACTUAL ConfigError
                raise ConfigError(f"Mock missing required config key: {key}")
        if current is None:
            # Raise the ACTUAL ConfigError
            raise ConfigError(f"Mock missing required config key: {key}")
        return current
    
    # Explicitly assign the methods to the mock attributes
    mock_config.get = mock_get
    mock_config.get_required = mock_get_required
    
    # Add the evaluation.num_replicates to avoid the error
    if 'evaluation' not in config_values:
        config_values['evaluation'] = {}
    if 'num_replicates' not in config_values.get('evaluation', {}):
        config_values['evaluation']['num_replicates'] = 1
    
    mock_config.config_path = '/fake/path/config.yaml'
    
    # Add the get_prompt method to all mock configs
    def mock_get_prompt(key):
        prompts = config_values.get('prompts', {})
        return prompts.get(key)
    
    mock_config.get_prompt = mock_get_prompt
    
    return mock_config
# --- Helper function for creating mock config --- End ---

# Define sample data
SAMPLE_TASKS = [
    {"task_id": "task1", "description": "desc1", "reference_stl": "ref1.stl"},
    {"task_id": "task2", "description": "desc2", "reference_stl": "ref2.stl"},
    {"task_id": "task3", "description": "desc3", "reference_stl": "ref3.stl"},
]
SAMPLE_MODELS_CONFIG = [
    {"name": "model_A", "provider": "provA"},
    {"name": "model_B", "provider": "provB"},
    {"name": "model_C", "provider": "provC"},
]

# Class to hold the static helper method
class TestRunEvaluationFiltering:
    @staticmethod
    def run_main_with_mocked_args(mock_args, mocks):
        """Helper function to set up mocks and run main(). Returns the mock logger."""
        (mock_parse_args, mock_get_config, mock_load_tasks,
         mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
         mock_generate_scad_for_task, mock_validate_openscad_config,
         mock_get_logger) = mocks

        # Make sure mock_args has a prompts attribute
        if not hasattr(mock_args, 'prompts'):
            mock_args.prompts = None
        
        mock_parse_args.return_value = mock_args

        # Create and set up the mock logger BEFORE config is loaded/main is run
        mock_logger = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger

        # Create the config data
        config_data = {
            # Provide a list of dictionaries for models
            'llm': {
                'models': [
                    {'name': 'model_A', 'provider': 'mock_provider', 'some_other_config': 'valA'},
                    {'name': 'model_B', 'provider': 'mock_provider', 'some_other_config': 'valB'},
                    {'name': 'model_C', 'provider': 'mock_provider', 'some_other_config': 'valC'}
                ]
            },
            'paths': {
                'openscad': '/mock/openscad'
            },
            'openscad': {
                'executable_path': '/mock/openscad',
                'minimum_version': '2021.01'
            },
            'evaluation': {
                'checks': ['render', 'watertight', 'manifold'],
                'num_replicates': 1
            },
            'timeouts': {'render': 60, 'generation': 120},
            'tasks': {
                'directory': 'mock_tasks',
                'schema_path': None
            },
            'prompts': {'default': 'Default prompt', 'concise': 'Concise prompt'}
        }
        
        # Get the mock Config object - now includes get_prompt method
        mock_config = create_mock_config(config_data)
        mock_get_config.return_value = mock_config

        mock_load_tasks.return_value = [
            {"task_id": "task1", "prompt": "prompt1", "reference_stl": "ref1.stl"},
            {"task_id": "task2", "prompt": "prompt2", "reference_stl": "ref2.stl"},
            {"task_id": "task3", "prompt": "prompt3", "reference_stl": "ref3.stl"}
        ]

        mock_validate_openscad_config.return_value = True
        mock_generate_scad_for_task.return_value = {
            "output_path": "mock_generated.scad",
            "output_scad_path": "mock_generated.scad",
            "success": True,
            "error": None,
            "generation_time": 5.0
        }
        mock_render_scad_file.return_value = {
            "scad_path": "mock_generated.scad",
            "stl_path": "mock_rendered.stl",
            "render_error": None,
            "render_time": 2.5,
            "status": "Success"
        }
        mock_perform_geometry_checks.return_value = {
            "checks": {
                "check_render_successful": True,
                "check_is_watertight": True,
                "check_is_single_component": True,
                "check_bounding_box_accurate": True,
                "check_volume_passed": True,
                "check_hausdorff_passed": True
            },
            "geometric_similarity_distance": 0.1,
            "icp_fitness_score": 0.99,
            "hausdorff_99p_distance": 0.2,
            "reference_volume_mm3": 1000.0,
            "generated_volume_mm3": 1005.0,
            "reference_bbox_mm": [10.0, 10.0, 10.0],
            "generated_bbox_aligned_mm": [10.1, 10.0, 9.9],
            "check_error": None
        }
        mock_assemble_final_results.return_value = {"final": "results"}

        try:
            main()
        except SystemExit as e:
            if e.code != 0:
                raise AssertionError(f"main() exited unexpectedly with code {e.code}") from e
        except Exception as e:
            raise AssertionError(f"main() raised an unexpected exception: {e}") from e
            
        return mock_logger

# **** Test functions moved to module level ****
def test_filtering_no_filters(caplog):
    """Test main() logic with no filters."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=None, models=None, 
                                   output_dir='results', run_id='testrun_nofilters', 
                                   log_level='INFO', log_file=None, prompts=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config, \
         patch('scripts.run_evaluation.os.path.exists') as mock_path_exists, \
         patch('scripts.run_evaluation.get_logger') as mock_get_logger:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config, mock_get_logger)

        mock_path_exists.return_value = True

        mock_logger = TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 9
        assert mock_render_scad_file.call_count == 9
        assert mock_perform_geometry_checks.call_count == 9
        assert mock_assemble_final_results.call_count == 1
        assert "Filtering tasks" not in caplog.text
        assert "Filtering models" not in caplog.text

        mock_logger.info.assert_called()

def test_filtering_with_task_filter(caplog):
    """Test main() logic with --tasks filter."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=["task2"], models=None,
                                   output_dir='results', run_id='testrun_taskfilter',
                                   log_level='INFO', log_file=None, prompts=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config, \
         patch('scripts.run_evaluation.os.path.exists') as mock_path_exists, \
         patch('scripts.run_evaluation.get_logger') as mock_get_logger:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config, mock_get_logger)
        
        mock_path_exists.return_value = True
        
        mock_logger = TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 3
        assert mock_render_scad_file.call_count == 3
        assert mock_perform_geometry_checks.call_count == 3
        calls = mock_generate_scad_for_task.call_args_list
        for call in calls:
            assert call.kwargs['task']['task_id'] == 'task2'

        # Use mock_logger assertions
        mock_logger.info.assert_any_call("Running specified tasks: task2")
        mock_logger.info.assert_any_call(f"Running all configured models: model_A, model_B, model_C")

def test_filtering_with_model_filter(caplog):
    """Test main() logic with --models filter."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=None, models=["model_B", "model_C"],
                                   output_dir='results', run_id='testrun_modelfilter',
                                   log_level='INFO', log_file=None, prompts=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config, \
         patch('scripts.run_evaluation.os.path.exists') as mock_path_exists, \
         patch('scripts.run_evaluation.get_logger') as mock_get_logger:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config, mock_get_logger)

        mock_path_exists.return_value = True

        mock_logger = TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 6
        assert mock_render_scad_file.call_count == 6
        assert mock_perform_geometry_checks.call_count == 6
        calls = mock_generate_scad_for_task.call_args_list
        models_called = {call.kwargs['model_config']['name'] for call in calls}
        assert models_called == {'model_B', 'model_C'}
        
        # Use mock_logger assertions - check if EITHER order was logged
        try:
            mock_logger.info.assert_any_call("Running specified models: model_B, model_C")
        except AssertionError:
            mock_logger.info.assert_any_call("Running specified models: model_C, model_B")
        mock_logger.info.assert_any_call("Running all found tasks.")

def test_filtering_with_both_filters(caplog):
    """Test main() logic with both --tasks and --models filters."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=["task1", "task3"], models=["model_A"],
                                   output_dir='results', run_id='testrun_bothfilters',
                                   log_level='INFO', log_file=None, prompts=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config, \
         patch('scripts.run_evaluation.os.path.exists') as mock_path_exists, \
         patch('scripts.run_evaluation.get_logger') as mock_get_logger:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config, mock_get_logger)
        
        mock_path_exists.return_value = True
        
        mock_logger = TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 2
        assert mock_render_scad_file.call_count == 2
        assert mock_perform_geometry_checks.call_count == 2
        calls = mock_generate_scad_for_task.call_args_list
        models_called = {call.kwargs['model_config']['name'] for call in calls}
        tasks_called = {call.kwargs['task']['task_id'] for call in calls}
        assert models_called == {'model_A'}
        assert tasks_called == {'task1', 'task3'}
        # Corrected log check (order might vary in set, check both)
        # assert ("Running specified tasks: task1, task3" in caplog.text or \ # Removed caplog check
        #         "Running specified tasks: task3, task1" in caplog.text) # Removed caplog check
        # assert "Running specified models: model_A" in caplog.text # Removed caplog check

        # Use mock_logger assertions - check if EITHER order was logged for tasks
        try:
            mock_logger.info.assert_any_call("Running specified tasks: task1, task3")
        except AssertionError:
            mock_logger.info.assert_any_call("Running specified tasks: task3, task1")
        mock_logger.info.assert_any_call("Running specified models: model_A")

# Keep TestRunEvaluationArgs and TestResultAssembly as unittest.TestCase for now if they don't use caplog
# Or convert them too if needed later.
class TestRunEvaluationArgs(unittest.TestCase):
    """Tests for the command-line argument parsing."""

    def test_parse_arguments_defaults(self):
        """Test parsing with default arguments."""
        # Simulate running with no arguments
        test_args = []
        # Use patch to simulate sys.argv
        with patch.object(sys, 'argv', ['scripts/run_evaluation.py'] + test_args):
            args = parse_arguments()

        self.assertEqual(args.config, 'config.yaml')
        self.assertIsNone(args.tasks) # Default is None
        self.assertIsNone(args.models) # Default is None
        self.assertEqual(args.output_dir, 'results')
        # Run ID is dynamic (timestamp), so we just check it exists
        self.assertIsNotNone(args.run_id)
        self.assertTrue(isinstance(args.run_id, str))
        self.assertEqual(args.log_level, 'INFO')
        self.assertIsNone(args.prompts)  # Default should be None

    def test_parse_arguments_specific_values(self):
        """Test parsing with specific arguments provided."""
        test_args = [
            '--config', 'my_config.yml',
            '--tasks', 'taskA', 'taskB',
            '--models', 'modelX', 'modelY',
            '--output-dir', 'my_outputs',
            '--run-id', 'test_run_123',
            '--log-level', 'DEBUG',
            '--prompts', 'default', 'concise'
        ]
        with patch.object(sys, 'argv', ['scripts/run_evaluation.py'] + test_args):
            args = parse_arguments()

        self.assertEqual(args.config, 'my_config.yml')
        self.assertEqual(args.tasks, ['taskA', 'taskB'])
        self.assertEqual(args.models, ['modelX', 'modelY'])
        self.assertEqual(args.output_dir, 'my_outputs')
        self.assertEqual(args.run_id, 'test_run_123')
        self.assertEqual(args.log_level, 'DEBUG')
        self.assertEqual(args.prompts, ['default', 'concise'])

    def test_parse_arguments_single_task_model(self):
        """Test parsing with single task and model."""
        test_args = [
            '--tasks', 'taskOnlyOne',
            '--models', 'modelJustOne'
        ]
        with patch.object(sys, 'argv', ['scripts/run_evaluation.py'] + test_args):
            args = parse_arguments()

        self.assertEqual(args.tasks, ['taskOnlyOne'])
        self.assertEqual(args.models, ['modelJustOne'])
        self.assertIsNone(args.prompts)  # Default should be None
        
    def test_parse_arguments_prompts(self):
        """Test parsing with the new --prompts argument."""
        # Test with multiple prompts
        test_args = [
            '--prompts', 'default', 'concise', 'detailed'
        ]
        with patch.object(sys, 'argv', ['scripts/run_evaluation.py'] + test_args):
            args = parse_arguments()
            
        self.assertEqual(args.prompts, ['default', 'concise', 'detailed'])
        
        # Test with a single prompt
        test_args = [
            '--prompts', 'concise'
        ]
        with patch.object(sys, 'argv', ['scripts/run_evaluation.py'] + test_args):
            args = parse_arguments()
            
        self.assertEqual(args.prompts, ['concise'])

# --- Test for assemble_final_results --- #
class TestResultAssembly(unittest.TestCase):
    """Tests focused on the assemble_final_results function."""

    def test_assemble_final_results_structure_and_content(self):
        """Verify the structure and content of the assembled result entry."""
        from scripts.run_evaluation import assemble_final_results, project_root # Import here

        # Create a mock logger
        mock_logger = MagicMock(spec=logging.Logger)

        # --- Input Data Mocks ---
        task_id = "t1"
        model_name = "m1"
        provider = "test_provider"
        scad_file_rel = "scad/t1_m1.scad"
        stl_file_rel = "stl/t1_m1.stl"
        scad_file_abs = os.path.join(project_root, scad_file_rel)
        stl_file_abs = os.path.join(project_root, stl_file_rel)
        summary_file_rel = "stl/t1_m1_summary.json"
        summary_file_abs = os.path.join(project_root, summary_file_rel)

        task_data = {"task_id": task_id, "reference_stl": "ref.stl", "description": "Test task"}
        scad_to_task_map_in = {scad_file_abs: {"task_data": task_data}} # Map SCAD path to task info

        gen_results_list_in = [
            {
                "task_id": task_id,
                "model": f"{provider}_{model_name}", # Example identifier
                "output_path": scad_file_abs,
                "error": None,
                "success": True,
                "generation_time": 10.5,
                "prompt_used": "Make a cube",
                "model_config_used": {"name": model_name, "provider": provider, "temperature": 0.5, "max_tokens": 100},
                "timestamp": "some_timestamp"
            }
        ]
        render_results_list_in = [
            {
                "scad_path": scad_file_abs,
                "stl_path": stl_file_abs,
                "summary_path": summary_file_abs,
                "status": "Success",
                "duration": 5.2,
                "error": None,
                "return_code": 0,
                "stdout": "",
                "stderr": ""
            }
        ]
        check_results = {
            "checks": {
                "check_render_successful": True,
                "check_is_watertight": True,
                "check_is_single_component": True,
                "check_bounding_box_accurate": False,
                "check_volume_passed": True,
                "check_hausdorff_passed": False
            },
            "geometric_similarity_distance": 0.8,
            "icp_fitness_score": 0.95,
            "hausdorff_99p_distance": 0.6,
            "reference_volume_mm3": 1000.0,
            "generated_volume_mm3": 1005.0,
            "reference_bbox_mm": [10.0, 10.0, 10.0],
            "generated_bbox_aligned_mm": [11.0, 10.0, 10.0],
            "error": None, # Orchestration error
            "check_errors": ["Bbox mismatch"] # Individual check errors
        }
        check_results_map_in = {scad_file_abs: check_results} # Map SCAD path to check results

        # --- Call the function with CORRECT ARGS ---
        final_results_list = assemble_final_results(
            gen_results_list_in, 
            render_results_list_in, 
            check_results_map_in, 
            scad_to_task_map_in, 
            mock_logger
        )

        # --- Assertions ---
        self.assertEqual(len(final_results_list), 1)
        final_entry = final_results_list[0] # Check the first (only) entry

        # Verify structure (spot check some keys)
        self.assertIn("task_id", final_entry)
        self.assertIn("model_name", final_entry)
        self.assertIn("llm_config", final_entry)
        self.assertIn("output_scad_path", final_entry)
        self.assertIn("output_stl_path", final_entry)
        self.assertIn("output_summary_json_path", final_entry)
        self.assertIn("render_status", final_entry)
        self.assertIn("checks", final_entry)
        self.assertIn("geometric_similarity_distance", final_entry)
        self.assertIn("hausdorff_99p_distance", final_entry)
        self.assertIn("reference_volume_mm3", final_entry)
        self.assertIn("generation_error", final_entry)
        self.assertIn("check_error", final_entry)

        # Verify content transfer
        self.assertEqual(final_entry["task_id"], task_id)
        self.assertEqual(final_entry["model_name"], model_name)
        self.assertEqual(final_entry["llm_config"]["provider"], provider)
        self.assertEqual(final_entry["output_scad_path"], scad_file_rel)
        self.assertEqual(final_entry["output_stl_path"], stl_file_rel)
        self.assertEqual(final_entry["output_summary_json_path"], summary_file_rel)
        self.assertEqual(final_entry["render_status"], "Success")
        self.assertEqual(final_entry["checks"]["check_is_watertight"], True)
        self.assertEqual(final_entry["checks"]["check_bounding_box_accurate"], False)
        self.assertEqual(final_entry["geometric_similarity_distance"], 0.8)
        self.assertEqual(final_entry["hausdorff_99p_distance"], 0.6)
        self.assertEqual(final_entry["reference_volume_mm3"], 1000.0)
        self.assertIsNone(final_entry["generation_error"])
        self.assertEqual(final_entry["check_error"], "Bbox mismatch") # Combined error
        self.assertEqual(final_entry["task_description"], "Test task")
        self.assertEqual(final_entry["reference_stl_path"], "ref.stl")
        mock_logger.info.assert_called() # Check logger was used

    def test_assemble_final_results_check_failure(self):
        """Test assembly when geometry check itself failed (orchestration error)."""
        from scripts.run_evaluation import assemble_final_results, project_root
        # Create a mock logger
        mock_logger = MagicMock(spec=logging.Logger)

        # --- Input Data Mocks ---
        task_id = "t2"
        model_name = "m2"
        provider = "test_provider2"
        scad_file_rel = "scad/t2_m2.scad"
        stl_file_rel = "stl/t2_m2.stl"
        scad_file_abs = os.path.join(project_root, scad_file_rel)
        stl_file_abs = os.path.join(project_root, stl_file_rel)

        task_data = {"task_id": task_id, "reference_stl": "ref2.stl"}
        scad_to_task_map_in = {scad_file_abs: {"task_data": task_data}}

        gen_results_list_in = [
            {
                "task_id": task_id,
                "model": f"{provider}_{model_name}",
                "output_path": scad_file_abs,
                "error": None,
                "success": True,
                 "model_config_used": {"name": model_name, "provider": provider}
            }
        ]
        render_results_list_in = [
            {
                "scad_path": scad_file_abs,
                "stl_path": stl_file_abs,
                "status": "Success",
                "error": None
            }
        ]
        # Mock geometry check failure (orchestration error)
        check_results = {"error": "Failed to load mesh", "checks": {"check_render_successful": None}} # No checks ran
        check_results_map_in = {scad_file_abs: check_results}

        # --- Call the function with CORRECT ARGS ---
        final_results_list = assemble_final_results(
             gen_results_list_in,
             render_results_list_in,
             check_results_map_in,
             scad_to_task_map_in,
             mock_logger
        )

        # --- Assertions ---
        self.assertEqual(len(final_results_list), 1)
        final_entry = final_results_list[0]

        self.assertEqual(final_entry["task_id"], task_id)
        self.assertEqual(final_entry["model_name"], model_name)
        self.assertEqual(final_entry["output_scad_path"], scad_file_rel)
        self.assertEqual(final_entry["output_stl_path"], stl_file_rel)
        self.assertEqual(final_entry["render_status"], "Success")
        self.assertIsNone(final_entry["generation_error"])
        self.assertEqual(final_entry["check_error"], "Failed to load mesh") # Check error propagated
        # Checks should be None as they didn't run
        self.assertIsNone(final_entry["checks"]["check_is_watertight"])
        self.assertIsNone(final_entry["geometric_similarity_distance"])
        self.assertIsNone(final_entry["hausdorff_99p_distance"])
        mock_logger.info.assert_called()

# Add more tests here for other parts of run_evaluation.py later
# (e.g., task/model filtering logic, results assembly)

if __name__ == '__main__':
    unittest.main() 