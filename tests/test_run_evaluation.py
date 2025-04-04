#!/usr/bin/env python3
"""
Unit tests for the run_evaluation.py script.
"""

import unittest
import argparse
import os
import sys
from unittest.mock import patch
import logging

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
from scripts.config_loader import Config

# --- Test Task/Model Filtering Logic ---
# We need to mock several functions called by main()

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

# Decorators to apply multiple patches
@patch('scripts.run_evaluation.setup_logger')
@patch('scripts.run_evaluation.get_logger')
@patch('scripts.run_evaluation.validate_openscad_config')
@patch('scripts.run_evaluation.generate_scad_for_task')
@patch('scripts.run_evaluation.render_scad_file')
@patch('scripts.run_evaluation.perform_geometry_checks')
@patch('scripts.run_evaluation.assemble_final_results')
@patch('scripts.run_evaluation.load_tasks')
@patch('scripts.run_evaluation.get_config')
@patch('scripts.run_evaluation.parse_arguments')
class TestRunEvaluationFiltering(unittest.TestCase):
    """Tests for the task and model filtering logic within main()."""

    # Add helper here for substring check
    def assert_log_message_contains_substring(self, mock_logger, expected_substring):
        """Helper to check if an expected substring exists in any logged info message."""
        found_log_call = False
        for call in mock_logger.info.call_args_list:
            if expected_substring in call.args[0]:
                found_log_call = True
                break
        self.assertTrue(found_log_call, f"Expected log substring not found: '{expected_substring}'")

    def run_main_with_mocked_args(self, mock_args, mocks):
        """Helper function to set up mocks and run main()."""
        # Unpack mocks (order depends on decorator order)
        (mock_parse_args, mock_get_config, mock_load_tasks, mock_assemble_results,
         mock_perform_checks, mock_render_scad, mock_generate_scad,
         mock_validate_openscad, mock_get_logger, mock_setup_logger) = mocks

        # --- Setup Mocks --- #
        print(f"\n--- [DEBUG TEST RUNNER] Running test with args: {mock_args} ---") # DEBUG PRINT
        mock_parse_args.return_value = mock_args
        mock_config_obj = unittest.mock.Mock(spec=Config)
        # Keep get_required for potential future use or other keys
        mock_config_obj.get_required.side_effect = lambda key: {
            # 'llm.models': SAMPLE_MODELS_CONFIG # get_required might not be used for this
        }.get(key, None) # Return None if key not found in get_required dict

        mock_config_obj.get.side_effect = lambda key, default=None: {
             'directories.tasks': './tasks',
             'directories.generated_outputs': './generated_outputs',
             'llm.models': SAMPLE_MODELS_CONFIG # <<< ADDED: Handle llm.models in get()
        }.get(key, default)
        print(f"--- [DEBUG TEST RUNNER] Mock config.get('llm.models') will return: {mock_config_obj.get('llm.models')} ---") # DEBUG PRINT
        mock_get_config.return_value = mock_config_obj
        mock_load_tasks.return_value = SAMPLE_TASKS
        mock_logger_instance = unittest.mock.Mock()
        mock_get_logger.return_value = mock_logger_instance
        mock_assemble_results.return_value = []

        # --- Setup mock return values for generate/render/check --- #
        # Make generate return a unique path each time to simulate different outputs
        mock_generate_scad.side_effect = lambda task, model_config, prompt, output_dir: {
            "success": True, "output_path": os.path.join(output_dir, f"{task['task_id']}_{model_config['name']}.scad"),
            "prompt_used": prompt, "model_config_used": model_config, "timestamp": "ts", "error": None
        }
        # Make render return success with a path
        mock_render_scad.side_effect = lambda scad_path, output_dir, config: {
            "scad_path": scad_path, "stl_path": scad_path.replace(".scad", ".stl"), "status": "Success", "error": None
        }
        # Make checks return a simple success dict
        mock_perform_checks.return_value = {"check_is_watertight": True, "check_errors": []}

        # --- Call main() --- #
        print("--- [DEBUG TEST RUNNER] Calling main()... ---") # DEBUG PRINT
        try:
            main()
            print("--- [DEBUG TEST RUNNER] main() completed without SystemExit. ---") # DEBUG PRINT
        except SystemExit as e:
            print(f"--- [DEBUG TEST RUNNER] main() exited with SystemExit({e}). ---") # DEBUG PRINT
            # Allow SystemExit(0) for normal exit, fail otherwise
            if e.code != 0:
                 self.fail(f"main() exited unexpectedly with code: {e.code}")
        except Exception as e:
            print(f"--- [DEBUG TEST RUNNER] main() raised exception: {type(e).__name__}: {e} ---") # DEBUG PRINT
            self.fail(f"main() raised an unexpected exception during filtering test: {e}")

        # --- Print captured logs --- #
        print("--- [DEBUG TEST RUNNER] Captured logger.info calls: ---")
        for i, call in enumerate(mock_logger_instance.info.call_args_list):
            print(f"    Log {i+1}: {call.args[0]}")
        print("---------------------------------------------------------")

        return mock_logger_instance # Return logger to check calls

    def assert_log_message_contains(self, mock_logger, expected_text):
        """Helper to check if an expected message was logged."""
        found_log_call = False
        for call in mock_logger.info.call_args_list:
            if expected_text in call.args[0]:
                found_log_call = True
                break
        self.assertTrue(found_log_call, f"Expected log message not found: '{expected_text}'")

    # Test cases call the helper with different mock_args
    def test_filtering_no_filters(self, *mocks):
        """Test main() logic when no --tasks or --models args are given."""
        mock_args = argparse.Namespace(config='config.yaml', tasks=None, models=None,
                                       output_dir='results', run_id='testrun_nofilter',
                                       log_level='INFO', log_file=None)
        mock_logger = self.run_main_with_mocked_args(mock_args, mocks)
        # Match the exact log format from run_evaluation.py log output
        total_combinations = len(SAMPLE_TASKS) * len(SAMPLE_MODELS_CONFIG)
        expected_log = f"Starting evaluation loop for {len(SAMPLE_TASKS)} tasks and {len(SAMPLE_MODELS_CONFIG)} models... Total: {total_combinations} combinations."
        self.assert_log_message_contains(mock_logger, expected_log)

        # --- Verify perform_geometry_checks call --- #
        # Access mocks by index based on decorator order
        mock_render_scad_func = mocks[5]
        mock_perform_checks_func = mocks[4]
        if mock_render_scad_func.call_count > 0:
            successful_render_call_args = None
            for call in mock_render_scad_func.call_args_list:
                successful_render_call_args = call.args
                break
            if successful_render_call_args:
                self.assertGreater(mock_perform_checks_func.call_count, 0, "perform_geometry_checks should be called if rendering succeeds")
                args, kwargs = mock_perform_checks_func.call_args
                self.assertIn('rendering_info', kwargs, "'rendering_info' missing from perform_geometry_checks call")
                self.assertIsInstance(kwargs['rendering_info'], dict, "'rendering_info' should be a dict")
                self.assertIn('status', kwargs['rendering_info'], "'status' key missing in rendering_info dict")
                self.assertEqual(kwargs['rendering_info']['status'], 'Success', "Incorrect status in rendering_info")

    def test_filtering_with_task_filter(self, *mocks):
        """Test main() logic with --tasks filter."""
        mock_args = argparse.Namespace(config='config.yaml', tasks=["task2"], models=None,
                                       output_dir='results', run_id='testrun_taskfilter',
                                       log_level='INFO', log_file=None)
        mock_logger = self.run_main_with_mocked_args(mock_args, mocks)
        # Match the exact log format
        tasks_run_count = 1 # Only task2
        models_run_count = len(SAMPLE_MODELS_CONFIG)
        total_combinations = tasks_run_count * models_run_count
        expected_log = f"Starting evaluation loop for {tasks_run_count} tasks and {models_run_count} models... Total: {total_combinations} combinations."
        self.assert_log_message_contains(mock_logger, expected_log)
        self.assert_log_message_contains(mock_logger, "Running specified tasks: task2")

        # --- Verify perform_geometry_checks call --- #
        mock_render_scad_func = mocks[5]
        mock_perform_checks_func = mocks[4]
        if mock_render_scad_func.call_count > 0:
            successful_render_call_args = None
            for call in mock_render_scad_func.call_args_list:
                successful_render_call_args = call.args
                break
            if successful_render_call_args:
                self.assertGreater(mock_perform_checks_func.call_count, 0, "perform_geometry_checks should be called if rendering succeeds")
                args, kwargs = mock_perform_checks_func.call_args
                self.assertIn('rendering_info', kwargs, "'rendering_info' missing from perform_geometry_checks call")
                self.assertIsInstance(kwargs['rendering_info'], dict, "'rendering_info' should be a dict")
                self.assertIn('status', kwargs['rendering_info'], "'status' key missing in rendering_info dict")
                self.assertEqual(kwargs['rendering_info']['status'], 'Success', "Incorrect status in rendering_info")

    def test_filtering_with_model_filter(self, *mocks):
        """Test main() logic with --models filter."""
        mock_args = argparse.Namespace(config='config.yaml', tasks=None, models=["model_B", "model_C"],
                                       output_dir='results', run_id='testrun_modelfilter',
                                       log_level='INFO', log_file=None)
        mock_logger = self.run_main_with_mocked_args(mock_args, mocks)
        # Match the exact log format
        tasks_run_count = len(SAMPLE_TASKS)
        models_run_count = 2 # model_B, model_C
        total_combinations = tasks_run_count * models_run_count
        expected_log = f"Starting evaluation loop for {tasks_run_count} tasks and {models_run_count} models... Total: {total_combinations} combinations."
        self.assert_log_message_contains(mock_logger, expected_log)
        # Check for presence of model names, ignoring order
        self.assert_log_message_contains_substring(mock_logger, "Running specified models")
        self.assert_log_message_contains_substring(mock_logger, "model_B")
        self.assert_log_message_contains_substring(mock_logger, "model_C")

        # --- Verify perform_geometry_checks call --- #
        mock_render_scad_func = mocks[5]
        mock_perform_checks_func = mocks[4]
        if mock_render_scad_func.call_count > 0:
            successful_render_call_args = None
            for call in mock_render_scad_func.call_args_list:
                successful_render_call_args = call.args
                break
            if successful_render_call_args:
                self.assertGreater(mock_perform_checks_func.call_count, 0, "perform_geometry_checks should be called if rendering succeeds")
                args, kwargs = mock_perform_checks_func.call_args
                self.assertIn('rendering_info', kwargs, "'rendering_info' missing from perform_geometry_checks call")
                self.assertIsInstance(kwargs['rendering_info'], dict, "'rendering_info' should be a dict")
                self.assertIn('status', kwargs['rendering_info'], "'status' key missing in rendering_info dict")
                self.assertEqual(kwargs['rendering_info']['status'], 'Success', "Incorrect status in rendering_info")

    def test_filtering_with_both_filters(self, *mocks):
        """Test main() logic with both --tasks and --models filters."""
        mock_args = argparse.Namespace(config='config.yaml', tasks=["task1", "task3"], models=["model_A"],
                                       output_dir='results', run_id='testrun_bothfilters',
                                       log_level='INFO', log_file=None)
        mock_logger = self.run_main_with_mocked_args(mock_args, mocks)
        # Match the exact log format
        tasks_run_count = 2 # task1, task3
        models_run_count = 1 # model_A
        total_combinations = tasks_run_count * models_run_count
        expected_log = f"Starting evaluation loop for {tasks_run_count} tasks and {models_run_count} models... Total: {total_combinations} combinations."
        self.assert_log_message_contains(mock_logger, expected_log)
        # Check for presence of task IDs, ignoring order
        self.assert_log_message_contains_substring(mock_logger, "Running specified tasks")
        self.assert_log_message_contains_substring(mock_logger, "task1")
        self.assert_log_message_contains_substring(mock_logger, "task3")
        # Check model log (order should be deterministic here)
        self.assert_log_message_contains(mock_logger, "Running specified models: model_A")

        # --- Verify perform_geometry_checks call --- #
        # Access mocks by index based on decorator order
        mock_render_scad_func = mocks[5]
        mock_perform_checks_func = mocks[4]
        if mock_render_scad_func.call_count > 0: # Check only if render was attempted
            successful_render_call_args = None
            for call in mock_render_scad_func.call_args_list:
                # This mock returns status="Success", so any call implies success
                successful_render_call_args = call.args
                break
            
            if successful_render_call_args:
                self.assertGreater(mock_perform_checks_func.call_count, 0, "perform_geometry_checks should be called if rendering succeeds")
                # Get the arguments from the first call to perform_geometry_checks
                args, kwargs = mock_perform_checks_func.call_args
                # Check if 'rendering_info' is in kwargs and is a dictionary
                self.assertIn('rendering_info', kwargs, "'rendering_info' missing from perform_geometry_checks call")
                self.assertIsInstance(kwargs['rendering_info'], dict, "'rendering_info' should be a dict")
                # Check if the passed dict has the expected 'status' key from the mock render result
                self.assertIn('status', kwargs['rendering_info'], "'status' key missing in rendering_info dict")
                self.assertEqual(kwargs['rendering_info']['status'], 'Success', "Incorrect status in rendering_info")

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

    def test_parse_arguments_specific_values(self):
        """Test parsing with specific arguments provided."""
        test_args = [
            '--config', 'my_config.yml',
            '--tasks', 'taskA', 'taskB',
            '--models', 'modelX', 'modelY',
            '--output-dir', 'my_outputs',
            '--run-id', 'test_run_123',
            '--log-level', 'DEBUG',
        ]
        with patch.object(sys, 'argv', ['scripts/run_evaluation.py'] + test_args):
            args = parse_arguments()

        self.assertEqual(args.config, 'my_config.yml')
        self.assertEqual(args.tasks, ['taskA', 'taskB'])
        self.assertEqual(args.models, ['modelX', 'modelY'])
        self.assertEqual(args.output_dir, 'my_outputs')
        self.assertEqual(args.run_id, 'test_run_123')
        self.assertEqual(args.log_level, 'DEBUG')

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

# --- Test for assemble_final_results --- #
class TestResultAssembly(unittest.TestCase):
    """Tests for the assemble_final_results function."""

    def test_assemble_final_results_runs(self):
        """Test that assemble_final_results can run without basic errors."""
        mock_logger = unittest.mock.Mock(spec=logging.Logger)
        # Provide minimal mock data structure expected by the function
        generation_results = [{
            "task_id": "t1", "model": "m1", "output_path": "/path/to/t1_m1.scad",
            "model_config_used": {"name": "m1", "provider": "p1"}, "success": True, "prompt_used": "p"
        }]
        render_results = [{"scad_path": "/path/to/t1_m1.scad", "stl_path": "/path/to/t1_m1.stl", "status": "Success"}]
        check_results_map = {"/path/to/t1_m1.scad": {"check_is_watertight": True}}
        scad_to_task_map = {"/path/to/t1_m1.scad": {"task_data": {"task_id": "t1", "reference_stl": "ref.stl"}, "model_config": {"name": "m1"}}}

        try:
            # Import here to avoid issues if the module has top-level problems
            from scripts.run_evaluation import assemble_final_results
            final_list = assemble_final_results(
                generation_results,
                render_results,
                check_results_map,
                scad_to_task_map,
                mock_logger # Pass the mock logger
            )
            # Basic check that it produced a list (content correctness not focus here)
            self.assertIsInstance(final_list, list)
            # Check that the logger was called (e.g., for the starting message)
            mock_logger.info.assert_called()
        except NameError as e:
            self.fail(f"assemble_final_results raised NameError (likely logger issue): {e}")
        except Exception as e:
            self.fail(f"assemble_final_results raised unexpected exception: {e}")

# Add more tests here for other parts of run_evaluation.py later
# (e.g., task/model filtering logic, results assembly)

if __name__ == '__main__':
    unittest.main() 