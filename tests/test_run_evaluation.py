#!/usr/bin/env python3
"""
Unit tests for the run_evaluation.py script.
"""

import unittest
import argparse
import os
import sys
from unittest.mock import patch

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

    def run_main_with_mocked_args(self, mock_args, mocks):
        """Helper function to set up mocks and run main()."""
        # Unpack mocks (order depends on decorator order)
        (mock_parse_args, mock_get_config, mock_load_tasks, mock_assemble_results,
         mock_perform_checks, mock_render_scad, mock_generate_scad,
         mock_validate_openscad, mock_get_logger, mock_setup_logger) = mocks

        # --- Setup Mocks --- #
        mock_parse_args.return_value = mock_args
        mock_config_obj = unittest.mock.Mock(spec=Config)
        mock_config_obj.get_required.side_effect = lambda key: {
            'llm.models': SAMPLE_MODELS_CONFIG
        }.get(key)
        mock_config_obj.get.side_effect = lambda key, default=None: {
             'directories.tasks': './tasks',
             'directories.generated_outputs': './generated_outputs'
        }.get(key, default)
        mock_get_config.return_value = mock_config_obj
        mock_load_tasks.return_value = SAMPLE_TASKS
        mock_logger_instance = unittest.mock.Mock()
        mock_get_logger.return_value = mock_logger_instance
        mock_assemble_results.return_value = []

        # --- Call main() --- #
        try:
            main()
        except SystemExit as e:
            self.fail(f"main() exited unexpectedly: {e}")
        except Exception as e:
            self.fail(f"main() raised an unexpected exception during filtering test: {e}")

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
        expected_log = f"Starting evaluation loop for {len(SAMPLE_TASKS)} tasks and {len(SAMPLE_MODELS_CONFIG)} models..."
        self.assert_log_message_contains(mock_logger, expected_log)

    def test_filtering_with_task_filter(self, *mocks):
        """Test main() logic with --tasks filter."""
        mock_args = argparse.Namespace(config='config.yaml', tasks=["task2"], models=None,
                                       output_dir='results', run_id='testrun_taskfilter',
                                       log_level='INFO', log_file=None)
        mock_logger = self.run_main_with_mocked_args(mock_args, mocks)
        expected_log = f"Starting evaluation loop for 1 tasks and {len(SAMPLE_MODELS_CONFIG)} models..."
        self.assert_log_message_contains(mock_logger, expected_log)
        self.assert_log_message_contains(mock_logger, "Running specified tasks: task2")

    def test_filtering_with_model_filter(self, *mocks):
        """Test main() logic with --models filter."""
        mock_args = argparse.Namespace(config='config.yaml', tasks=None, models=["model_B", "model_C"],
                                       output_dir='results', run_id='testrun_modelfilter',
                                       log_level='INFO', log_file=None)
        mock_logger = self.run_main_with_mocked_args(mock_args, mocks)
        expected_log = f"Starting evaluation loop for {len(SAMPLE_TASKS)} tasks and 2 models..."
        self.assert_log_message_contains(mock_logger, expected_log)
        self.assert_log_message_contains(mock_logger, "Running specified models: model_B, model_C")

    def test_filtering_with_both_filters(self, *mocks):
        """Test main() logic with both --tasks and --models filters."""
        mock_args = argparse.Namespace(config='config.yaml', tasks=["task1", "task3"], models=["model_A"],
                                       output_dir='results', run_id='testrun_bothfilters',
                                       log_level='INFO', log_file=None)
        mock_logger = self.run_main_with_mocked_args(mock_args, mocks)
        expected_log = f"Starting evaluation loop for 2 tasks and 1 models..."
        self.assert_log_message_contains(mock_logger, expected_log)
        self.assert_log_message_contains(mock_logger, "Running specified tasks: task1, task3")
        self.assert_log_message_contains(mock_logger, "Running specified models: model_A")

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
        self.assertIsNone(args.log_file) # Default is None

    def test_parse_arguments_specific_values(self):
        """Test parsing with specific arguments provided."""
        test_args = [
            '--config', 'my_config.yml',
            '--tasks', 'taskA', 'taskB',
            '--models', 'modelX', 'modelY',
            '--output-dir', 'my_outputs',
            '--run-id', 'test_run_123',
            '--log-level', 'DEBUG',
            '--log-file', '/tmp/test.log'
        ]
        with patch.object(sys, 'argv', ['scripts/run_evaluation.py'] + test_args):
            args = parse_arguments()

        self.assertEqual(args.config, 'my_config.yml')
        self.assertEqual(args.tasks, ['taskA', 'taskB'])
        self.assertEqual(args.models, ['modelX', 'modelY'])
        self.assertEqual(args.output_dir, 'my_outputs')
        self.assertEqual(args.run_id, 'test_run_123')
        self.assertEqual(args.log_level, 'DEBUG')
        self.assertEqual(args.log_file, '/tmp/test.log')

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

# Add more tests here for other parts of run_evaluation.py later
# (e.g., task/model filtering logic, results assembly)

if __name__ == '__main__':
    unittest.main() 