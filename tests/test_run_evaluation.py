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
    mock_config = MagicMock(spec=Config)

    def mock_get(key, default=None):
        # print(f"Mock Get: key={key}, default={default}, returning={config_values.get(key, default)}") # DEBUG
        return config_values.get(key, default)

    def mock_get_required(key):
        val = config_values.get(key)
        # print(f"Mock Get Required: key={key}, returning={val}") # DEBUG
        if val is None:
            # print(f"Mock Get Required: Raising ConfigError for key={key}") # DEBUG
            raise ConfigError(f"Mock missing required config key: {key}")
        return val

    # Explicitly assign the methods to the mock attributes
    mock_config.get = mock_get
    mock_config.get_required = mock_get_required

    # mock_config.get.side_effect = mock_get # Keep side_effect commented out for now
    # mock_config.get_required.side_effect = mock_get_required

    mock_config.config_path = '/fake/path/config.yaml'
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
        """Helper function to set up mocks and run main()."""
        (mock_parse_args, mock_get_config, mock_load_tasks,
         mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
         mock_generate_scad_for_task, mock_validate_openscad_config) = mocks

        mock_parse_args.return_value = mock_args
        mock_config = create_mock_config({
            # Provide a list of dictionaries for models
            'llm.models': [
                {'name': 'model_A', 'provider': 'mock_provider', 'some_other_config': 'valA'},
                {'name': 'model_B', 'provider': 'mock_provider', 'some_other_config': 'valB'},
                {'name': 'model_C', 'provider': 'mock_provider', 'some_other_config': 'valC'}
            ],
            'paths': {'openscad': '/mock/openscad'},
            'evaluation': {'checks': ['render', 'watertight', 'manifold']},
            'timeouts': {'render': 60, 'generation': 120},
            'tasks.directory': 'mock_tasks',
            'tasks.schema_path': None
        })
        mock_get_config.return_value = mock_config

        mock_load_tasks.return_value = [
            {"task_id": "task1", "prompt": "prompt1", "reference_stl": "ref1.stl"},
            {"task_id": "task2", "prompt": "prompt2", "reference_stl": "ref2.stl"},
            {"task_id": "task3", "prompt": "prompt3", "reference_stl": "ref3.stl"}
        ]

        mock_validate_openscad_config.return_value = True
        mock_generate_scad_for_task.return_value = "mock_generated.scad"
        mock_render_scad_file.return_value = ("mock_rendered.stl", "", 0)
        mock_perform_geometry_checks.return_value = {"check_passed": True}
        mock_assemble_final_results.return_value = {"final": "results"}

        try:
            main()
        except SystemExit as e:
            if e.code != 0:
                raise AssertionError(f"main() exited unexpectedly with code {e.code}") from e
        except Exception as e:
            raise AssertionError(f"main() raised an unexpected exception: {e}") from e

# **** Test functions moved to module level ****
def test_filtering_no_filters(caplog):
    """Test main() logic when no --tasks or --models args are given."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=None, models=None,
                                   output_dir='results', run_id='testrun_nofilter',
                                   log_level='INFO', log_file=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config)

        TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 9
        assert mock_render_scad_file.call_count == 9
        assert mock_perform_geometry_checks.call_count == 9
        assert mock_assemble_final_results.call_count == 1
        assert "Filtering tasks" not in caplog.text
        assert "Filtering models" not in caplog.text

def test_filtering_with_task_filter(caplog):
    """Test main() logic with --tasks filter."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=["task2"], models=None,
                                   output_dir='results', run_id='testrun_taskfilter',
                                   log_level='INFO', log_file=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config)
        TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 3
        assert mock_render_scad_file.call_count == 3
        assert mock_perform_geometry_checks.call_count == 3
        calls = mock_generate_scad_for_task.call_args_list
        for call in calls:
            assert call.args[1]['id'] == 'task2'
        assert "Filtering tasks to: ['task2']" in caplog.text
        assert "Filtering models" not in caplog.text

def test_filtering_with_model_filter(caplog):
    """Test main() logic with --models filter."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=None, models=["model_B", "model_C"],
                                   output_dir='results', run_id='testrun_modelfilter',
                                   log_level='INFO', log_file=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config)
        TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 6
        assert mock_render_scad_file.call_count == 6
        assert mock_perform_geometry_checks.call_count == 6
        calls = mock_generate_scad_for_task.call_args_list
        models_called = {call.args[0] for call in calls}
        assert models_called == {'model_B', 'model_C'}
        assert "Filtering tasks" not in caplog.text
        assert "Filtering models to: ['model_B', 'model_C']" in caplog.text

def test_filtering_with_both_filters(caplog):
    """Test main() logic with both --tasks and --models filters."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(config='config.yaml', tasks=["task1", "task3"], models=["model_A"],
                                   output_dir='results', run_id='testrun_bothfilters',
                                   log_level='INFO', log_file=None)

    with patch('scripts.run_evaluation.parse_arguments') as mock_parse_args, \
         patch('scripts.run_evaluation.get_config') as mock_get_config, \
         patch('scripts.run_evaluation.load_tasks') as mock_load_tasks, \
         patch('scripts.run_evaluation.assemble_final_results') as mock_assemble_final_results, \
         patch('scripts.run_evaluation.perform_geometry_checks') as mock_perform_geometry_checks, \
         patch('scripts.run_evaluation.render_scad_file') as mock_render_scad_file, \
         patch('scripts.run_evaluation.generate_scad_for_task') as mock_generate_scad_for_task, \
         patch('scripts.run_evaluation.validate_openscad_config') as mock_validate_openscad_config:

        all_mocks = (mock_parse_args, mock_get_config, mock_load_tasks,
                     mock_assemble_final_results, mock_perform_geometry_checks, mock_render_scad_file,
                     mock_generate_scad_for_task, mock_validate_openscad_config)
        TestRunEvaluationFiltering.run_main_with_mocked_args(mock_args, all_mocks)

        assert mock_generate_scad_for_task.call_count == 2
        assert mock_render_scad_file.call_count == 2
        assert mock_perform_geometry_checks.call_count == 2
        calls = mock_generate_scad_for_task.call_args_list
        models_called = {call.args[0] for call in calls}
        tasks_called = {call.args[1]['id'] for call in calls}
        assert models_called == {'model_A'}
        assert tasks_called == {'task1', 'task3'}
        assert "Filtering tasks to: ['task1', 'task3']" in caplog.text
        assert "Filtering models to: ['model_A']" in caplog.text

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