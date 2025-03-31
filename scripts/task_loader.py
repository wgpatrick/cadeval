#!/usr/bin/env python3
"""
Task Loader for CadEval

This module provides utilities for loading task definition files (YAML)
from a specified directory.
"""

import os
import sys
import yaml
import glob
import json
import jsonschema
from typing import List, Dict, Any, Optional
import argparse

# Add parent directory to path to allow importing logger_setup
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from scripts.logger_setup import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback basic logger if logger_setup is not available initially
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Could not import logger_setup. Using basic logging.")


class TaskLoadError(Exception):
    """Exception raised for errors during task loading."""
    pass


def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load a JSON schema from a file.

    Args:
        schema_path: Path to the JSON schema file.

    Returns:
        The loaded schema as a dictionary.

    Raises:
        TaskLoadError: If the schema file doesn't exist or has invalid JSON.
    """
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        return schema
    except FileNotFoundError:
        error_msg = f"Schema file not found: {os.path.abspath(schema_path)}"
        logger.error(error_msg)
        raise TaskLoadError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in schema file {schema_path}: {e}"
        logger.error(error_msg)
        raise TaskLoadError(error_msg)


def validate_task(task: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate a task against a JSON schema.

    Args:
        task: The task data to validate.
        schema: The JSON schema to validate against.

    Returns:
        True if the task is valid, False otherwise.
    """
    try:
        jsonschema.validate(instance=task, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logger.warning(f"Task validation failed: {e}")
        return False


def load_tasks(tasks_dir: str = "tasks", validate: bool = True, schema_path: str = None) -> List[Dict[str, Any]]:
    """
    Scans a directory for YAML task files, loads them, and returns their content.

    Args:
        tasks_dir: The path to the directory containing task YAML files.
                   Defaults to "tasks".
        validate: Whether to validate tasks against the schema. Defaults to True.
        schema_path: Path to the JSON schema file. If None, will use the default path.

    Returns:
        A list of dictionaries, where each dictionary represents a loaded task.

    Raises:
        TaskLoadError: If the tasks directory doesn't exist or other critical errors occur.
    """
    if not os.path.isdir(tasks_dir):
        error_msg = f"Tasks directory not found: {os.path.abspath(tasks_dir)}"
        logger.error(error_msg)
        raise TaskLoadError(error_msg)

    logger.info(f"Scanning for task YAML files in: {os.path.abspath(tasks_dir)}")

    # Load schema if validation is enabled
    schema = None
    if validate:
        if schema_path is None:
            # Determine schema path relative to the project root
            script_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            schema_path = os.path.join(project_root, 'schemas', 'task_schema.json')
        
        try:
            schema = load_schema(schema_path)
            logger.info(f"Loaded JSON schema from {schema_path}")
        except TaskLoadError as e:
            logger.warning(f"Could not load schema, validation will be skipped: {e}")
            validate = False

    # Find all .yaml files in the directory
    # Using recursive=False to avoid potential issues if subdirs are added later
    yaml_pattern = os.path.join(tasks_dir, "*.yaml")
    task_files = glob.glob(yaml_pattern)

    if not task_files:
        logger.warning(f"No task YAML files found in {tasks_dir}")
        return []

    loaded_tasks: List[Dict[str, Any]] = []
    for task_file_path in task_files:
        logger.debug(f"Attempting to load task from: {task_file_path}")
        try:
            with open(task_file_path, 'r') as f:
                task_data = yaml.safe_load(f)
                if task_data:
                    # Basic validation: check for task_id presence
                    if 'task_id' not in task_data:
                         logger.warning(f"Task file '{task_file_path}' is missing 'task_id'. Skipping.")
                         continue
                    
                    # JSON schema validation if enabled
                    if validate and schema:
                        if not validate_task(task_data, schema):
                            logger.warning(f"Task file '{task_file_path}' failed schema validation. Skipping.")
                            continue
                    
                    loaded_tasks.append(task_data)
                    logger.info(f"Successfully loaded task '{task_data.get('task_id', 'UNKNOWN')}' from {task_file_path}")
                else:
                    logger.warning(f"Task file is empty or invalid: {task_file_path}")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {task_file_path}: {e}")
            # Decide whether to skip or raise an error - skipping for now
            continue
        except FileNotFoundError:
            # Should not happen with glob, but good practice
            logger.error(f"Task file disappeared unexpectedly: {task_file_path}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error loading task file {task_file_path}: {e}")
            continue # Skip this file

    logger.info(f"Finished loading tasks. Total loaded: {len(loaded_tasks)}")
    return loaded_tasks


if __name__ == "__main__":
    # Example usage when run directly
    print("Running Task Loader directly...")
    import logging # <<< Import logging at the start of the __main__ block

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load and validate task definition files")
    parser.add_argument("--tasks-dir", default=None, help="Directory containing task YAML files")
    parser.add_argument("--schema", default=None, help="Path to JSON schema file")
    parser.add_argument("--no-validate", action="store_true", help="Disable schema validation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    try:
        # Ensure logger is configured if run standalone
        # You might want to adjust log levels here for testing
        from scripts.logger_setup import setup_logger
        # Now 'logging.DEBUG' will be defined when this is called
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logger(__name__, level=log_level, log_file="logs/task_loader_run.log")

        # Define the directory relative to the script's location for robustness
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        
        # Use command line argument for tasks directory if provided
        tasks_directory = args.tasks_dir
        if tasks_directory is None:
            tasks_directory = os.path.join(project_root, 'tasks')  # Assumes tasks/ is in project root

        # Load tasks with validation according to command line arguments
        tasks = load_tasks(
            tasks_dir=tasks_directory, 
            validate=not args.no_validate,
            schema_path=args.schema
        )

        if tasks:
            print(f"\nSuccessfully loaded {len(tasks)} task(s):")
            for i, task in enumerate(tasks):
                print(f"\n--- Task {i+1} (ID: {task.get('task_id', 'N/A')}) ---")
                print(yaml.dump(task, default_flow_style=False)) # Pretty print
        else:
            print("\nNo tasks were loaded.")

    except TaskLoadError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError:
        # Fallback if logger_setup isn't runnable directly easily
        # 'logging' is already imported from above
        logger = logging.getLogger(__name__) # Re-initialize basic logger for this scope
        logger.warning("Could not import logger_setup for standalone run. Using basic logging.")
        # Adjust path guessing if needed, this assumes script is run from project root
        tasks_directory = args.tasks_dir or 'tasks'
        tasks = load_tasks(
            tasks_dir=tasks_directory,
            validate=not args.no_validate,
            schema_path=args.schema
        )
        if tasks:
             print(f"\nSuccessfully loaded {len(tasks)} task(s) (basic logging):")
             # Basic print if YAML dump fails without imports
             for task in tasks: print(task)
        else:
             print("\nNo tasks loaded (basic logging).")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
