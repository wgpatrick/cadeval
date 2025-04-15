#!/usr/bin/env python3
"""
Run Geometry Checks Only Script for CadEval

This script runs only the geometry check phase on an existing evaluation run directory.
It assumes STL files have already been generated (and optionally rendered).
"""

import os
import sys
import argparse
import json
import glob
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.logger_setup import setup_logger, get_logger
from scripts.config_loader import get_config, Config, ConfigError
from scripts.task_loader import load_tasks
from scripts.geometry_check import perform_geometry_checks, GeometryCheckError

# Initialize logger (will be configured by setup_logger later)
logger = get_logger(__name__)

# --- Constants ---
STL_DIR_NAME = "stl"
SCAD_DIR_NAME = "scad" # Needed for assembling results later potentially
RESULTS_FILENAME_SUFFIX = "_geometry_check_run.json"
LOG_FILENAME_SUFFIX = "_geometry_check_run.log"
# Regex to parse filenames like: task1_openai_gpt-4.1-2025-04-14_default_rep1.stl
# Needs to be robust to different provider/model name structures
FILENAME_PATTERN = re.compile(
    r"^(?P<task_id>task\d+)_"
    r"(?P<provider>[^_]+)_"
    r"(?P<model_name>.+)_"
    r"(?P<prompt_id>[^_]+)_"
    r"rep(?P<replicate_id>\d+)"
    r"\.(stl|scad)$", # Match both for potential future use
    re.IGNORECASE
)


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parses the standard CadEval filename format."""
    match = FILENAME_PATTERN.match(os.path.basename(filename))
    if match:
        return match.groupdict()
    else:
        logger.warning(f"Could not parse filename: {filename}")
        return None

def assemble_results_from_checks(
    run_dir: str,
    all_check_results: List[Dict[str, Any]],
    config: Config,
    tasks: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assembles the final results structure based *only* on the geometry check outputs.
    This version aims to match the structure of run_evaluation.py's output.
    """
    logger.info("Assembling final results from geometry checks...")
    final_results = []

    # Create a map for quick lookup of check results by a unique key
    check_results_map = {}
    for check_res in all_check_results:
        gen_path = check_res.get('generated_stl_path')
        if not gen_path:
            logger.warning("Skipping result assembly for entry missing 'generated_stl_path': {check_res}")
            continue
        parsed_info = parse_filename(gen_path)
        if parsed_info:
            key = (
                parsed_info['task_id'],
                parsed_info['provider'],
                parsed_info['model_name'],
                parsed_info['prompt_id'],
                int(parsed_info['replicate_id'])
            )
            check_results_map[key] = check_res

    # Iterate through the check results map
    for key, check_res in check_results_map.items():
        task_id, provider, model_name, prompt_id, replicate_id = key
        base_filename_no_ext = f"{task_id}_{provider}_{model_name}_{prompt_id}_rep{replicate_id}"

        task_info = tasks.get(task_id)
        if not task_info:
            logger.warning(f"Task info for {task_id} not found during results assembly.")
            continue

        # --- Start building the entry, matching run_evaluation.py structure ---
        result_entry = {
            "task_id": task_id,
            "replicate_id": replicate_id, # Ensure this is present
            "model_name": model_name,
            "provider": provider,
            "prompt_key_used": prompt_id, # Match key name
            "task_description": task_info.get("description"),
            "reference_stl_path": task_info.get("reference_stl"), # Match key name
            # Add placeholders for fields not available in this script
            "prompt_used": None, # Not available
            "llm_config": None, # Not available
            "timestamp_utc": None, # Not available
            "output_scad_path": os.path.join(run_dir, SCAD_DIR_NAME, f"{base_filename_no_ext}.scad"), # Construct path
            "output_stl_path": os.path.join(run_dir, STL_DIR_NAME, f"{base_filename_no_ext}.stl"), # Construct path
            "output_summary_json_path": None, # Not generated here
            "render_status": "Success", # Assume success if STL exists
            "render_duration_seconds": None, # Not available
            "render_error_message": None, # Not available

            # --- Nested "checks" dictionary --- START ---
            "checks": check_res.get("checks", {}), # Get the nested checks dict directly
            # --- Nested "checks" dictionary --- END ---

            # --- Promote metric values to top level --- START ---
            "geometric_similarity_distance": check_res.get("geometric_similarity_distance"),
            "icp_fitness_score": check_res.get("icp_fitness_score"),
            "hausdorff_95p_distance": check_res.get("hausdorff_95p_distance"),
            "hausdorff_99p_distance": check_res.get("hausdorff_99p_distance"),
            "reference_volume_mm3": check_res.get("reference_volume_mm3"),
            "generated_volume_mm3": check_res.get("generated_volume_mm3"),
            "reference_bbox_mm": check_res.get("reference_bbox_mm"),
            "generated_bbox_aligned_mm": check_res.get("generated_bbox_aligned_mm"),
            # --- Promote metric values to top level --- END ---

            # --- Error fields --- START ---
            "generation_error": None, # Not available
            "check_error": None, # Initialize
            # --- Error fields --- END ---

            # --- Other fields (placeholders) --- START ---
            "llm_duration_seconds": None,
            "generation_duration_seconds": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "estimated_cost": None
            # --- Other fields (placeholders) --- END ---
        }

        # Populate top-level "check_error" based on check_res["error"] or check_res["check_errors"]
        top_level_error = check_res.get("error")
        check_errors_list = check_res.get("check_errors", [])
        if top_level_error:
            result_entry["check_error"] = top_level_error
        elif check_errors_list:
            # Combine specific check errors into a single message if no top-level error exists
            result_entry["check_error"] = f"One or more checks failed: {'; '.join(check_errors_list)}"
        # Otherwise, check_error remains None

        final_results.append(result_entry)

    logger.info(f"Assembled {len(final_results)} final result entries from check data.")
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Run CadEval Geometry Checks on an existing run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to the existing run directory (e.g., results/YYYYMMDD_HHMMSS).")
    parser.add_argument("--log-level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Logging level.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Setup logging to console and a file within the run_dir
    log_file_path = os.path.join(run_dir, f"run{LOG_FILENAME_SUFFIX}")
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logger(name="", level=log_level_int, log_file=log_file_path)

    logger.info(f"--- Starting CadEval Geometry Check Run --- Target Dir: {run_dir}")
    logger.info(f"Command Line Arguments: {vars(args)}")

    # --- Load Config ---
    try:
        config_path = os.path.join(parent_dir, 'config.yaml') # Assume config is in project root
        if not os.path.exists(config_path):
             logger.warning(f"config.yaml not found at {config_path}, using defaults where possible.")
             # Handle default loading or error if config is essential
             config = Config({}, {}) # Empty config
        else:
             logger.info(f"Loading configuration from: {config_path}")
             config = get_config(config_path)
             logger.info("Successfully loaded configuration.")
    except ConfigError as e:
        logger.critical(f"Failed to load configuration: {e}")
        sys.exit(1)
    except Exception as e:
         logger.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
         sys.exit(1)

    # --- Load Tasks ---
    try:
        tasks_dir = config.get("tasks.directory", os.path.join(parent_dir, 'tasks'))
        tasks_schema_path = config.get("tasks.schema_path") # Optional schema
        if tasks_schema_path:
            tasks_schema_path = os.path.join(parent_dir, tasks_schema_path) # Make absolute if relative

        logger.info(f"Loading tasks from directory: {tasks_dir}")
        loaded_tasks_list = load_tasks(tasks_dir, tasks_schema_path)
        logger.info(f"Loaded {len(loaded_tasks_list)} task definitions initially.")
        if not loaded_tasks_list:
            logger.critical("No tasks loaded. Cannot proceed.")
            sys.exit(1)

        # Convert list of tasks to a dictionary keyed by task_id
        tasks_dict = {}
        tasks_without_id = 0
        for task in loaded_tasks_list:
            if isinstance(task, dict) and 'task_id' in task:
                tasks_dict[task['task_id']] = task
            else:
                logger.warning(f"Loaded task definition is not a dict or missing 'task_id': {task}")
                tasks_without_id += 1

        logger.info(f"Created tasks dictionary with {len(tasks_dict)} entries.")
        if tasks_without_id > 0:
            logger.warning(f"Skipped {tasks_without_id} loaded task definitions missing 'task_id'.")

    except Exception as e:
        logger.critical(f"Failed to load tasks: {e}", exc_info=True)
        sys.exit(1)

    # --- Find STL files ---
    stl_dir = os.path.join(run_dir, STL_DIR_NAME)
    if not os.path.isdir(stl_dir):
        logger.critical(f"STL directory not found in run directory: {stl_dir}")
        sys.exit(1)

    stl_files = glob.glob(os.path.join(stl_dir, "*.stl"))
    if not stl_files:
        logger.warning(f"No STL files found in {stl_dir}. Nothing to check.")
        # Decide whether to exit or just produce empty results
        # For now, let's proceed and assembly will handle empty checks list
        # sys.exit(0) # Or exit here

    logger.info(f"Found {len(stl_files)} STL files to check in {stl_dir}.")
    # Sort files for deterministic order (optional, but good practice)
    stl_files.sort()

    # --- Run Geometry Checks ---
    all_check_results = []
    logger.info(f"--- Starting Batch Geometry Checks for {len(stl_files)} STL files ---")
    check_start_time = datetime.now()

    for i, gen_stl_path in enumerate(stl_files):
        logger.info(f"  [{i+1}/{len(stl_files)}] Checking STL: {os.path.basename(gen_stl_path)}...")

        # Parse filename to get task ID etc.
        parsed_info = parse_filename(gen_stl_path)
        if not parsed_info:
            logger.error(f"    Skipping check for {os.path.basename(gen_stl_path)} due to filename parsing error.")
            # Optionally create a basic error result entry
            error_result = {"generated_stl_path": gen_stl_path, "error": "Filename parse error", "checks": {}}
            all_check_results.append(error_result)
            continue

        task_id = parsed_info['task_id']
        task_info = tasks_dict.get(task_id)

        if not task_info:
            logger.error(f"    Skipping check for {os.path.basename(gen_stl_path)} - Task '{task_id}' not found in loaded tasks.")
            error_result = {"generated_stl_path": gen_stl_path, "error": f"Task {task_id} definition not found", "checks": {}}
            all_check_results.append(error_result)
            continue

        ref_stl_rel_path = task_info.get("reference_stl")
        if not ref_stl_rel_path:
            logger.warning(f"    Reference STL path missing for task '{task_id}'. Some checks will be skipped for {os.path.basename(gen_stl_path)}.")
            ref_stl_abs_path = None
        else:
            # Assume reference_stl path in task yaml is relative to project root
            ref_stl_abs_path = os.path.normpath(os.path.join(parent_dir, ref_stl_rel_path))
            if not os.path.exists(ref_stl_abs_path):
                logger.error(f"    Reference STL file not found at resolved path: {ref_stl_abs_path} (from task '{task_id}')")
                # Decide whether to skip or continue with limited checks
                ref_stl_abs_path = None # Treat as missing

        task_requirements = task_info.get("requirements", {})

        # Simulate rendering_info - assume success if STL exists
        rendering_info = {"status": "Success"} # Simple assumption

        try:
            # Perform the checks using the imported function
            check_results = perform_geometry_checks(
                generated_stl_path=gen_stl_path,
                reference_stl_path=ref_stl_abs_path,
                task_requirements=task_requirements,
                rendering_info=rendering_info,
                config=config
            )
            all_check_results.append(check_results)
            if check_results.get("error") or check_results.get("check_errors"):
                 logger.info(f"    Checks completed with issues/errors for {os.path.basename(gen_stl_path)}.")
            else:
                 logger.info(f"    Checks completed successfully for {os.path.basename(gen_stl_path)}.")

        except GeometryCheckError as e:
            logger.error(f"    GeometryCheckError during checks for {os.path.basename(gen_stl_path)}: {e}", exc_info=True)
            error_result = {"generated_stl_path": gen_stl_path, "error": f"GeometryCheckError: {e}", "checks": {}}
            all_check_results.append(error_result)
        except Exception as e:
            # This might catch the malloc error if it propagates as a Python exception
            # Or it might catch other unexpected errors
            logger.critical(f"    UNEXPECTED CRITICAL ERROR during checks for {os.path.basename(gen_stl_path)}: {e}", exc_info=True)
            error_result = {"generated_stl_path": gen_stl_path, "error": f"Critical Unexpected Error: {e}", "checks": {}}
            all_check_results.append(error_result)
            # Decide if we should stop the whole script here or try to continue
            # For debugging malloc, stopping might be okay after the first crash.
            # print(f"Critical error encountered. Aborting check loop.")
            # break # Uncomment to stop after the first critical error

    check_duration = datetime.now() - check_start_time
    logger.info(f"--- Batch Geometry Checks Complete ({check_duration.total_seconds():.2f}s) ---")

    # --- Assemble and Save Results ---
    final_results_data = assemble_results_from_checks(run_dir, all_check_results, config, tasks_dict)

    results_filename = f"results{RESULTS_FILENAME_SUFFIX}"
    results_path = os.path.join(run_dir, results_filename)
    try:
        with open(results_path, 'w') as f:
            json.dump(final_results_data, f, indent=2)
        logger.info(f"Successfully saved geometry check results to: {results_path}")
    except IOError as e:
        logger.critical(f"Failed to save results file: {e}")
    except TypeError as e:
        logger.critical(f"Failed to serialize results to JSON: {e}")


    logger.info(f"--- CadEval Geometry Check Run Finished --- Target Dir: {run_dir}")

if __name__ == "__main__":
    main() 