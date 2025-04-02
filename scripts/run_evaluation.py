#!/usr/bin/env python3
"""
CadEval Main Orchestration Script

This script coordinates the end-to-end evaluation pipeline:
1. Loads configuration and tasks.
2. Iterates through specified tasks and LLMs.
3. Calls scripts to generate SCAD code.
4. Calls scripts to render SCAD to STL.
5. Calls scripts to perform geometry checks.
6. Assembles and saves results to a JSON file.
"""

import os
import sys
import argparse
import json
import datetime
import logging
import time
from typing import List, Dict, Any

# Add project root to path to allow importing other scripts
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary functions/classes from other scripts
from scripts.config_loader import get_config, Config, ConfigError
from scripts.task_loader import load_tasks
from scripts.logger_setup import setup_logger, get_logger
from scripts.generate_scad import generate_scad_for_task # Assume this returns prompt_used
# --- Fix Missing Render Imports --- #
from scripts.render_scad import render_scad_file, validate_openscad_config, RenderError
from scripts.geometry_check import perform_geometry_checks, GeometryCheckError

# Initialize logger for this script
logger = get_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the CadEval evaluation pipeline.")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--tasks",
        nargs='+', # Allows specifying one or more task IDs
        type=str,
        default=None, # Default is None, meaning run all tasks found
        help="Optional list of specific task IDs to run (e.g., task1 task3). Runs all if not specified.",
    )
    parser.add_argument(
        "--models",
        nargs='+',
        type=str,
        default=None, # Default is None, meaning run all models in config
        help="Optional list of specific LLM model names to run (must match keys in config). Runs all configured models if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save the final results JSON file (default: results)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Unique identifier for this evaluation run (default: generated timestamp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None, # Default behavior handled in main
        help="Optional path to a file for logging output. Set to 'none' to disable file logging.", # Updated help text
    )

    return parser.parse_args()

# --- Helper Function for Final Result Assembly --- # Inserted Here #
def assemble_final_results(
    generation_results: List[Dict[str, Any]],
    render_results: List[Dict[str, Any]],
    check_results_map: Dict[str, Dict[str, Any]],
    scad_to_task_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merges results from generation, rendering, and checking into the final schema.
    """
    logger.info("Assembling final results...")
    final_results_list = []

    # Create a lookup map for render results keyed by SCAD path
    render_results_map = {res.get("scad_path"): res for res in render_results if res.get("scad_path")}

    # Iterate through each generation attempt, as this is the basis for each final entry
    for gen_result in generation_results:
        task_id = gen_result.get("task_id")
        model_identifier = gen_result.get("model") # e.g., "openai_o1_2024_12_17"
        scad_path = gen_result.get("output_path")
        model_config_used = gen_result.get("model_config_used", {}) # Get the stored config

        # --- Find corresponding task, render, and check data ---
        task_info = scad_to_task_map.get(scad_path) if scad_path else None
        render_info = render_results_map.get(scad_path) if scad_path else None
        check_info = check_results_map.get(scad_path) if scad_path else None

        # --- Initialize the final entry ---
        final_entry = {
            "task_id": task_id,
            "model_name": model_config_used.get("name"),
            "task_description": None,
            "reference_stl_path": None,
            "prompt_used": gen_result.get("prompt_used"),
            "llm_config": {
                 "provider": model_config_used.get("provider"),
                 "name": model_config_used.get("name"),
                 "temperature": model_config_used.get("temperature"),
                 "max_tokens": model_config_used.get("max_tokens")
             },
            "timestamp_utc": gen_result.get("timestamp"),
            "output_scad_path": None,
            "output_stl_path": None,
            "output_summary_json_path": None,
            "render_status": None,
            "render_duration_seconds": None,
            "render_error_message": None,
            "checks": {
                "check_render_successful": None,
                "check_is_watertight": None,
                "check_is_single_component": None,
                "check_bounding_box_accurate": None
            },
            "geometric_similarity_distance": None,
            "icp_fitness_score": None,
            "generation_error": gen_result.get("error") if not gen_result.get("success") else None,
            "check_error": None # Placeholder for check errors
        }

        # --- Populate from Task Info (if available) ---
        if task_info:
            task_data = task_info.get("task_data", {})
            final_entry["task_description"] = task_data.get("description")
            final_entry["reference_stl_path"] = task_data.get("reference_stl")

        # --- Populate Paths (Relative to Project Root) ---
        if scad_path and os.path.isabs(scad_path):
             try: final_entry["output_scad_path"] = os.path.relpath(scad_path, project_root)
             except ValueError: final_entry["output_scad_path"] = scad_path
        else: final_entry["output_scad_path"] = scad_path

        if render_info:
            stl_path = render_info.get("stl_path")
            if stl_path and os.path.isabs(stl_path):
                 try: final_entry["output_stl_path"] = os.path.relpath(stl_path, project_root)
                 except ValueError: final_entry["output_stl_path"] = stl_path
            else: final_entry["output_stl_path"] = stl_path

            summary_path = render_info.get("summary_path")
            if summary_path and os.path.isabs(summary_path):
                 try: final_entry["output_summary_json_path"] = os.path.relpath(summary_path, project_root)
                 except ValueError: final_entry["output_summary_json_path"] = summary_path
            else: final_entry["output_summary_json_path"] = summary_path

            final_entry["render_status"] = render_info.get("status")
            final_entry["render_duration_seconds"] = render_info.get("duration")
            final_entry["render_error_message"] = render_info.get("error")

        # --- Populate from Check Info (if available) ---
        if check_info:
            # Check if the checks themselves ran into an error (e.g., setup failure)
            # Use a key like 'check_errors' or a dedicated validity flag if added
            check_run_error = check_info.get("error") # Check for orchestrator/setup errors stored
            check_internal_errors = check_info.get("check_errors", []) # Check for errors within checks

            if check_run_error or not check_info.get("check_results_valid", True): # Handle explicit invalid flag or orchestrator error
                logger.warning(f"Check results for {scad_path} are marked as invalid or failed during setup. Error: {check_run_error or check_internal_errors}")
                final_entry["check_error"] = check_run_error or "; ".join(check_internal_errors)
            else:
                # Directly access the results from the check_info dictionary
                final_entry["checks"]["check_render_successful"] = check_info.get("check_render_successful")
                final_entry["checks"]["check_is_watertight"] = check_info.get("check_is_watertight")
                final_entry["checks"]["check_is_single_component"] = check_info.get("check_is_single_component")
                final_entry["checks"]["check_bounding_box_accurate"] = check_info.get("check_bounding_box_accurate")

                # Directly access top-level similarity scores
                final_entry["geometric_similarity_distance"] = check_info.get("geometric_similarity_distance")
                final_entry["icp_fitness_score"] = check_info.get("icp_fitness_score")

                # Combine internal check errors if any were recorded
                if check_internal_errors:
                     final_entry["check_error"] = "; ".join(check_internal_errors)

        final_results_list.append(final_entry)

    logger.info(f"Assembled {len(final_results_list)} final result entries.")
    return final_results_list

def main():
    """Main execution function."""
    args = parse_arguments()
    start_run_time = datetime.datetime.now(datetime.timezone.utc) # Record start time

    # --- Setup Logging (Revised) ---
    # Ensure output dir exists before setting up file logging there
    os.makedirs(args.output_dir, exist_ok=True)
    # Define log file path (use None if args.log_file is explicitly set to skip file logging)
    log_file_path = args.log_file
    if args.log_file is None:
        # Default log file path if --log-file is not provided
        log_file_path = os.path.join(args.output_dir, f"run_{args.run_id}.log")
        print(f"Logging to console and file: {log_file_path}") # Inform user about default file logging
    elif args.log_file.lower() == 'none': # Allow explicitly disabling file log
         log_file_path = None
         print("Logging to console only.")
    else:
        print(f"Logging to console and file: {log_file_path}")

    # Configure the ROOT logger. Handlers added here will apply to all child loggers.
    # Pass console=True explicitly to ensure console output.
    setup_logger(
        name='', # Empty string targets the root logger
        level=logging.getLevelName(args.log_level.upper()), # Get integer level from string
        log_file=log_file_path,
        console=True # Explicitly enable console handler on the root logger
    )
    # No need to re-get the logger; the initial 'logger = get_logger(__name__)'
    # will now inherit handlers/levels from the configured root logger.

    logger.info(f"--- Starting CadEval Run --- Run ID: {args.run_id}")
    logger.info(f"Command Line Arguments: {vars(args)}")
    logger.info(f"Logging Level: {args.log_level.upper()}") # Added log level info
    logger.info(f"Results will be saved in: {args.output_dir}")

    # --- Load Configuration ---
    try:
        config = get_config(args.config)
        logger.info(f"Successfully loaded configuration from: {args.config}")
    except (ConfigError, FileNotFoundError) as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    # --- Validate OpenSCAD Setup ---
    try:
        openscad_version = validate_openscad_config(config)
        logger.info(f"OpenSCAD validation successful (Version: {openscad_version}).")
    except (RenderError, Exception) as e:
        logger.error(f"OpenSCAD validation failed: {e}", exc_info=True)
        logger.error("Cannot proceed with rendering. Please check your configuration ('openscad' section) and OpenSCAD installation.")
        sys.exit(1)

    # --- Load Tasks ---
    tasks_dir = config.get('directories.tasks', 'tasks') # Use directories key
    try:
        all_tasks = load_tasks(tasks_dir)
        if not all_tasks:
            logger.error(f"No tasks found in directory: {tasks_dir}")
            sys.exit(1)
        logger.info(f"Loaded {len(all_tasks)} tasks from {tasks_dir}")

        # --- Adjust Task Filtering for List --- #
        if args.tasks:
            requested_task_ids = set(args.tasks)
            tasks_to_run = [task for task in all_tasks if task.get('task_id') in requested_task_ids]
            found_task_ids = {task.get('task_id') for task in tasks_to_run}
            missing_tasks = requested_task_ids - found_task_ids
            if missing_tasks:
                logger.warning(f"Specified task IDs not found: {', '.join(missing_tasks)}")
            if not tasks_to_run:
                logger.error("No specified tasks were found. Exiting.")
                sys.exit(1)
            logger.info(f"Running specified tasks: {', '.join(sorted([t.get('task_id', 'Unknown') for t in tasks_to_run]))}")
        else:
            tasks_to_run = all_tasks # tasks_to_run is a list of task dictionaries
            logger.info("Running all found tasks.")

    except (FileNotFoundError, Exception) as e:
        logger.error(f"Error loading tasks from {tasks_dir}: {e}", exc_info=True)
        sys.exit(1)

    # --- Determine Models to Run --- Changed Logic ---
    try:
        # Read the list of model dictionaries from 'llm.models'
        configured_models_list = config.get_required('llm.models')
        if not isinstance(configured_models_list, list):
            raise ConfigError("'llm.models' in config must be a list.")
        if not configured_models_list:
             raise ConfigError("No models found in 'llm.models' list in config.")

    except ConfigError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
         logger.error(f"Unexpected error reading models from config: {e}", exc_info=True)
         sys.exit(1)

    # Filter models based on command-line argument args.models (comparing against 'name')
    if args.models:
        requested_model_names = set(args.models)
        # models_to_run_configs is the list of dictionaries for the models we will run
        models_to_run_configs = [m for m in configured_models_list if m.get("name") in requested_model_names]
        found_model_names = {m.get("name") for m in models_to_run_configs}
        missing_models = requested_model_names - found_model_names
        if missing_models:
            logger.warning(f"Specified models not found in config: {', '.join(missing_models)}")
        if not models_to_run_configs:
            logger.error("None of the specified models were found in the configuration. Exiting.")
            sys.exit(1)
        logger.info(f"Running specified models: {', '.join(sorted(list(found_model_names)))}") # Log found names
    else:
        models_to_run_configs = configured_models_list # Run all configured models
        all_model_names = [m.get("name", "Unknown") for m in models_to_run_configs]
        logger.info(f"Running all configured models: {', '.join(sorted(all_model_names))}")

    # --- Initialize Results Containers ---
    generation_results: List[Dict[str, Any]] = []
    scad_to_task_map: Dict[str, Dict[str, Any]] = {}
    render_results: List[Dict[str, Any]] = []
    check_results_map: Dict[str, Dict[str, Any]] = {} # Map scad_path -> check_results_dict
    generated_outputs_dir = config.get('directories.generated_outputs', 'generated_outputs')
    os.makedirs(generated_outputs_dir, exist_ok=True)

    logger.info(f"Starting evaluation loop for {len(tasks_to_run)} tasks and {len(models_to_run_configs)} models...")

    # --- SCAD Generation Loop --- Adjust Loop Iteration ---
    total_attempts = len(tasks_to_run) * len(models_to_run_configs)
    current_attempt = 0
    # Iterate through the list of task dictionaries
    for task_data in tasks_to_run:
        task_id = task_data.get("task_id")
        if not task_id:
             logger.warning(f"Skipping task data due to missing 'task_id': {task_data}")
             continue # Skip if a task is missing its ID

        # Iterate through the list of model configuration dictionaries
        for model_config in models_to_run_configs:
            current_attempt += 1
            model_name = model_config.get("name", "Unknown") # Get name for logging etc.
            logger.info(f"--- [{current_attempt}/{total_attempts}] Task='{task_id}', Model='{model_name}' ---") # Updated loop start log

            # Get prompt parameters from config if they exist
            prompt_params = config.get('prompt_parameters', None) # Assuming this is still global

            # --- 1. Generate SCAD ---
            logger.info(f"  Generating SCAD...") # Added generation start log
            generation_success = False # Added flag
            scad_path = None # Added variable init
            try:
                # Pass the whole model_config dictionary directly
                generation_result = generate_scad_for_task(
                    task=task_data,
                    model_config=model_config, # Pass the specific model's config dict
                    output_dir=generated_outputs_dir,
                    prompt_params=prompt_params
                )

                generation_results.append(generation_result) # Store generation attempt result

                if generation_result.get("success"):
                    scad_path = generation_result.get("output_path")
                    if scad_path and os.path.exists(scad_path):
                        logger.info(f"  SCAD generated successfully: {os.path.basename(scad_path)}") # Added success log
                        generation_success = True # Mark as success for later steps
                        # Add to map for later lookup during checks
                        scad_to_task_map[scad_path] = {
                            "task_id": task_id,
                            "model_name": model_name, # Store the model name (e.g., "gpt-4o-mini")
                            "task_data": task_data
                            # model_config is already stored inside generation_result
                        }
                    else:
                         error_msg = f"SCAD generation reported success, but output file missing or path not returned: {scad_path}" # Capture error message
                         logger.error(f"  SCAD generation failed: {error_msg}") # Updated error log
                         generation_result["error"] = generation_result.get("error", "") + f"; Orchestrator: {error_msg}" # Append orchestrator error
                         generation_result["success"] = False
                else:
                    error_msg = generation_result.get("error", "Unknown generation error")
                    logger.error(f"  SCAD generation failed: {error_msg}") # Updated error log

            except Exception as e:
                logger.error(f"  Unexpected error during SCAD generation: {e}", exc_info=True) # Updated error log
                # Create a basic error entry for results_list
                error_result = {
                    "task_id": task_id,
                    "model": model_name, # Use the simple name here
                    "model_config_used": model_config, # Include config even on orchestrator error
                    "timestamp": datetime.datetime.now().isoformat(),
                    "success": False,
                    "output_path": None,
                    "error": f"Orchestrator Error during generation: {str(e)}",
                    "duration_seconds": 0 # Or calculate if possible
                }
                generation_results.append(error_result)

    logger.info(f"--- SCAD Generation Phase Complete ---")
    logger.info(f"Attempted {total_attempts} generations.")
    scad_files_generated = list(scad_to_task_map.keys()) # Get list of successfully generated files
    logger.info(f"Successfully generated {len(scad_files_generated)} SCAD files.")
    if not scad_files_generated:
         logger.warning("No SCAD files were successfully generated. Skipping rendering and checking.")
         # Skip to saving results and exit


    # --- Batch Render ---
    generated_stls = {} # Map scad_path -> stl_path for successful renders # Added map
    if scad_files_generated:
        logger.info(f"--- Starting Batch Rendering for {len(scad_files_generated)} SCAD files ---")
        render_start_time = time.time()
        render_output_dir = generated_outputs_dir
        os.makedirs(render_output_dir, exist_ok=True)

        for i, scad_path in enumerate(scad_files_generated, 1):
            # --- Add pre-render log with Task/Model info --- # Refined lookup
            task_info = scad_to_task_map.get(scad_path)
            task_id = task_info.get("task_id", "UnknownTask") if task_info else "UnknownTask"
            model_name = task_info.get("model_name", "UnknownModel") if task_info else "UnknownModel"

            logger.info(f"  [{i}/{len(scad_files_generated)}] Rendering Task='{task_id}', Model='{model_name}', File='{os.path.basename(scad_path)}'...") # Updated pre-render log
            render_success = False # Added flag
            stl_path = None # Added variable init
            try:
                 result = render_scad_file(scad_path=scad_path, output_dir=render_output_dir, config=config)
                 render_results.append(result)
                 if result.get("status") == "Success":
                      stl_path = result.get('stl_path')
                      logger.info(f"    Render successful: {os.path.basename(stl_path) if stl_path else 'N/A'}") # Added success log
                      render_success = True # Set flag
                      if stl_path:
                          generated_stls[scad_path] = stl_path # Store successful STL path
                 else:
                      logger.error(f"    Render failed: Status='{result.get('status')}', Error='{result.get('error')}'") # Updated error log
            except Exception as e:
                 logger.error(f"    Unexpected error during rendering: {e}", exc_info=True) # Updated error log
                 # Create an error entry for this file
                 error_render_result = {
                     "scad_path": scad_path,
                     "status": "Failed",
                     "stl_path": None,
                     "summary_path": None,
                     "duration": 0,
                     "return_code": None,
                     "stdout": None,
                     "stderr": None,
                     "error": f"Orchestrator Error during rendering: {str(e)}"
                 }
                 render_results.append(error_render_result)

        render_duration = time.time() - render_start_time
        successful_renders = len(generated_stls) # Count successful renders based on stored paths
        logger.info(f"--- Batch Rendering Complete ({render_duration:.2f}s) ---")
        logger.info(f"Successfully rendered {successful_renders}/{len(scad_files_generated)} files.")
    else:
         logger.info("Skipping Batch Rendering as no SCAD files were generated.")


    # --- Batch Check ---
    check_results_map: Dict[str, Dict[str, Any]] = {} # Map scad_path -> check_results_dict
    # Use the generated_stls map which contains only successfully rendered files
    stls_to_check = list(generated_stls.items()) # List of (scad_path, stl_path) tuples # Changed source for loop

    if stls_to_check:
        logger.info(f"--- Starting Batch Geometry Checks for {len(stls_to_check)} successfully rendered STL files ---")
        check_start_time = time.time()
        reference_dir = config.get('directories.reference', 'reference') # Get reference dir from config

        for i, (scad_path, generated_stl_path) in enumerate(stls_to_check, 1): # Changed loop variable names
            # Lookup task info using scad_path
            task_info = scad_to_task_map.get(scad_path)
            if not task_info:
                logger.error(f"  [{i}/{len(stls_to_check)}] Cannot find original task info for SCAD '{os.path.basename(scad_path)}' (STL: '{os.path.basename(generated_stl_path)}'). Skipping checks.") # Updated error log
                continue

            task_id = task_info.get("task_id", "UnknownTask") # Added default
            model_name = task_info.get("model_name", "UnknownModel") # Added default
            task_data = task_info.get("task_data", {}) # Added default

            logger.info(f"  [{i}/{len(stls_to_check)}] Checking Task='{task_id}', Model='{model_name}', STL='{os.path.basename(generated_stl_path)}'...") # Updated pre-check log

            reference_stl_path = None # Init var
            check_success = False # Added flag
            # --- Get Inputs for perform_geometry_checks ---
            # 1. Generated STL Path (already have)
            # 2. Reference STL Path
            try:
                 # Ensure reference path is relative to project root or absolute
                 raw_ref_path = task_data.get('reference_stl')
                 if not raw_ref_path:
                      raise ValueError("Missing 'reference_stl' in task data.")
                 # Assume reference path in YAML is relative to project root
                 reference_stl_path_resolved = os.path.join(project_root, raw_ref_path) # Adjust if assumption is wrong
                 if not os.path.exists(reference_stl_path_resolved):
                      raise FileNotFoundError(f"Reference STL not found at resolved path: {reference_stl_path_resolved}")
                 reference_stl_path = reference_stl_path_resolved # Store the valid path
                 logger.debug(f"    Reference STL: {reference_stl_path}")

                 # 3. Task Requirements # Moved inside try block
                 task_requirements = task_data.get('requirements', {})
                 if not task_requirements:
                      logger.warning(f"    No 'requirements' found in task data for {task_id}. Checks requiring them may fail or be skipped.")

                 # 4. Rendering Info (Find corresponding render_info dictionary) # Moved inside try block
                 render_info = next((r for r in render_results if r.get("scad_path") == scad_path), None)
                 if not render_info:
                      raise ValueError("Could not find corresponding render results for this STL.")

                 # 5. Config (already have 'config') # Moved inside try block

                 # --- Call Check Function ---
                 check_results = perform_geometry_checks(
                     generated_stl_path=generated_stl_path,
                     reference_stl_path=reference_stl_path,
                     task_requirements=task_requirements,
                     rendering_info=render_info, # Pass the whole dict from render_results
                     config=config
                 )
                 check_results_map[scad_path] = check_results # Store results mapped by original SCAD path
                 logger.info(f"    Checks completed.") # Added success log
                 check_success = True # Mark check as successful

            except (ValueError, FileNotFoundError, TypeError) as e:
                 logger.error(f"    Check Setup Failed: Invalid or missing reference STL path. Error: {e}") # Updated error log
                 check_results_map[scad_path] = {"error": f"Check Setup Error: Invalid reference STL - {str(e)}", "check_results_valid": False}
            except GeometryCheckError as e:
                 logger.error(f"    GeometryCheckError: {e}") # Updated error log
                 check_results_map[scad_path] = {"error": f"GeometryCheckError: {str(e)}", "check_results_valid": False}
            except Exception as e:
                 logger.error(f"    Unexpected error during geometry checks: {e}", exc_info=True) # Updated error log
                 check_results_map[scad_path] = {"error": f"Orchestrator Error during checks: {str(e)}", "check_results_valid": False}


        check_duration = time.time() - check_start_time
        successful_checks = sum(1 for scad, res in check_results_map.items() if not res.get("error") and res.get("check_results_valid", True)) # Approximate success count # Added calculation
        logger.info(f"--- Batch Geometry Checks Complete ({check_duration:.2f}s) ---")
        logger.info(f"Performed checks for {len(check_results_map)} files. Successful checks: {successful_checks}") # Log success count # Added success count log
    else:
         logger.info("Skipping Batch Geometry Checks as no STL files were successfully rendered.")


    # --- Assemble Final Results ---
    logger.info("Assembling final results...") # Added info message
    final_results = assemble_final_results(
        generation_results=generation_results,
        render_results=render_results,
        check_results_map=check_results_map,
        scad_to_task_map=scad_to_task_map,
        # config=config # Config is no longer passed here # Removed config pass
    )

    # --- Save Final Results ---
    results_filename = os.path.join(args.output_dir, f"results_{args.run_id}.json")
    try:
        # Ensure results_list contains JSON-serializable data only
        # (Numpy types might need conversion: np.float64 -> float)
        with open(results_filename, 'w') as f:
            json.dump(final_results, f, indent=4)
        # Final log message indicating success and file path
        logger.info(f"Successfully saved results to: {results_filename}") # Moved log inside try
    except IOError as e:
        logger.error(f"Failed to write results to {results_filename}: {e}")
        logger.error(f"Attempted to save results to: {results_filename}") # Still log the path
    except TypeError as e:
         logger.error(f"Failed to serialize results to JSON (check data types): {e}")
         logger.error(f"Attempted to save results to: {results_filename}") # Still log the path


    logger.info(f"--- CadEval Run Finished --- Run ID: {args.run_id}")

if __name__ == "__main__":
    main() 