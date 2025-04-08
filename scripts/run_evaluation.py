# scripts/run_evaluation.py
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
    # --output-dir now refers to the PARENT directory for all runs
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Parent directory to store run-specific output folders (default: results)", # Updated help text
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
    # --log-file argument is removed as it's now implicitly inside the run directory
    # parser.add_argument(
    #     "--log-file",
    #     type=str,
    #     default=None, # Default behavior handled in main
    #     help="Optional path to a file for logging output. Set to 'none' to disable file logging.", # Updated help text
    # )

    return parser.parse_args()

# --- Helper Function for Final Result Assembly --- (Unchanged from previous version)
def assemble_final_results(
    generation_results: List[Dict[str, Any]],
    render_results: List[Dict[str, Any]],
    check_results_map: Dict[str, Dict[str, Any]],
    scad_to_task_map: Dict[str, Dict[str, Any]],
    logger: logging.Logger
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

        # --- Find corresponding task, render, and check data ---\
        task_info = scad_to_task_map.get(scad_path) if scad_path else None
        render_info = render_results_map.get(scad_path) if scad_path else None
        check_info = check_results_map.get(scad_path) if scad_path else None

        # --- Initialize the final entry --- Add new fields
        final_entry = {
            "task_id": task_id,
            "replicate_id": None, # Placeholder for replicate ID
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
            "checks": { # Add new check flags here
                "check_render_successful": None,
                "check_is_watertight": None,
                "check_is_single_component": None,
                "check_bounding_box_accurate": None,
                "check_volume_passed": None, # New
                "check_hausdorff_passed": None # New
            },
            "geometric_similarity_distance": None, # Chamfer
            "icp_fitness_score": None,
            "hausdorff_95p_distance": None,
            "hausdorff_99p_distance": None, # New
            "reference_volume_mm3": None, # New
            "generated_volume_mm3": None, # New
            "reference_bbox_mm": None, # New
            "generated_bbox_aligned_mm": None, # New
            "generation_error": gen_result.get("error") if not gen_result.get("success") else None,
            "check_error": None # Consolidated check errors
        }

        # --- Populate from Task Info (if available) ---
        if task_info:
            task_data = task_info.get("task_data", {})
            final_entry["task_description"] = task_data.get("description")
            # Keep reference STL path relative to project root as defined in YAML
            final_entry["reference_stl_path"] = task_data.get("reference_stl")
            # Get replicate_id from the task_info (mapped from SCAD path)
            final_entry["replicate_id"] = task_info.get("replicate_id")

        # --- Populate Paths (Relative to Project Root) ---
        if scad_path and os.path.isabs(scad_path):
             try: final_entry["output_scad_path"] = os.path.relpath(scad_path, project_root)
             except ValueError: final_entry["output_scad_path"] = scad_path # Fallback if different drive etc.
        else: final_entry["output_scad_path"] = scad_path # Should not happen with new structure

        if render_info:
            stl_path = render_info.get("stl_path")
            if stl_path and os.path.isabs(stl_path):
                 try: final_entry["output_stl_path"] = os.path.relpath(stl_path, project_root)
                 except ValueError: final_entry["output_stl_path"] = stl_path
            else: final_entry["output_stl_path"] = stl_path

            # TODO: Decide if render summary JSON path also needs updating/moving
            summary_path = render_info.get("summary_path")
            if summary_path and os.path.isabs(summary_path):
                 try: final_entry["output_summary_json_path"] = os.path.relpath(summary_path, project_root)
                 except ValueError: final_entry["output_summary_json_path"] = summary_path
            else: final_entry["output_summary_json_path"] = summary_path

            final_entry["render_status"] = render_info.get("status")
            final_entry["render_duration_seconds"] = render_info.get("duration")
            final_entry["render_error_message"] = render_info.get("error")

        # --- Populate from Check Info (if available) --- Update population logic
        if check_info:
            check_run_error = check_info.get("error") # Check for orchestrator/setup errors stored
            check_internal_errors = check_info.get("check_errors", []) # Check for errors within individual checks

            # Combine all errors
            all_errors = []
            if check_run_error: all_errors.append(check_run_error)
            if check_internal_errors: all_errors.extend(check_internal_errors)
            if all_errors:
                final_entry["check_error"] = "; ".join(all_errors)
                logger.warning(f"Check issues for {os.path.basename(scad_path or 'UNKNOWN')}: {final_entry['check_error']}")

            # Populate boolean checks from the sub-dictionary if available
            checks_sub_dict = check_info.get("checks", {})
            # Explicitly handle boolean/None assignment AND ensure Python bool type
            for check_key in final_entry["checks"]:
                check_value = checks_sub_dict.get(check_key)
                if check_value is None:
                    final_entry["checks"][check_key] = None
                else:
                    # This converts any truthy value (including numpy.bool_(True)) to True,
                    # and any falsy value (including numpy.bool_(False)) to False.
                    final_entry["checks"][check_key] = bool(check_value)

            # Populate metric values
            final_entry["geometric_similarity_distance"] = check_info.get("geometric_similarity_distance")
            final_entry["icp_fitness_score"] = check_info.get("icp_fitness_score")
            final_entry["hausdorff_95p_distance"] = check_info.get("hausdorff_95p_distance")
            final_entry["hausdorff_99p_distance"] = check_info.get("hausdorff_99p_distance") # New
            final_entry["reference_volume_mm3"] = check_info.get("reference_volume_mm3") # New
            final_entry["generated_volume_mm3"] = check_info.get("generated_volume_mm3") # New
            final_entry["reference_bbox_mm"] = check_info.get("reference_bbox_mm") # New
            final_entry["generated_bbox_aligned_mm"] = check_info.get("generated_bbox_aligned_mm") # New

        final_results_list.append(final_entry)

    logger.info(f"Assembled {len(final_results_list)} final result entries.")
    return final_results_list


def main():
    """Main execution function."""
    args = parse_arguments()
    start_run_time = datetime.datetime.now(datetime.timezone.utc) # Record start time

    # --- Define Run-Specific Directories ---
    run_base_dir = os.path.join(args.output_dir, args.run_id)
    run_scad_dir = os.path.join(run_base_dir, "scad")
    run_stl_dir = os.path.join(run_base_dir, "stl")
    # Potentially add more dirs here later (e.g., logs, summaries)

    # --- Create Directories ---
    try:
        os.makedirs(run_base_dir, exist_ok=True)
        os.makedirs(run_scad_dir, exist_ok=True)
        os.makedirs(run_stl_dir, exist_ok=True)
        # We could create a dedicated logs dir too:
        # run_log_dir = os.path.join(run_base_dir, "logs")
        # os.makedirs(run_log_dir, exist_ok=True)
        print(f"Created output directories for run {args.run_id} inside {args.output_dir}")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        sys.exit(1)


    # --- Setup Logging ---
    # Log file will now always be inside the run directory
    log_file_path = os.path.join(run_base_dir, f"run_{args.run_id}.log")
    print(f"Logging to console and file: {log_file_path}")

    setup_logger(
        name='', # Root logger
        level=logging.getLevelName(args.log_level.upper()),
        log_file=log_file_path,
        console=True
    )

    # --- Get Logger Instance AFTER Setup ---
    logger = get_logger(__name__) # Now gets the properly configured/mocked logger

    # Now the logger instance obtained earlier will work with the setup
    logger.info(f"--- Starting CadEval Run --- Run ID: {args.run_id}")
    logger.info(f"Command Line Arguments: {vars(args)}")
    logger.info(f"Logging Level: {args.log_level.upper()}")
    logger.info(f"Run Output Directory: {run_base_dir}") # Log the specific run dir


    # --- Load Configuration ---
    try:
        config = get_config(args.config)
        logger.info(f"Successfully loaded configuration from: {args.config}")
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        sys.exit(1)

    # --- Validate OpenSCAD ---
    try:
        # Pass the whole config object to the validation function
        # It will access 'openscad.executable_path' and 'openscad.minimum_version' internally
        detected_version = validate_openscad_config(config) # Pass config object

        # The function raises RenderError on failure, so if we get here, it's valid.
        logger.info(f"OpenSCAD validation successful (Version: {detected_version}).")

    except RenderError as e: # Catch the specific error raised by the validation function
        logger.error(f"OpenSCAD validation failed: {e}")
        sys.exit(1)
    except ConfigError as e: # Catch potential errors if config keys are missing *before* calling validate
        logger.error(f"OpenSCAD configuration error: {e}")
        sys.exit(1)
    except Exception as e: # Catch other unexpected errors
        logger.error(f"Error during OpenSCAD validation orchestration: {e}")
        sys.exit(1)


    # --- Load Tasks ---
    try:
        tasks_dir = config.get("tasks.directory", "tasks") # Default to "tasks" if not in config
        schema_path = config.get("tasks.schema_path") # Can be None
        all_tasks = load_tasks(tasks_dir=tasks_dir, schema_path=schema_path)
        if not all_tasks:
            logger.error("No tasks were loaded. Exiting.")
            sys.exit(1)
        logger.info(f"Loaded {len(all_tasks)} tasks from {tasks_dir}")
    except Exception as e: # Catch errors from task_loader
        logger.error(f"Failed to load tasks: {e}")
        sys.exit(1)

    # --- Filter Tasks and Models based on CLI args ---
    tasks_to_run = []
    if args.tasks:
        requested_task_ids = set(args.tasks)
        tasks_to_run = [task for task in all_tasks if task.get("task_id") in requested_task_ids]
        loaded_task_ids = {task.get("task_id") for task in tasks_to_run}
        missing_tasks = requested_task_ids - loaded_task_ids
        if missing_tasks:
            logger.warning(f"Could not find requested tasks: {', '.join(missing_tasks)}")
        if not tasks_to_run:
             logger.error("None of the requested tasks were found. Exiting.")
             sys.exit(1)
        logger.info(f"Running specified tasks: {', '.join(loaded_task_ids)}")
    else:
        tasks_to_run = all_tasks # Run all loaded tasks
        logger.info("Running all found tasks.")


    models_to_run = []
    available_models = config.get("llm.models", [])
    if not available_models:
         logger.error("No models defined in configuration file under 'llm.models'. Exiting.")
         sys.exit(1)

    if args.models:
        requested_model_names = set(args.models)
        # Ensure model config includes 'provider' and 'name' for uniqueness if needed
        # For now, assume model names (keys in config like 'gemini-1.5-flash') are unique identifiers
        available_model_map = {model_conf.get("name"): model_conf for model_conf in available_models if model_conf.get("name")}

        models_to_run = [available_model_map[name] for name in requested_model_names if name in available_model_map]

        loaded_model_names = {model.get("name") for model in models_to_run}
        missing_models = requested_model_names - loaded_model_names
        if missing_models:
            logger.warning(f"Could not find requested models in config: {', '.join(missing_models)}")
        if not models_to_run:
             logger.error("None of the requested models were found in the configuration. Exiting.")
             sys.exit(1)
        logger.info(f"Running specified models: {', '.join(loaded_model_names)}")
    else:
        models_to_run = available_models # Run all configured models
        logger.info(f"Running all configured models: {', '.join([m.get('name','?') for m in models_to_run])}")


    # --- Get Replicates Setting ---
    try:
        num_replicates = int(config.get_required('evaluation.num_replicates'))
        if num_replicates < 1:
            logger.warning("evaluation.num_replicates must be >= 1. Setting to 1.")
            num_replicates = 1
    except (ConfigError, ValueError, TypeError) as e:
        logger.warning(f"Invalid or missing 'evaluation.num_replicates' in config ({e}). Setting to 1.")
        num_replicates = 1
    logger.info(f"Running {num_replicates} replicate(s) for each task/model combination.")

    # --- Main Evaluation Loop ---
    generation_results = []
    scad_paths_to_render = []
    scad_to_task_map = {}

    for task_data in tasks_to_run:
        task_id = task_data.get("task_id", "unknown_task")
        for model_config in models_to_run:
            model_identifier_for_filename = f"{model_config['provider']}_{model_config['name']}".replace("/", "_") # Sanitize for filename
            logger.info(f"Starting Task: {task_id}, Model: {model_config['name']}")

            for replicate_id in range(1, num_replicates + 1):
                logger.info(f"  Replicate {replicate_id}/{num_replicates}")

                # Generate unique filename including model and replicate ID
                base_filename = f"{task_id}_{model_identifier_for_filename}_rep{replicate_id}"
                scad_output_path = os.path.join(run_scad_dir, f"{base_filename}.scad")

                try:
                    # Call the correct function: generate_scad_for_task
                    gen_result = generate_scad_for_task(
                        task=task_data,
                        model_config=model_config,
                        output_dir=run_scad_dir, # Pass directory, function creates filename
                        replicate_id=replicate_id # Pass replicate_id
                        # config object is implicitly loaded by generate_scad_for_task
                        # logger object is implicitly available via get_logger
                    )
                    generation_results.append(gen_result)

                    # Use the path returned by the function for mapping and rendering list
                    actual_scad_path = gen_result.get("output_path")

                    if gen_result.get("success") and actual_scad_path and os.path.exists(actual_scad_path):
                        logger.info(f"  SCAD generated successfully: {os.path.basename(actual_scad_path)}")
                        scad_paths_to_render.append(actual_scad_path)
                        # Map the actual SCAD path to task/model info
                        scad_to_task_map[actual_scad_path] = {"task_id": task_id, "task_data": task_data, "model_config": model_config, "replicate_id": replicate_id}
                    elif not gen_result.get("success"):
                        logger.error(f"  SCAD generation failed for Replicate {replicate_id}: {gen_result.get('error')}")
                    else: # Success=True but path issue
                        logger.warning(f"  SCAD generation reported success but file path invalid/missing for Replicate {replicate_id}: {actual_scad_path}")

                except Exception as e:
                    logger.error(f"  Unexpected error during SCAD generation orchestration for Replicate {replicate_id}: {e}", exc_info=True)
                    # Record failure in results (use the intended path)
                    generation_results.append({
                        "task_id": task_id,
                        "model": f"{model_config['provider']}/{model_config['name']}",
                        "model_config_used": model_config,
                        "replicate_id": replicate_id, # Add replicate ID to error result
                        "output_path": scad_output_path, # Log the intended path
                        "success": False,
                        "error": f"Orchestration error: {e}",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "prompt_used": None
                    })

    # --- Render SCAD Files ---
    render_results = []
    if scad_paths_to_render:
        logger.info(f"--- Starting Batch Rendering for {len(scad_paths_to_render)} SCAD files ---")
        render_start_time = time.time()

        for i, scad_file_path in enumerate(scad_paths_to_render):
            task_info = scad_to_task_map.get(scad_file_path)
            task_id = task_info["task_data"].get("task_id", "unknown")
            model_name = task_info["model_config"].get("name", "unknown")
            scad_filename_base = os.path.basename(scad_file_path)

            logger.info(f"  [{i+1}/{len(scad_paths_to_render)}] Rendering Task='{task_id}', Model='{model_name}', File='{scad_filename_base}'...")

            # Construct STL output path
            stl_filename = scad_filename_base.replace(".scad", ".stl")
            stl_output_path = os.path.join(run_stl_dir, stl_filename) # Use new STL dir

            try:
                # Call render function with correct argument names and config
                render_result = render_scad_file(
                    scad_path=scad_file_path,
                    output_dir=run_stl_dir, # Pass the directory for STL output
                    config=config # Pass the config object
                )
                render_results.append(render_result)

                if render_result.get("status") == "Success":
                     logger.info(f"    Render successful: {os.path.basename(stl_output_path)}")
                else:
                     logger.error(f"    Render failed: Status='{render_result.get('status')}', Error='{render_result.get('error')}'")

            except Exception as e:
                 logger.error(f"    Unexpected error during rendering orchestration: {e}", exc_info=True)
                 render_results.append({
                     "scad_path": scad_file_path,
                     "stl_path": stl_output_path,
                     "status": "Orchestration Error",
                     "error": str(e),
                     "duration": None,
                     "summary_path": None # Or create summary path even on error?
                 })

        render_duration = time.time() - render_start_time
        successful_renders = [res for res in render_results if res.get("status") == "Success"]
        logger.info(f"--- Batch Rendering Complete ({render_duration:.2f}s) ---")
        logger.info(f"Successfully rendered {len(successful_renders)}/{len(scad_paths_to_render)} files.")
    else:
        logger.info("Skipping rendering phase as no SCAD files were generated successfully.")


    # --- Batch Geometry Checks ---
    check_results_map = {} # Maps SCAD path to its check results dict
    stl_files_to_check = [res.get("stl_path") for res in render_results if res.get("status") == "Success" and res.get("stl_path")]

    if stl_files_to_check:
        logger.info(f"--- Starting Batch Geometry Checks for {len(stl_files_to_check)} successfully rendered STL files ---")
        check_start_time = time.time()

        # Need to map STL paths back to SCAD paths to retrieve task/model info
        stl_to_scad_map = {res.get("stl_path"): res.get("scad_path") for res in render_results if res.get("stl_path") and res.get("scad_path")}

        for i, stl_file_path in enumerate(stl_files_to_check):
            scad_file_path = stl_to_scad_map.get(stl_file_path)
            if not scad_file_path:
                logger.warning(f"Could not map STL file '{stl_file_path}' back to its source SCAD. Skipping checks.")
                continue

            task_info = scad_to_task_map.get(scad_file_path)
            if not task_info:
                 logger.warning(f"Could not find task/model info for SCAD file '{scad_file_path}'. Skipping checks for '{stl_file_path}'.")
                 continue

            task_data = task_info["task_data"]
            task_id = task_data.get("task_id", "unknown")
            model_name = task_info["model_config"].get("name", "unknown")
            stl_filename_base = os.path.basename(stl_file_path)

            logger.info(f"  [{i+1}/{len(stl_files_to_check)}] Checking Task='{task_id}', Model='{model_name}', STL='{stl_filename_base}'...")

            # Find the render results corresponding to this STL file
            render_info_for_check = next((r for r in render_results if r.get("stl_path") == stl_file_path), None)
            if not render_info_for_check:
                logger.error(f"    Could not find render result info for {stl_filename_base}. Skipping checks.")
                check_results_map[scad_file_path] = {"error": f"Missing render info for {stl_filename_base}", "check_results_valid": False}
                continue

            try:
                # Ensure the reference STL path exists and is resolved correctly from project root
                reference_stl_rel_path = task_data.get("reference_stl")
                if not reference_stl_rel_path:
                     raise GeometryCheckError("Missing 'reference_stl' path in task definition.")

                # Resolve reference path relative to project root
                reference_stl_abs_path = os.path.abspath(os.path.join(project_root, reference_stl_rel_path))

                if not os.path.exists(reference_stl_abs_path):
                    raise GeometryCheckError(f"Reference STL not found at resolved path: {reference_stl_abs_path}")


                # Perform checks using the generated STL path and the resolved reference path
                check_results = perform_geometry_checks(
                    generated_stl_path=stl_file_path,
                    reference_stl_path=reference_stl_abs_path,
                    task_requirements=task_data.get("requirements", {}),
                    rendering_info=render_info_for_check, # Pass the render result dict
                    config=config # Pass full config for check-specific parameters
                )
                check_results_map[scad_file_path] = check_results
                logger.info("    Checks completed.") # Add more detail from check_results if desired

            except GeometryCheckError as e: # Catch errors specific to check setup/logic
                logger.error(f"    Check Setup Failed: {e}")
                check_results_map[scad_file_path] = {"error": f"Check Setup Error: {e}", "check_results_valid": False} # Mark as invalid
            except Exception as e:
                logger.error(f"    Unexpected error during geometry check orchestration: {e}", exc_info=True)
                check_results_map[scad_file_path] = {"error": f"Orchestration Error: {e}", "check_results_valid": False} # Mark as invalid

        check_duration = time.time() - check_start_time
        successful_checks = sum(1 for res in check_results_map.values() if res.get("check_results_valid", True) and not res.get("error")) # Crude count of valid runs
        logger.info(f"--- Batch Geometry Checks Complete ({check_duration:.2f}s) ---")
        logger.info(f"Performed checks for {len(check_results_map)} files. Successful checks: {successful_checks}") # Revisit definition of 'successful'
    else:
        logger.info("Skipping geometry check phase as no STL files were rendered successfully.")


    # --- Assemble and Save Final Results ---
    final_results = assemble_final_results(
        generation_results,
        render_results,
        check_results_map,
        scad_to_task_map,
        logger
    )

    # Save to the run-specific directory
    results_filename = f"results_{args.run_id}.json"
    results_path = os.path.join(run_base_dir, results_filename)

    try:
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"Successfully saved results to: {results_path}")
    except IOError as e:
        logger.error(f"Failed to save final results JSON to {results_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving final results: {e}")

    logger.info(f"--- CadEval Run Finished --- Run ID: {args.run_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fallback logger if setup failed before main() completed
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)
