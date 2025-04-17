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
import subprocess
import tempfile
import shutil
import glob
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

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
# --- Import moved functions --- Start ---
from scripts.generate_zoo import generate_stl_with_zoo
from scripts.result_assembler import assemble_final_results
# --- Import moved functions --- End ---

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
        "--prompts",
        nargs='+',
        type=str,
        default=None, # Default is None, meaning use 'default' prompt
        help="Optional list of specific prompt keys to use (must match keys in config.prompts). Uses 'default' if not specified.",
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
        # --- Pass only the openscad dictionary section --- 
        openscad_config_dict = config.get('openscad', {})
        if not openscad_config_dict:
             raise ConfigError("'openscad' section missing from configuration.")
        detected_version = validate_openscad_config(openscad_config_dict)
        # -------------------------------------------------
        logger.info(f"OpenSCAD validation successful (Version: {detected_version}).")
    except RenderError as e: # Catches ValidationError as well
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


    # --- Determine Prompts to Run ---
    prompts_to_run = args.prompts
    if prompts_to_run is None:
        prompts_to_run = ['default']
        logger.info("No prompts specified. Using default prompt.")
    
    # Validate that all requested prompts exist in config
    for prompt_key in prompts_to_run:
        prompt = config.get_prompt(prompt_key)
        if prompt is None:
            logger.error(f"Prompt key '{prompt_key}' not found in configuration. Available keys: {', '.join(config.get('prompts', {}).keys())}")
            sys.exit(1)
    
    logger.info(f"Running prompts: {', '.join(prompts_to_run)}")


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
    generation_results = [] # Stores results from generate_scad_for_task
    direct_stl_generation_results = [] # Stores results from generate_stl_with_zoo
    scad_paths_to_render = []
    # Unified map: Expected final STL path -> generation info (task, model, paths, etc.)
    stl_path_to_gen_info = {}

    for task_data in tasks_to_run:
        task_id = task_data.get("task_id", "unknown_task")
        task_description = task_data.get("description", "")

        for model_config in models_to_run:
            model_name = model_config.get("name", "unknown")
            provider = model_config.get("provider", "unknown")
            model_identifier_for_filename = f"{provider}_{model_name}".replace("/", "_")

            for prompt_key in prompts_to_run:
                logger.info(f"Starting Task: {task_id}, Model: {model_name}, Provider: {provider}, Prompt: {prompt_key}")

                for replicate_id in range(1, num_replicates + 1):
                    logger.info(f"  Replicate {replicate_id}/{num_replicates}")

                    base_filename = f"{task_id}_{model_identifier_for_filename}_{prompt_key}_rep{replicate_id}"
                    # Define the final expected STL path regardless of generation method
                    expected_stl_path = os.path.join(run_stl_dir, f"{base_filename}.stl")

                    # --- Conditional Generation --- START ---
                    if provider == 'zoo_cli':
                        # --- Zoo CLI Path ---
                        actual_scad_path = None # No SCAD file for Zoo
                        try:
                            # Use imported function
                            final_stl_path, error_msg, cmd_str, duration = generate_stl_with_zoo(
                                prompt=task_description,
                                output_stl_path=expected_stl_path, # Aim for the final path
                                model_config=model_config,
                                logger=logger
                            )
                            # Store Zoo result separately
                            direct_stl_generation_results.append({
                                "task_id": task_id,
                                "model_name": model_name,
                                "provider": provider,
                                "model_config_used": model_config,
                                "prompt_key_used": prompt_key,
                                "replicate_id": replicate_id,
                                "success": final_stl_path is not None,
                                "output_stl_path": final_stl_path, # This will be None on failure
                                "error": error_msg,
                                "command_executed": cmd_str,
                                "generation_duration_seconds": duration,
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                            })
                            # Add to map ONLY if successful
                            if final_stl_path:
                                logger.info(f"  Zoo STL generated successfully: {os.path.basename(final_stl_path)}")
                                stl_path_to_gen_info[final_stl_path] = {
                                    "task_id": task_id,
                                    "task_data": task_data,
                                    "model_config": model_config,
                                    "replicate_id": replicate_id,
                                    "prompt_key_used": prompt_key,
                                    "provider": provider,
                                    "scad_path": None # Explicitly None
                                }
                            else:
                                logger.error(f"  Zoo STL generation failed for Replicate {replicate_id}: {error_msg}")
                        except Exception as e:
                            # Handle unexpected errors during Zoo call
                            logger.error(f"  Unexpected error during Zoo CLI orchestration for Replicate {replicate_id}: {e}", exc_info=True)
                            direct_stl_generation_results.append({
                                "task_id": task_id, "model_name": model_name, "provider": provider,
                                "model_config_used": model_config, "prompt_key_used": prompt_key,
                                "replicate_id": replicate_id, "success": False, "output_stl_path": None,
                                "error": f"Orchestration error: {e}", "command_executed": None,
                                "generation_duration_seconds": None,
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                            })
                    else:
                        # --- Existing LLM -> SCAD Path ---
                        scad_output_path = os.path.join(run_scad_dir, f"{base_filename}.scad")
                        try:
                            gen_result = generate_scad_for_task(
                                task=task_data,
                                model_config=model_config,
                                output_dir=run_scad_dir,
                                replicate_id=replicate_id,
                                prompt_key=prompt_key
                            )
                            # --- Ensure consistency: Add/Rename prompt_key_used ---
                            if 'prompt_key' in gen_result and 'prompt_key_used' not in gen_result:
                                gen_result['prompt_key_used'] = gen_result['prompt_key']
                            # -------------------------------------------------------
                            generation_results.append(gen_result)
                            actual_scad_path = gen_result.get("output_path")
                            # Add to map ONLY if successful
                            if gen_result.get("success") and actual_scad_path and os.path.exists(actual_scad_path):
                                logger.info(f"  SCAD generated successfully: {os.path.basename(actual_scad_path)}")
                                scad_paths_to_render.append(actual_scad_path)
                                # Use EXPECTED STL path as the key
                                stl_path_to_gen_info[expected_stl_path] = {
                                    "task_id": task_id,
                                    "task_data": task_data,
                                    "model_config": model_config,
                                    "replicate_id": replicate_id,
                                    "prompt_key_used": prompt_key,
                                    "provider": provider,
                                    "scad_path": actual_scad_path # Store the SCAD path
                                }
                            elif not gen_result.get("success"):
                                logger.error(f"  SCAD generation failed for Replicate {replicate_id}: {gen_result.get('error')}")
                            else:
                                logger.warning(f"  SCAD generation reported success but file path invalid/missing for Replicate {replicate_id}: {actual_scad_path}")
                        except Exception as e:
                            # Handle unexpected errors during SCAD generation
                            logger.error(f"  Unexpected error during SCAD generation orchestration for Replicate {replicate_id}: {e}", exc_info=True)
                            generation_results.append({
                                "task_id": task_id, "model_name": model_name, "provider": provider,
                                "model_config_used": model_config, "replicate_id": replicate_id,
                                "output_path": scad_output_path, "success": False,
                                "error": f"Orchestration error: {e}", "prompt_used": None,
                                "prompt_key": prompt_key,
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                            })
                    # --- Conditional Generation --- END ---

    # --- Render SCAD Files ---
    render_results = []
    if scad_paths_to_render:
        logger.info(f"--- Starting Batch Rendering for {len(scad_paths_to_render)} SCAD files ---")
        render_start_time = time.time()

        for i, scad_file_path in enumerate(scad_paths_to_render):
            # Find corresponding gen info using scad_path (we need to find the right key in stl_path_to_gen_info)
            expected_stl_path_for_scad = None
            gen_info_for_scad = None
            for stl_key, gen_info in stl_path_to_gen_info.items():
                if gen_info.get("scad_path") == scad_file_path:
                    expected_stl_path_for_scad = stl_key
                    gen_info_for_scad = gen_info
                    break
            
            if not gen_info_for_scad:
                logger.warning(f"Could not find generation info for SCAD file '{scad_file_path}'. Skipping render.")
                continue

            task_id = gen_info_for_scad["task_data"].get("task_id", "unknown")
            model_name = gen_info_for_scad["model_config"].get("name", "unknown")
            scad_filename_base = os.path.basename(scad_file_path)

            logger.info(f"  [{i+1}/{len(scad_paths_to_render)}] Rendering Task='{task_id}', Model='{model_name}', File='{scad_filename_base}'...")

            # Use the expected STL directory (run_stl_dir)
            try:
                render_result = render_scad_file(
                    scad_path=scad_file_path,
                    output_dir=run_stl_dir,
                    # --- Pass only the openscad dictionary section --- 
                    openscad_config=config.get('openscad', {})
                    # -------------------------------------------------
                )
                # IMPORTANT: Update the render_result to include the expected STL path key
                render_result['expected_stl_path'] = expected_stl_path_for_scad
                render_results.append(render_result)

                if render_result.get("status") == "Success":
                    # Verify the output STL path matches the expected one
                    if render_result.get("stl_path") != expected_stl_path_for_scad:
                         logger.warning(f"    Rendered STL path '{render_result.get('stl_path')}' does not match expected path '{expected_stl_path_for_scad}'. Map might be incorrect.")
                    logger.info(f"    Render successful: {os.path.basename(render_result.get('stl_path', 'UNKNOWN'))}")
                else:
                    logger.error(f"    Render failed: Status='{render_result.get('status')}', Error='{render_result.get('error')}'")
                    # Add a failure entry to the map keyed by expected_stl_path?
                    # Or handle this during assembly?

            except Exception as e:
                 logger.error(f"    Unexpected error during rendering orchestration: {e}", exc_info=True)
                 # Add failure entry? How to link back?
                 # Storing failure in render_results is probably best
                 render_results.append({
                     "scad_path": scad_file_path,
                     "stl_path": None, # Render failed
                     "expected_stl_path": expected_stl_path_for_scad, # Store expected path
                     "status": "Orchestration Error",
                     "error": str(e),
                     "duration": None,
                     "summary_path": None
                 })
        # ... (rest of render summary) ...
    else:
        logger.info("Skipping rendering phase as no SCAD files were generated successfully.")


    # --- Batch Geometry Checks --- (Adapted for Zoo CLI)
    check_results_map = {} # Maps STL path to its check results dict

    # Collect STL paths from successful renders (non-Zoo)
    rendered_stl_paths = [res.get("stl_path") for res in render_results if res.get("status") == "Success" and res.get("stl_path")]
    # Collect STL paths from successful direct Zoo generation
    zoo_stl_paths = [res.get("output_stl_path") for res in direct_stl_generation_results if res.get("success") and res.get("output_stl_path")]

    stl_files_to_check = rendered_stl_paths + zoo_stl_paths

    if stl_files_to_check:
        logger.info(f"--- Starting Batch Geometry Checks for {len(stl_files_to_check)} successfully generated/rendered STL files ---")
        check_start_time = time.time()

        for i, stl_file_path in enumerate(stl_files_to_check):
            # Find generation info using the unified map keyed by STL path
            gen_info = stl_path_to_gen_info.get(stl_file_path)
            if not gen_info:
                 # This case might happen if a render succeeded but mapping failed earlier, or Zoo success was reported but mapping failed.
                 logger.warning(f"Could not find generation info for STL file '{stl_file_path}'. Skipping checks.")
                 continue

            task_data = gen_info["task_data"]
            task_id = task_data.get("task_id", "unknown")
            model_name = gen_info["model_config"].get("name", "unknown")
            provider = gen_info.get("provider", "unknown") # Get provider info
            stl_filename_base = os.path.basename(stl_file_path)

            logger.info(f"  [{i+1}/{len(stl_files_to_check)}] Checking Task='{task_id}', Model='{model_name}', Provider='{provider}', STL='{stl_filename_base}'...")

            # Get rendering info only if it was NOT a direct Zoo generation
            rendering_info_for_check = None
            if provider != 'zoo_cli':
                rendering_info_for_check = next((r for r in render_results if r.get("stl_path") == stl_file_path), None)
                # It's possible render succeeded but isn't in render_results due to an orchestration error before append
                if not rendering_info_for_check:
                    logger.error(f"    Could not find render result info for non-Zoo STL {stl_filename_base}. Skipping checks.")
                    check_results_map[stl_file_path] = {"error": f"Missing render info for {stl_filename_base}", "check_results_valid": False}
                    continue
            # For Zoo runs, rendering_info_for_check remains None

            try:
                reference_stl_rel_path = task_data.get("reference_stl")
                if not reference_stl_rel_path:
                     raise GeometryCheckError("Missing 'reference_stl' path in task definition.")
                # --- Use project_root defined at the top of the script ---
                reference_stl_abs_path = os.path.abspath(os.path.join(project_root, reference_stl_rel_path))
                if not os.path.exists(reference_stl_abs_path):
                    raise GeometryCheckError(f"Reference STL not found at resolved path: {reference_stl_abs_path}")

                check_results = perform_geometry_checks(
                    generated_stl_path=stl_file_path,
                    reference_stl_path=reference_stl_abs_path,
                    task_requirements=task_data.get("requirements", {}),
                    rendering_info=rendering_info_for_check, # Will be None for Zoo runs
                    # --- Pass only the geometry_check dictionary section --- 
                    geometry_config=config.get('geometry_check', {})
                    # ------------------------------------------------------
                )
                check_results_map[stl_file_path] = check_results # Key by STL path
                logger.info("    Checks completed.")

            except GeometryCheckError as e:
                logger.error(f"    Check Setup Failed: {e}")
                check_results_map[stl_file_path] = {"error": f"Check Setup Error: {e}", "check_results_valid": False}
            except Exception as e:
                logger.error(f"    Unexpected error during geometry check orchestration: {e}", exc_info=True)
                check_results_map[stl_file_path] = {"error": f"Orchestration Error: {e}", "check_results_valid": False}

        check_duration = time.time() - check_start_time
        successful_checks = sum(1 for res in check_results_map.values() if res.get("check_results_valid", True) and not res.get("error"))
        logger.info(f"--- Batch Geometry Checks Complete ({check_duration:.2f}s) ---")
        logger.info(f"Performed checks for {len(check_results_map)} files. Successful checks: {successful_checks}")
    else:
        logger.info("Skipping geometry check phase as no STL files were generated or rendered successfully.")


    # --- Assemble and Save Final Results ---
    logger.info("Calling assemble_final_results...") # Add log point
    final_results = assemble_final_results(
        generation_results=generation_results, # Pass non-Zoo results
        direct_stl_generation_results=direct_stl_generation_results, # Pass Zoo results
        render_results=render_results,
        check_results_map=check_results_map,
        stl_path_to_gen_info=stl_path_to_gen_info, # Pass the unified map
        logger=logger,
        project_root=project_root # Pass project_root here
    )
    logger.info(f"Finished assembling {len(final_results)} results.") # Add log point

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
