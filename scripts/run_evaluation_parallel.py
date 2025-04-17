#!/usr/bin/env python3
"""
CadEval Parallel Evaluation Orchestration Script

This script coordinates the evaluation pipeline using parallel execution:
1. Loads configuration and tasks.
2. Creates job lists for all task/model/prompt/replicate combinations.
3. Executes LLM and Zoo generation jobs in parallel.
4. (Future Phases) Executes rendering and geometry checks in parallel.
5. (Future Phases) Assembles and saves aggregated results.
"""

import os
import sys
import argparse
import json
import datetime
import logging
import time
import concurrent.futures # Added for parallelism
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# --- Add project root to path ---
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import necessary functions/classes from other scripts ---
from scripts.config_loader import get_config, Config, ConfigError
from scripts.task_loader import load_tasks
from scripts.logger_setup import setup_logger, get_logger
# Import the core functions needed for generation
from scripts.generate_scad import generate_scad_for_task
from scripts.generate_zoo import generate_stl_with_zoo
# Import functions needed for future phases (rendering, checks, assembly)
from scripts.render_scad import render_scad_file, validate_openscad_config, RenderError
from scripts.geometry_check import perform_geometry_checks, GeometryCheckError
from scripts.result_assembler import assemble_final_results

# --- Argument Parsing (Restored) ---
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the CadEval parallel evaluation pipeline.")

    # Arguments copied from run_evaluation.py
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--tasks", nargs='+', type=str, default=None,
        help="Optional list of specific task IDs to run. Runs all if not specified.",
    )
    parser.add_argument(
        "--models", nargs='+', type=str, default=None,
        help="Optional list of specific LLM model names to run. Runs all configured models if not specified.",
    )
    parser.add_argument(
        "--prompts", nargs='+', type=str, default=None,
        help="Optional list of specific prompt keys to use. Uses 'default' if not specified.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Parent directory to store run-specific output folders (default: results)",
    )
    parser.add_argument(
        "--run-id", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Unique identifier for this evaluation run (default: generated timestamp)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument( # New argument for replicates
        "--replicates", type=int, default=None,
        help="Number of replicates per task. Overrides config file if specified.",
    )

    # --- New arguments for concurrency ---
    cpu_count = os.cpu_count() or 1 # Default to 1 if cpu_count() returns None
    parser.add_argument(
        "--max-workers-llm", type=int, default=10, # Default higher for I/O bound LLM calls
        help="Maximum number of parallel threads for LLM API calls (default: 10)",
    )
    parser.add_argument(
        "--max-workers-zoo", type=int, default=max(1, cpu_count // 2), # Default based on CPU cores for local CLI
        help=f"Maximum number of parallel processes for Zoo CLI calls (default: {max(1, cpu_count // 2)})",
    )
    parser.add_argument(
        "--max-workers-render", type=int, default=cpu_count, # Default based on CPU cores
        help=f"Maximum number of parallel processes for OpenSCAD rendering (default: {cpu_count})",
    )
    parser.add_argument(
        "--max-workers-check", type=int, default=cpu_count, # Default based on CPU cores
        help=f"Maximum number of parallel processes for Geometry Checks (default: {cpu_count})",
    )
    # --- End of new arguments ---

    return parser.parse_args()

# --- Helper function definitions (moved to top level for pickling) ---

# --- Helper function to wrap generation calls for executors ---
# We need separate wrappers because the function signatures and return types differ slightly
# and because one uses threads (LLM) and the other processes (Zoo).

def run_llm_generation_job(job_info: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper to run a single LLM generation job."""
    task_data = job_info["task_data"]
    model_config = job_info["model_config"]
    output_dir = job_info["output_dir"]
    replicate_id = job_info["replicate_id"]
    prompt_key = job_info["prompt_key"]
    # Logger setup should be handled globally, but need logger instance
    logger = get_logger(__name__) # Get logger configured in main()

    logger.debug(f"Starting LLM job: Task {task_data['task_id']}, Model {model_config['name']}, Rep {replicate_id}")
    try:
        result = generate_scad_for_task(
            task=task_data,
            model_config=model_config,
            output_dir=output_dir,
            replicate_id=replicate_id,
            prompt_key=prompt_key
            # Note: generate_scad_for_task gets its own logger instance internally currently
        )
        if 'prompt_key' in result and 'prompt_key_used' not in result:
            result['prompt_key_used'] = result['prompt_key']
        # --- Add task_data to the result ---
        result['task_data'] = task_data
        # ----------------------------------
        logger.debug(f"Finished LLM job: Task {task_data['task_id']}, Model {model_config['name']}, Rep {replicate_id}, Success: {result.get('success')}")
        return result
    except Exception as e:
        logger.error(f"Unhandled exception in LLM generation job wrapper: {e}", exc_info=True)
        # Return a failure dictionary consistent with generate_scad_for_task format
        # --- Also add task_data to the error result ---
        error_result = {
            "task_id": task_data.get("task_id"),
            "model_name": model_config.get("name"),
            "provider": model_config.get("provider"),
            "model_config_used": model_config,
            "replicate_id": replicate_id,
            "prompt_key": prompt_key,
            "prompt_key_used": prompt_key,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "success": False,
            "output_path": None,
            "error": f"Parallel wrapper error: {e}",
            "llm_duration_seconds": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "output_scad_path": None,
            "prompt_used": None,
            "task_data": task_data # Add here too for consistency if needed later
        }
        return error_result

def run_zoo_generation_job(job_info: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper to run a single Zoo generation job."""
    task_description = job_info["task_description"]
    output_stl_path = job_info["output_stl_path"]
    model_config = job_info["model_config"]
    task_id = job_info["task_id"]
    prompt_key = job_info["prompt_key"]
    replicate_id = job_info["replicate_id"]
    provider = model_config["provider"]
    model_name = model_config["name"]
    # Logger setup should be handled globally, but need logger instance
    logger = get_logger(__name__) # Get logger configured in main()

    logger.debug(f"Starting Zoo job: Task {task_id}, Model {model_name}, Rep {replicate_id}")
    try:
        final_stl_path, error_msg, cmd_str, duration = generate_stl_with_zoo(
            prompt=task_description,
            output_stl_path=output_stl_path,
            model_config=model_config,
            logger=logger # Pass the logger instance
        )
        logger.debug(f"Finished Zoo job: Task {task_id}, Model {model_name}, Rep {replicate_id}, Success: {final_stl_path is not None}")
        # Construct result dictionary similar to run_evaluation.py's direct_stl_generation_results
        return {
            "task_id": task_id,
            "model_name": model_name,
            "provider": provider,
            "model_config_used": model_config,
            "prompt_key_used": prompt_key, # Use the prompt key from the job
            "replicate_id": replicate_id,
            "success": final_stl_path is not None,
            "output_stl_path": final_stl_path,
            "error": error_msg,
            "command_executed": cmd_str,
            "generation_duration_seconds": duration,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            # --- Add task_data to the result ---
            "task_data": job_info["task_data"] # Get it from the original job_info
            # ----------------------------------
        }
    except Exception as e:
        logger.error(f"Unhandled exception in Zoo generation job wrapper: {e}", exc_info=True)
        # Return a failure dictionary
        # --- Also add task_data to the error result ---
        error_result = {
            "task_id": task_id,
            "model_name": model_name,
            "provider": provider,
            "model_config_used": model_config,
            "prompt_key_used": prompt_key,
            "replicate_id": replicate_id,
            "success": False,
            "output_stl_path": None,
            "error": f"Parallel wrapper error: {e}",
            "command_executed": None,
            "generation_duration_seconds": None,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task_data": job_info["task_data"] # Add here too
        }
        return error_result

# --- Render Job Wrapper (Moved to Top Level) ---
def run_render_job(job_info: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper to run a single OpenSCAD render job."""
    scad_path = job_info["scad_path"]
    output_dir = job_info["output_dir"]
    # --- Accept openscad config dictionary directly --- 
    openscad_config_dict = job_info["openscad_config_dict"]
    # --- Get logger (assumes logger is initialized in the worker process if needed,
    # --- or relies on base logging if not explicitly configured in worker) ---
    # --- For simplicity, we might rely on the fact that render_scad_file uses its own logger setup,
    # --- or pass logger name / config details if needed.
    # --- Let's assume render_scad_file handles its logging for now.
    logger_instance = get_logger(__name__) # Get logger instance

    scad_filename = os.path.basename(scad_path)
    logger_instance.debug(f"Starting Render job for: {scad_filename}")

    try:
        # Call the actual render function from render_scad.py
        # --- Pass the dictionary where the config object was expected --- 
        result = render_scad_file(
            scad_path=scad_path,
            output_dir=output_dir,
            openscad_config=openscad_config_dict 
        )
        logger_instance.debug(f"Finished Render job for: {scad_filename}, Status: {result.get('status')}")
        # Note: render_scad_file already returns a dict with needed keys
        # (scad_path, stl_path, status, error, duration, summary_path)
        return result
    except Exception as e:
        logger_instance.error(f"Unhandled exception in render job wrapper for {scad_filename}: {e}", exc_info=True)
        # Return a failure dictionary consistent with render_scad_file format
        return {
            "scad_path": scad_path,
            "stl_path": None,
            "status": "Orchestration Error",
            "error": f"Parallel wrapper error: {e}",
            "duration": None,
            "summary_path": None
        }

# --- Check Job Wrapper (Top Level) ---
def run_check_job(job_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Wrapper to run a single geometry check job."""
    generated_stl_path = job_info["generated_stl_path"]
    task_data = job_info["task_data"]
    rendering_info = job_info["rendering_info"] # Can be None
    geometry_config = job_info["geometry_config"] # Geometry check config dict
    logger_instance = get_logger(__name__) # Get logger instance

    stl_filename = os.path.basename(generated_stl_path)
    task_id = task_data.get("task_id", "unknown")
    logger_instance.debug(f"Starting Check job for: Task {task_id}, STL {stl_filename}")

    # --- Resolve Reference STL Path --- START ---
    # We need the project root to resolve the relative reference path from task_data
    # Since this runs in a separate process, project_root needs to be available.
    # Option 1: Pass project_root in job_info (simplest)
    # Option 2: Re-calculate project_root within the worker (less ideal)
    # Let's go with Option 1. We'll add project_root to job_info creation.
    project_root = job_info.get("project_root")
    if not project_root:
         # Fallback if project_root wasn't passed (shouldn't happen with Option 1)
         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
         logger_instance.warning(f"Project root not passed to check job worker, calculated as {project_root}")

    reference_stl_rel_path = task_data.get("reference_stl")
    reference_stl_abs_path = None
    check_error = None
    if not reference_stl_rel_path:
        check_error = "Missing 'reference_stl' path in task definition."
    else:
        reference_stl_abs_path = os.path.abspath(os.path.join(project_root, reference_stl_rel_path))
        if not os.path.exists(reference_stl_abs_path):
            check_error = f"Reference STL not found at resolved path: {reference_stl_abs_path}"
    # --- Resolve Reference STL Path --- END ---

    try:
        if check_error: # If reference path failed, create minimal error result
             results = {
                "generated_stl_path": generated_stl_path,
                "reference_stl_path": reference_stl_rel_path, # Store relative path even if resolution failed
                "error": f"Check Setup Error: {check_error}",
                "checks": {},
                "check_errors": [check_error]
             }
             logger_instance.error(f"Check Setup Failed for {stl_filename}: {check_error}")
        else: # Proceed with checks if reference path is okay
            results = perform_geometry_checks(
                generated_stl_path=generated_stl_path,
                reference_stl_path=reference_stl_abs_path, # Pass absolute path
                task_requirements=task_data.get("requirements", {}),
                rendering_info=rendering_info, # Pass render info (can be None)
                geometry_config=geometry_config # Pass the specific config dict
            )
        logger_instance.debug(f"Finished Check job for: {stl_filename}")
        # Return the path (for keying results) and the results dictionary
        return generated_stl_path, results

    except Exception as e:
        logger_instance.error(f"Unhandled exception in check job wrapper for {stl_filename}: {e}", exc_info=True)
        # Return a generic error structure
        error_result = {
            "generated_stl_path": generated_stl_path,
            "reference_stl_path": reference_stl_rel_path,
            "error": f"Parallel wrapper error: {e}",
            "checks": {},
            "check_errors": [f"Parallel wrapper error: {e}"]
        }
        return generated_stl_path, error_result

# --- Main Execution ---
def main():
    """Main parallel execution function."""
    args = parse_arguments()
    start_run_time = datetime.datetime.now(datetime.timezone.utc)

    # --- Setup Directories (Copied from run_evaluation.py) ---
    run_base_dir = os.path.join(args.output_dir, args.run_id)
    run_scad_dir = os.path.join(run_base_dir, "scad")
    run_stl_dir = os.path.join(run_base_dir, "stl")
    try:
        os.makedirs(run_base_dir, exist_ok=True)
        os.makedirs(run_scad_dir, exist_ok=True)
        os.makedirs(run_stl_dir, exist_ok=True)
        print(f"Created output directories for run {args.run_id} inside {args.output_dir}")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        sys.exit(1)

    # --- Setup Logging (Copied from run_evaluation.py) ---
    log_file_path = os.path.join(run_base_dir, f"run_{args.run_id}.log")
    print(f"Logging to console and file: {log_file_path}")
    setup_logger(
        name='', level=logging.getLevelName(args.log_level.upper()),
        log_file=log_file_path, console=True
    )
    logger = get_logger(__name__) # Get logger instance AFTER setup

    logger.info(f"--- Starting CadEval Parallel Run --- Run ID: {args.run_id}")
    logger.info(f"Command Line Arguments: {vars(args)}")
    logger.info(f"Max Workers: LLM={args.max_workers_llm}, Zoo={args.max_workers_zoo}, Render={args.max_workers_render}, Check={args.max_workers_check}")

    # --- Load Config, Validate OpenSCAD, Load Tasks (Copied) ---
    try:
        config = get_config(args.config)
        logger.info(f"Successfully loaded configuration from: {args.config}")
        # --- Pass only the openscad dictionary section to validator --- 
        openscad_config_dict = config.get('openscad', {})
        if not openscad_config_dict:
             raise ConfigError("'openscad' section missing from configuration.")
        detected_version = validate_openscad_config(openscad_config_dict)
        # ----------------------------------------------------------
        logger.info(f"OpenSCAD validation successful (Version: {detected_version}).")
        tasks_dir = config.get("tasks.directory", "tasks")
        schema_path = config.get("tasks.schema_path")
        all_tasks = load_tasks(tasks_dir=tasks_dir, schema_path=schema_path)
        if not all_tasks:
            logger.error("No tasks were loaded. Exiting.")
            sys.exit(1)
        logger.info(f"Loaded {len(all_tasks)} tasks from {tasks_dir}")
    except (ConfigError, RenderError, Exception) as e:
        logger.error(f"Setup error (Config/OpenSCAD/Tasks): {e}", exc_info=True)
        sys.exit(1)

    # --- Filter Tasks, Models, Prompts (Copied) ---
    # (Assuming the filtering logic from run_evaluation.py is sufficient)
    tasks_to_run = []
    # ... [Copy task filtering logic from run_evaluation.py lines ~647-662] ...
    if args.tasks:
        requested_task_ids = set(args.tasks)
        tasks_to_run = [task for task in all_tasks if task.get("task_id") in requested_task_ids]
        loaded_task_ids = {task.get("task_id") for task in tasks_to_run}
        missing_tasks = requested_task_ids - loaded_task_ids
        if missing_tasks: logger.warning(f"Could not find requested tasks: {', '.join(missing_tasks)}")
        if not tasks_to_run: logger.error("None of the requested tasks were found. Exiting."); sys.exit(1)
        logger.info(f"Running specified tasks: {', '.join(sorted(list(loaded_task_ids)))}")
    else:
        tasks_to_run = all_tasks
        logger.info(f"Running all {len(tasks_to_run)} found tasks.")

    models_to_run = []
    # ... [Copy model filtering logic from run_evaluation.py lines ~665-685] ...
    available_models = config.get("llm.models", [])
    if not available_models: logger.error("No models defined in config 'llm.models'. Exiting."); sys.exit(1)
    if args.models:
        requested_model_names = set(args.models)
        available_model_map = {model_conf.get("name"): model_conf for model_conf in available_models if model_conf.get("name")}
        models_to_run = [available_model_map[name] for name in requested_model_names if name in available_model_map]
        loaded_model_names = {model.get("name") for model in models_to_run}
        missing_models = requested_model_names - loaded_model_names
        if missing_models: logger.warning(f"Could not find requested models in config: {', '.join(missing_models)}")
        if not models_to_run: logger.error("None of the requested models found in config. Exiting."); sys.exit(1)
        logger.info(f"Running specified models: {', '.join(sorted(list(loaded_model_names)))}")
    else:
        models_to_run = available_models
        logger.info(f"Running all configured models: {', '.join(sorted([m.get('name','?') for m in models_to_run]))}")

    prompts_to_run = args.prompts or ['default']
    # ... [Copy prompt validation logic from run_evaluation.py lines ~693-698] ...
    for prompt_key in prompts_to_run:
        prompt = config.get_prompt(prompt_key)
        if prompt is None:
            logger.error(f"Prompt key '{prompt_key}' not found in config. Available: {', '.join(config.get('prompts', {}).keys())}")
            sys.exit(1)
    logger.info(f"Running prompts: {', '.join(prompts_to_run)}")

    # --- Get Replicates (Handle CLI override) ---
    num_replicates = args.replicates # Use CLI arg if provided
    if num_replicates is None: # Otherwise, use config file
        try:
            num_replicates = int(config.get_required('evaluation.num_replicates'))
        except (ConfigError, ValueError, TypeError) as e:
            logger.warning(f"Invalid or missing 'evaluation.num_replicates' in config ({e}). Setting to 1.")
            num_replicates = 1
    if num_replicates < 1:
        logger.warning("Number of replicates must be >= 1. Setting to 1.")
        num_replicates = 1
    logger.info(f"Running {num_replicates} replicate(s) for each task/model/prompt combination.")


    # =========================================================================
    # --- Phase 1: Parallel Generation ---
    # =========================================================================
    logger.info("--- Phase 1: Starting Parallel Generation ---")
    start_gen_time = time.time()
    llm_jobs = []
    zoo_jobs = []
    total_jobs = 0

    # --- Create Job Descriptions ---
    for task_data in tasks_to_run:
        task_id = task_data.get("task_id", "unknown_task")
        task_description = task_data.get("description", "")

        for model_config in models_to_run:
            model_name = model_config.get("name", "unknown")
            provider = model_config.get("provider", "unknown")
            model_identifier_for_filename = f"{provider}_{model_name}".replace("/", "_")

            for prompt_key in prompts_to_run:
                for replicate_id in range(1, num_replicates + 1):
                    total_jobs += 1
                    base_filename = f"{task_id}_{model_identifier_for_filename}_{prompt_key}_rep{replicate_id}"

                    if provider == 'zoo_cli':
                        job = {
                            "task_id": task_id,
                            "task_description": task_description,
                            "output_stl_path": os.path.join(run_stl_dir, f"{base_filename}.stl"),
                            "model_config": model_config,
                            "prompt_key": prompt_key,
                            "replicate_id": replicate_id,
                            "task_data": task_data
                        }
                        zoo_jobs.append(job)
                    else: # Standard LLM -> SCAD
                        job = {
                            "task_data": task_data,
                            "model_config": model_config,
                            "output_dir": run_scad_dir, # generate_scad creates filename internally
                            "replicate_id": replicate_id,
                            "prompt_key": prompt_key
                        }
                        llm_jobs.append(job)

    logger.info(f"Created {len(llm_jobs)} LLM generation jobs and {len(zoo_jobs)} Zoo generation jobs.")
    logger.info(f"Total generation jobs to run: {total_jobs}")

    # --- Execute Jobs Concurrently ---
    generation_results = [] # Results from LLM jobs
    direct_stl_generation_results = [] # Results from Zoo jobs
    stl_path_to_gen_info = {} # NEW: Map STL paths back to original generation info

    # Run LLM jobs (I/O bound - use ThreadPoolExecutor)
    if llm_jobs:
        logger.info(f"Starting LLM ThreadPoolExecutor with max_workers={args.max_workers_llm}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers_llm) as executor:
            future_to_job = {executor.submit(run_llm_generation_job, job): job for job in llm_jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    generation_results.append(result)
                except Exception as exc:
                    task_id = job['task_data'].get('task_id', 'unknown')
                    model_name = job['model_config'].get('name', 'unknown')
                    rep_id = job['replicate_id']
                    logger.error(f'LLM job (Task {task_id}, Model {model_name}, Rep {rep_id}) generated an exception: {exc}')
                    # Append a failure result if exception wasn't caught in wrapper
                    generation_results.append({
                        "task_id": task_id, "model_name": model_name, "provider": job['model_config'].get('provider'),
                        "model_config_used": job['model_config'], "replicate_id": rep_id,
                        "prompt_key": job['prompt_key'], "prompt_key_used": job['prompt_key'],
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "success": False, "output_path": None, "error": f"Executor exception: {exc}",
                         "llm_duration_seconds": None, "prompt_tokens": None, "completion_tokens": None,
                         "output_scad_path": None, "prompt_used": None,
                         "task_data": job['task_data'] # Add here too for consistency if needed later
                    })
        logger.info(f"Finished {len(generation_results)} LLM generation jobs.")

    # Run Zoo jobs (CPU/local resource bound - use ProcessPoolExecutor)
    if zoo_jobs:
        logger.info(f"Starting Zoo ProcessPoolExecutor with max_workers={args.max_workers_zoo}...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers_zoo) as executor:
            future_to_job = {executor.submit(run_zoo_generation_job, job): job for job in zoo_jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    direct_stl_generation_results.append(result)
                    # Populate stl_path_to_gen_info directly for successful Zoo jobs
                    if result.get("success") and result.get("output_stl_path"):
                        stl_path_to_gen_info[result["output_stl_path"]] = result
                except Exception as exc:
                    task_id = job.get('task_id', 'unknown')
                    model_name = job['model_config'].get('name', 'unknown')
                    rep_id = job.get('replicate_id')
                    logger.error(f'Zoo job (Task {task_id}, Model {model_name}, Rep {rep_id}) generated an exception: {exc}')
                     # Append a failure result if exception wasn't caught in wrapper
                    direct_stl_generation_results.append({
                        "task_id": task_id, "model_name": model_name, "provider": job['model_config'].get('provider'),
                        "model_config_used": job['model_config'], "prompt_key_used": job['prompt_key'],
                        "replicate_id": rep_id, "success": False, "output_stl_path": None,
                        "error": f"Executor exception: {exc}", "command_executed": None,
                        "generation_duration_seconds": None,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "task_data": job.get("task_data") # Add here too
                    })
        logger.info(f"Finished {len(direct_stl_generation_results)} Zoo generation jobs.")

    gen_duration = time.time() - start_gen_time
    logger.info(f"--- Phase 1: Parallel Generation Complete ({gen_duration:.2f}s) ---")

    # --- Phase 1 Gate: Log summary of generation results ---
    llm_success_count = sum(1 for r in generation_results if r.get("success"))
    zoo_success_count = sum(1 for r in direct_stl_generation_results if r.get("success"))
    # logger.info(f"Generation Summary: LLM Success={llm_success_count}/{len(llm_jobs)}, Zoo Success={zoo_success_count}/{len(zoo_jobs)}") # Removed for Phase 4

    # =========================================================================
    # --- Phase 2: Parallel Rendering --- 
    # =========================================================================
    logger.info("--- Phase 2: Starting Parallel Rendering ---")
    start_render_time = time.time()
    render_jobs = []
    render_results = [] # List to store results from render jobs

    # --- Create Render Job Descriptions --- 
    # Filter successful SCAD generations that have a valid output path
    scad_gens_to_render = [ 
        gen_res for gen_res in generation_results 
        if gen_res.get("success") and gen_res.get("output_path") and os.path.exists(gen_res["output_path"])
    ]
    
    if scad_gens_to_render:
        logger.info(f"Found {len(scad_gens_to_render)} successful SCAD generations to render.")
        for gen_result in scad_gens_to_render:
            scad_path = gen_result["output_path"] 
            # We also need the original gen_info to link back later for assembly, 
            # but render_scad_file doesn't need it. We can pass necessary parts.
            job = {
                "scad_path": scad_path,
                "output_dir": run_stl_dir, # Render to the main STL output dir for the run
                # --- Pass openscad config dict instead of whole object ---
                "openscad_config_dict": config.get('openscad', {}) 
                # -------------------------------------------------------
            }
            render_jobs.append(job)
    else:
        logger.info("No successful SCAD generations found to render.")

    # --- Execute Render Jobs Concurrently --- 
    if render_jobs:
        logger.info(f"Starting Render ProcessPoolExecutor with max_workers={args.max_workers_render}...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers_render) as executor:
            future_to_job = {executor.submit(run_render_job, job): job for job in render_jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    render_results.append(result)
                except Exception as exc:
                    scad_filename = os.path.basename(job.get('scad_path', 'unknown_scad'))
                    logger.error(f'Render job ({scad_filename}) generated an exception in executor: {exc}')
                    # Append a failure result if exception wasn't caught in wrapper
                    render_results.append({
                        "scad_path": job.get('scad_path'),
                        "stl_path": None,
                        "status": "Executor Error",
                        "error": f"Executor exception: {exc}",
                        "duration": None,
                        "summary_path": None
                    })
        logger.info(f"Finished processing {len(render_results)} render results.")

    render_duration = time.time() - start_render_time
    logger.info(f"--- Phase 2: Parallel Rendering Complete ({render_duration:.2f}s) ---")

    # --- Populate stl_path_to_gen_info for successful Render jobs --- 
    # Need to link render results back to the *original* LLM generation info
    scad_path_to_gen_info = {gr["output_path"]: gr for gr in generation_results if gr.get("output_path")} 
    for res in render_results:
        if res.get("status") == "Success" and res.get("stl_path"):
            original_gen_info = scad_path_to_gen_info.get(res.get("scad_path"))
            if original_gen_info:
                stl_path_to_gen_info[res["stl_path"]] = original_gen_info
            else:
                logger.warning(f"Could not find original generation info for successfully rendered STL: {res.get('stl_path')} (from SCAD: {res.get('scad_path')})")
    # -----------------------------------------------------------

    # --- Phase 2 Gate: Log summary of render results --- 
    render_success_count = sum(1 for r in render_results if r.get("status") == "Success")
    # logger.info(f"Rendering Summary: Success={render_success_count}/{len(render_jobs)}") # Removed for Phase 4

    # =========================================================================
    # --- Phase 3: Parallel Geometry Checks --- 
    # =========================================================================
    logger.info("--- Phase 3: Starting Parallel Geometry Checks ---")
    start_check_time = time.time()
    check_jobs = []
    check_results_map = {} # Map STL path -> check result dict

    # --- Create a lookup for render results by SCAD path ---
    render_results_lookup = {res.get("scad_path"): res for res in render_results if res.get("scad_path")}

    # --- Identify all successful STL files and their context ---
    stls_to_check_context = []

    # 1. STLs from successful renders
    for gen_res in generation_results:
        if gen_res.get("success"):
            scad_path = gen_res.get("output_path")
            render_info = render_results_lookup.get(scad_path)
            if render_info and render_info.get("status") == "Success":
                stl_path = render_info.get("stl_path")
                task_data = gen_res.get("task_data") # Retrieve task_data saved in Phase 1 job
                if stl_path and os.path.exists(stl_path) and task_data:
                    stls_to_check_context.append({
                        "generated_stl_path": stl_path,
                        "task_data": task_data,
                        "rendering_info": render_info # Pass render info for potential use
                    })
                elif not task_data:
                     logger.warning(f"Missing task_data for successful render of {scad_path}, cannot create check job.")

    # 2. STLs from successful direct Zoo generation
    for zoo_res in direct_stl_generation_results:
        if zoo_res.get("success"):
            stl_path = zoo_res.get("output_stl_path")
            task_data = zoo_res.get("task_data") # Retrieve task_data saved in Phase 1 job
            if stl_path and os.path.exists(stl_path) and task_data:
                stls_to_check_context.append({
                    "generated_stl_path": stl_path,
                    "task_data": task_data,
                    "rendering_info": None # No rendering info for Zoo
                })
            elif not task_data:
                 logger.warning(f"Missing task_data for successful zoo generation {stl_path}, cannot create check job.")

    logger.info(f"Found {len(stls_to_check_context)} STL files to check.")

    # --- Create Check Job Descriptions ---
    geometry_config_dict = config.get('geometry_check', {}) # Extract geometry config section
    if not geometry_config_dict:
        logger.warning("Geometry check configuration ('geometry_check') missing or empty in main config.")

    for context in stls_to_check_context:
        job = {
            "generated_stl_path": context["generated_stl_path"],
            "task_data": context["task_data"],
            "rendering_info": context["rendering_info"],
            "geometry_config": geometry_config_dict,
            "project_root": project_root # Pass project_root for reference path resolution
        }
        check_jobs.append(job)

    # --- Execute Check Jobs Concurrently ---
    if check_jobs:
        logger.info(f"Starting Check ProcessPoolExecutor with max_workers={args.max_workers_check}...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers_check) as executor:
            future_to_job = {executor.submit(run_check_job, job): job for job in check_jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                stl_path_key = job["generated_stl_path"] # Key is the STL path
                try:
                    # run_check_job returns tuple: (stl_path, result_dict)
                    _, result_dict = future.result()
                    check_results_map[stl_path_key] = result_dict
                except Exception as exc:
                    stl_filename = os.path.basename(stl_path_key)
                    logger.error(f'Check job ({stl_filename}) generated an exception in executor: {exc}')
                    # Store an error placeholder
                    check_results_map[stl_path_key] = {
                        "generated_stl_path": stl_path_key,
                        "error": f"Executor exception: {exc}",
                        "checks": {},
                        "check_errors": [f"Executor exception: {exc}"]
                    }
        logger.info(f"Finished processing {len(check_results_map)} check results.")

    check_duration = time.time() - start_check_time
    logger.info(f"--- Phase 3: Parallel Geometry Checks Complete ({check_duration:.2f}s) ---")

    # --- Phase 3 Gate: Log summary of check results ---
    # Note: Defining 'success' for a check is complex (depends on which checks passed).
    # We'll just log the number attempted vs completed without error for now.
    check_completed_count = len(check_results_map)
    check_error_count = sum(1 for res in check_results_map.values() if res.get("error") or res.get("check_errors"))
    # logger.info(f"Checking Summary: Attempted={len(check_jobs)}, Completed={check_completed_count}, Errors Encountered={check_error_count}") # Removed for Phase 4


    # =========================================================================
    # --- Phase 4: Final Assembly --- 
    # =========================================================================
    logger.info("--- Phase 4: Starting Final Result Assembly ---")
    start_assembly_time = time.time()

    # Prepare arguments for assemble_final_results
    assembly_args = {
        "generation_results": generation_results, # LLM results
        "direct_stl_generation_results": direct_stl_generation_results, # Zoo results
        "render_results": render_results, 
        "check_results_map": check_results_map, # Map: generated_stl_path -> check_result_dict
        "stl_path_to_gen_info": stl_path_to_gen_info, # Map: generated_stl_path -> original_gen_info_dict
        # Pass logger instance (get it again just in case)
        "logger": get_logger(__name__),
        "project_root": project_root # Pass project root for path resolution
    }

    try:
        final_results_list = assemble_final_results(**assembly_args)
        logger.info(f"Successfully assembled {len(final_results_list)} final result entries.")

        # --- Save final results --- 
        final_output_path = os.path.join(run_base_dir, f"results_{args.run_id}.json")
        try:
            with open(final_output_path, "w") as f:
                json.dump(final_results_list, f, indent=4)
            logger.info(f"Successfully saved final results to: {final_output_path}")
        except IOError as e:
            logger.error(f"Failed to write final results JSON to {final_output_path}: {e}")

    except Exception as e:
        logger.error(f"Error during final result assembly: {e}", exc_info=True)

    assembly_duration = time.time() - start_assembly_time
    logger.info(f"--- Phase 4: Final Result Assembly Complete ({assembly_duration:.2f}s) ---")

    logger.info(f"--- CadEval Parallel Run Finished --- Run ID: {args.run_id}")


if __name__ == "__main__":
    # Note: ProcessPoolExecutor requires the main guard
    try:
        main()
    except Exception as e:
        # Fallback logger if setup failed
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)
