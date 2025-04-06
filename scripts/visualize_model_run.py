#!/usr/bin/env python3
"""
Script to visualize all tasks for a single model from a specific evaluation run
by launching a separate comparison window for each task.
"""

import argparse
import os
import sys
import logging
import subprocess # To launch separate processes

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# No longer need to import alignment/visualization functions directly
# Import only logger setup
try:
    from scripts.logger_setup import setup_logger
    from scripts.config_loader import get_config, Config # Add Config
except ImportError as e:
    print(f"Error importing script components: {e}")
    sys.exit(1)

# Removed generate_distinct_colors

def main(run_id, model_id, num_tasks=10):
    """Loops through tasks and launches a separate comparison visualization process for each."""
    logger = setup_logger(__name__, level=logging.INFO, console=True, log_file=None)
    logger.info(f"Launching separate visualizations for run '{run_id}', model '{model_id}'")

    # --- Load Config to find provider --- Start ---
    try:
        config = get_config()
        provider = None
        models_in_config = config.get('llm.models', [])
        if not isinstance(models_in_config, list):
             logger.warning("'llm.models' in config is not a list. Cannot find provider.")
             models_in_config = [] # Prevent error below

        for model_conf in models_in_config:
            if isinstance(model_conf, dict) and model_conf.get('name') == model_id:
                provider = model_conf.get('provider')
                break
        if not provider:
            logger.error(f"Could not find model '{model_id}' or its provider in config.yaml. Cannot determine filenames.")
            return
        logger.debug(f"Found provider '{provider}' for model '{model_id}'")
    except Exception as e:
        logger.error(f"Error loading or parsing config to find provider: {e}", exc_info=True)
        return
    # --- Load Config to find provider --- End ---

    # --- Use Absolute Paths --- Start ---
    base_results_dir = os.path.join(project_root, "results", run_id, "stl")
    reference_base_dir = os.path.join(project_root, "reference")
    # --- Use Absolute Paths --- End ---

    comparison_script_path = os.path.join(project_root, "scripts", "visualize_comparison.py")

    if not os.path.isdir(base_results_dir):
        logger.error(f"Results directory not found using absolute path: {base_results_dir}")
        return
    if not os.path.isfile(comparison_script_path):
         logger.error(f"Comparison script not found: {comparison_script_path}")
         return

    launched_count = 0
    processes = [] # Keep track of launched processes (optional)

    for task_num in range(1, num_tasks + 1):
        task_id = f"task{task_num}"
        logger.info(f"--- Processing {task_id} ---")

        ref_filename = f"{task_id}.stl"
        # --- Convert hyphens in model_id to underscores for filename --- Start ---
        model_id_for_filename = model_id.replace('-', '_')
        # --- Convert hyphens in model_id to underscores for filename --- End ---
        gen_filename = f"{task_id}_{provider}_{model_id_for_filename}.stl" # Use the modified model_id
        # These will now use the absolute base paths defined above
        ref_path = os.path.join(reference_base_dir, ref_filename)
        gen_path = os.path.join(base_results_dir, gen_filename)

        logger.debug(f"Checking absolute reference path: {ref_path}") # Add debug logging
        if not os.path.exists(ref_path):
            logger.warning(f"Reference file not found (abs check), skipping: {ref_path}")
            continue

        logger.debug(f"Checking absolute generated path: {gen_path}") # Restore debug logging
        # --- Restore the explicit check for generated file --- Start ---
        if not os.path.exists(gen_path):
            logger.warning(f"Generated file not found (abs check): {gen_path}") # Simplified message
            continue
        # --- Restore the explicit check for generated file --- End ---

        # Construct the command to run visualize_comparison.py
        window_title = f"{task_id} vs {model_id} ({run_id})"
        command = [
            sys.executable, # Use the same python interpreter
            comparison_script_path,
            "--ref", ref_path,
            "--gen", gen_path,
            "--title", window_title
            # Add --points here if you want non-default sampling
        ]

        logger.info(f"Launching visualization process for {task_id}: {' '.join(command)}") # Restore original log message
        try:
            # Launch the process without waiting for it to complete
            process = subprocess.Popen(command)
            processes.append(process) # Optionally store process handle
            launched_count += 1
        except Exception as e:
            logger.error(f"Failed to launch visualization for {task_id}: {e}", exc_info=True)

    # --- Script End ---
    if launched_count == 0:
         logger.warning(f"No valid comparison pairs found to launch for model '{model_id}' in run '{run_id}'.")
    else:
         logger.info(f"Launched {launched_count} visualization processes.")
         logger.info("Note: Windows might take a moment to appear and are independent.")

    # Optional: Wait for processes if needed, but usually not desired for this use case
    # for p in processes:
    #     p.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch separate visualization windows for all task comparisons for a specific model and run.")
    parser.add_argument("--run_id", required=True, help="The ID of the evaluation run (e.g., 20250404_155435).")
    parser.add_argument("--model_id", required=True, help="The full identifier of the model (e.g., anthropic_claude_3_7_sonnet_20250219).")
    parser.add_argument("--num_tasks", type=int, default=10, help="The number of tasks to loop through (default: 10).")

    args = parser.parse_args()

    main(args.run_id, args.model_id, args.num_tasks) 