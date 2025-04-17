import os
import logging
import time
import subprocess
import tempfile
import shutil
import glob
from typing import List, Dict, Any, Tuple, Optional

def generate_stl_with_zoo(prompt: str, output_stl_path: str, model_config: Dict[str, Any], logger: logging.Logger) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    """Generates an STL file using the Zoo CLI.

    Args:
        prompt: The text prompt to send to the Zoo CLI.
        output_stl_path: The desired final path for the generated STL file.
        model_config: The configuration dictionary for the zoo_cli model.
        logger: The logger instance to use.

    Returns:
        A tuple containing:
        - The final path to the generated STL if successful, otherwise None.
        - An error message string if generation failed, otherwise None.
        - The command string that was executed.
        - The duration of the subprocess call in seconds, or None if it failed early.
    """
    cli_args = model_config.get("cli_args", {})
    output_format = cli_args.get("output_format", "stl") # Default to stl
    if output_format.lower() != "stl":
        logger.warning(f"Zoo CLI output format configured as '{output_format}', but forcing STL for evaluation.")
        output_format = "stl"

    # Create a temporary directory for Zoo output within the run's main dir
    output_dir = os.path.dirname(output_stl_path) # e.g., results/<run_id>/stl
    run_base_dir = os.path.dirname(output_dir) # e.g., results/<run_id>
    # Ensure the base directory exists before creating tempdir inside it
    os.makedirs(run_base_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(prefix="zoo_tmp_", dir=run_base_dir)
    logger.debug(f"Created temporary directory for Zoo output: {temp_output_dir}")

    # Construct command parts carefully
    cmd_list = [
        "zoo", "ml", "text-to-cad", "export",
        f"--output-format={output_format}",
        f"--output-dir={temp_output_dir}",
        prompt
    ]
    # Handle potential spaces in prompt for display/logging only
    cmd_str_display = f'zoo ml text-to-cad export --output-format={output_format} --output-dir="{temp_output_dir}" "{prompt}"' # Fixed quote
    logger.info(f"Executing Zoo CLI command:")
    logger.info(f"> {cmd_str_display}") # Log the more readable version

    error_message = None
    generated_stl_src_path = None
    final_stl_path = None
    duration = None # Initialize duration

    try:
        start_time = time.time() # Ensure time is imported
        process = subprocess.run(
            cmd_list, # Pass the list for safety with args
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        duration = time.time() - start_time
        logger.info(f"Zoo CLI finished in {duration:.2f} seconds with exit code {process.returncode}")

        if process.stdout:
            logger.debug(f"Zoo CLI stdout:\n{process.stdout}")
        if process.stderr:
            log_level = logging.WARNING if process.returncode == 0 else logging.ERROR
            logger.log(log_level, f"Zoo CLI stderr:\n{process.stderr}")

        if process.returncode != 0:
            error_message = f"Zoo CLI failed with exit code {process.returncode}. Stderr: {process.stderr.strip()}"
        else:
            stl_files = glob.glob(os.path.join(temp_output_dir, "*.stl"))
            if not stl_files:
                error_message = "Zoo CLI succeeded but no STL file found in temporary output directory."
            elif len(stl_files) > 1:
                logger.warning(f"Multiple STL files found in {temp_output_dir}. Using the first one: {os.path.basename(stl_files[0])}")
                generated_stl_src_path = stl_files[0]
            else:
                generated_stl_src_path = stl_files[0]
                logger.info(f"Found generated STL: {os.path.basename(generated_stl_src_path)}")

            if generated_stl_src_path:
                try:
                    os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
                    shutil.move(generated_stl_src_path, output_stl_path)
                    logger.info(f"Moved generated STL to final path: {output_stl_path}")
                    final_stl_path = output_stl_path
                except Exception as move_err:
                    error_message = f"Failed to move generated STL from {generated_stl_src_path} to {output_stl_path}: {move_err}"
                    final_stl_path = None

    except subprocess.TimeoutExpired:
        error_message = f"Zoo CLI command timed out after 300 seconds."
        logger.error(error_message)
    except Exception as e:
        error_message = f"Error executing Zoo CLI: {e}"
        logger.error(error_message, exc_info=True)
    finally:
        # Defensively check if temp_output_dir was created before trying to remove
        if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
            try:
                shutil.rmtree(temp_output_dir)
                logger.debug(f"Removed temporary directory: {temp_output_dir}")
            except Exception as clean_err:
                logger.warning(f"Failed to remove temporary directory {temp_output_dir}: {clean_err}")

    return final_stl_path, error_message, cmd_str_display, duration
