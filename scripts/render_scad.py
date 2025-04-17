#!/usr/bin/env python3
"""
OpenSCAD Rendering Script for CadEval

This module handles the conversion of generated .scad files to .stl format
using OpenSCAD in headless mode. It also performs validation checks.
"""

import os
import sys
import subprocess
import time
import json
import glob
import logging
from typing import Dict, Any, List, Optional, Tuple
from packaging import version
import re

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.logger_setup import get_logger, setup_logger
from scripts.config_loader import get_config, Config, ConfigError

# Initialize logger for this module
logger = get_logger(__name__)

class RenderError(Exception):
    """Exception raised for errors during OpenSCAD rendering."""
    pass


def validate_openscad_config(
    openscad_config: Dict[str, Any]
) -> str:
    """
    Validates OpenSCAD configuration (executable path and version).

    Args:
        openscad_config: Dictionary containing OpenSCAD configuration settings.

    Returns:
        The detected OpenSCAD version string if validation passes.

    Raises:
        ValidationError: If executable is not found or version is too low.
        ConfigError: If required config keys are missing in the dictionary.
    """
    logger.info("Validating OpenSCAD configuration...")

    # --- Get required path from dictionary ---
    executable_path = openscad_config.get('executable_path')
    if not executable_path:
         raise ConfigError("Missing required configuration key: 'openscad.executable_path'")
    # ---

    minimum_version_str = openscad_config.get('minimum_version', '2021.01') # Default if missing

    if not os.path.exists(executable_path):
        raise RenderError(f"OpenSCAD executable not found at specified path: {executable_path}")
    logger.info(f"OpenSCAD executable found at: {executable_path}")

    try:
        # Use --version to get the version string
        process = subprocess.run(
            [executable_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )
        # Output is usually on stderr for --version
        version_output = process.stderr.strip() or process.stdout.strip()
        # Example output: "OpenSCAD version 2023.08.18"
        match = re.search(r"(\d{4}\.\d{2}(\.\d{2})?)", version_output) # Simpler regex for YYYY.MM or YYYY.MM.DD
        if not match:
            # Try matching development snapshot format like "2024.01.31.ai12345"
             match_dev = re.search(r"(\d{4}\.\d{2}\.\d{2}\.\w+)", version_output)
             if match_dev:
                 # Treat dev builds as potentially meeting the requirement, extract base YYYY.MM
                 dev_version_str = match_dev.group(1)
                 version_str = ".".join(dev_version_str.split('.')[:2]) # Extract YYYY.MM part
                 logger.warning(f"Detected OpenSCAD development version: {dev_version_str}. Comparing base version {version_str}.")
             else:
                raise RenderError(f"Could not parse OpenSCAD version from output: {version_output}")
        else:
            version_str = match.group(1)

        logger.info(f"Detected OpenSCAD version: {version_str}")

        # Compare versions (simple string comparison works for YYYY.MM format)
        if version_str < minimum_version_str:
            raise RenderError(f"Detected OpenSCAD version {version_str} is below minimum required {minimum_version_str}.")

        logger.info(f"OpenSCAD version meets minimum requirement ({minimum_version_str}).")
        return version_str # Return detected version

    except FileNotFoundError: # Should be caught by os.path.exists, but just in case
         raise RenderError(f"OpenSCAD executable check failed (not found?): {executable_path}")
    except subprocess.TimeoutExpired:
        raise RenderError("Checking OpenSCAD version timed out.")
    except subprocess.CalledProcessError as e:
        raise RenderError(f"Checking OpenSCAD version failed (Exit Code {e.returncode}). Stderr: {e.stderr}")
    except Exception as e:
        raise RenderError(f"Unexpected error during OpenSCAD version check: {e}")


def render_scad_file(
    scad_path: str,
    output_dir: str,
    openscad_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Renders a single .scad file to .stl using OpenSCAD.

    Args:
        scad_path: Absolute path to the input .scad file.
        output_dir: Directory where the .stl file should be saved.
        openscad_config: Dictionary containing OpenSCAD configuration settings
                         (e.g., executable_path, render_timeout_seconds, etc.).

    Returns:
        A dictionary containing the render status and output paths.
        Keys: 'scad_path', 'stl_path', 'status', 'error', 'duration', 'summary_path'
    """
    logger = logging.getLogger(__name__) # Use standard logger
    scad_filename = os.path.basename(scad_path)
    stl_filename = scad_filename.replace(".scad", ".stl")
    summary_filename = scad_filename.replace(".scad", "_summary.json")

    stl_path = os.path.join(output_dir, stl_filename)
    summary_path = os.path.join(output_dir, summary_filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Default result structure
    result = {
        "scad_path": scad_path,
        "stl_path": None,
        "status": "Error",
        "error": None,
        "duration": None,
        "summary_path": None
    }

    start_time = time.time()

    try:
        logger.info(f"Starting render for: {scad_filename}")

        # Build the command using the config dictionary
        command = _build_openscad_command(scad_path, stl_path, summary_path, openscad_config)
        logger.debug(f"Executing OpenSCAD command: {' '.join(command)}")

        # Get timeout from the dictionary, provide default
        timeout_seconds = openscad_config.get('render_timeout_seconds', 120)

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False # Don't raise exception on non-zero exit code
        )

        duration = time.time() - start_time
        result["duration"] = duration

        if process.returncode == 0:
            if os.path.exists(stl_path):
                logger.info(f"Render successful for {scad_filename} in {duration:.2f}s")
                result["status"] = "Success"
                result["stl_path"] = stl_path
                if os.path.exists(summary_path):
                     result["summary_path"] = summary_path
                else:
                     logger.warning(f"Render summary file not found: {summary_path}")
            else:
                result["error"] = f"OpenSCAD completed but output STL not found. Stderr: {process.stderr.strip()}"
                logger.error(f"Render failed for {scad_filename}: Output STL missing. Stderr: {process.stderr.strip()}")
        else:
             result["error"] = f"OpenSCAD failed with exit code {process.returncode}. Stderr: {process.stderr.strip()}"
             logger.error(f"Render failed for {scad_filename} (Exit Code {process.returncode}). Stderr: {process.stderr.strip()}")

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        result["duration"] = duration
        result["error"] = f"OpenSCAD render timed out after {timeout_seconds} seconds."
        logger.error(f"Render timed out for {scad_filename} after {timeout_seconds}s.")
    except FileNotFoundError:
        # This usually means the executable path is wrong
        exec_path = openscad_config.get('executable_path', 'OpenSCAD (Not Found)')
        result["error"] = f"OpenSCAD executable not found at path: {exec_path}"
        logger.error(result["error"])
        # Don't log exc_info=True here as FileNotFoundError is less helpful
    except Exception as e:
        duration = time.time() - start_time # Capture time even on unexpected errors
        result["duration"] = duration
        result["error"] = f"Unexpected rendering error: {e}"
        logger.error(f"Unexpected rendering error for {scad_filename}: {e}", exc_info=True) # Log full traceback

    return result


def _build_openscad_command(
    scad_path: str,
    stl_path: str,
    summary_path: str,
    openscad_config: Dict[str, Any]
) -> List[str]:
    """Builds the OpenSCAD command list."""
    # --- Get values from dictionary ---
    executable_path = openscad_config.get('executable_path')
    if not executable_path:
         # This should ideally be caught by validate_openscad_config earlier
         raise ConfigError("Missing required configuration key: 'openscad.executable_path' in provided dictionary")

    export_format = openscad_config.get('export_format', 'asciistl').lower()
    backend = openscad_config.get('backend') # Optional
    summary_options = openscad_config.get('summary_options') # Optional
    # ---

    if export_format not in ['asciistl', 'stl', 'binarystl']:
         logger.warning(f"Unsupported openscad.export_format '{export_format}'. Defaulting to 'asciistl'.")
         export_format = 'asciistl'
    # Use 'stl' for binary STL in OpenSCAD command line
    if export_format == 'binarystl':
        export_format = 'stl'

    command = [
        executable_path,
        "-o", stl_path, # Output STL path
        scad_path, # Input SCAD path
        "--export-format", export_format,
    ]

    # Add optional backend if specified
    if backend:
        command.extend(["--backend", backend])

    # Add optional summary export if specified
    if summary_options:
        command.extend(["--summary-file", summary_path])
        command.extend(["--summary", summary_options])

    return command

def render_all_scad(input_dir: str, output_dir: str, config: Config) -> List[Dict[str, Any]]:
    """
    Renders all .scad files found in the input directory by calling render_scad_file.
    
    Args:
        input_dir: Directory containing .scad files to render.
        output_dir: Directory where output .stl/.json files should be saved.
        config: The loaded Config object.
    
    Returns:
        A list of dictionaries, each representing the result of a render attempt.
    """
    # Resolve input directory path relative to config if needed
    if input_dir in config.get("directories", {}):
        input_dir = config.resolve_path(f"directories.{input_dir}")
    else:
        # Assume it's a path relative to the execution directory or absolute
        input_dir = os.path.abspath(input_dir)

    logger.info(f"Scanning for .scad files in: {input_dir}")
    
    if not os.path.isdir(input_dir):
        error_msg = f"Input directory not found: {input_dir}"
        logger.error(error_msg)
        raise RenderError(error_msg)
        
    # Find all .scad files in the input directory
    scad_files = glob.glob(os.path.join(input_dir, "*.scad"))
    
    if not scad_files:
        logger.warning(f"No .scad files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(scad_files)} .scad files to render.")
    
    # Resolve output directory path
    if output_dir in config.get("directories", {}):
        output_dir = config.resolve_path(f"directories.{output_dir}")
    else:
        output_dir = os.path.abspath(output_dir)
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")
    
    results = []
    for scad_path in scad_files:
        try:
            render_result = render_scad_file(scad_path, output_dir, config)
            results.append(render_result)
        except Exception as e:
            # Catch unexpected errors during the call itself
            logger.error(f"Unexpected error processing {scad_path}: {e}")
            results.append({
                "scad_path": scad_path,
                "status": "Failed",
                "stl_path": None,
                "summary_path": None,
                "duration": 0,
                "error": f"Unexpected error during batch processing: {e}"
            })
            
    logger.info(f"Finished rendering. Processed {len(results)} files.")
    return results


if __name__ == "__main__":
    # Example usage when run directly
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Render .scad files using OpenSCAD")
    parser.add_argument("--input-dir", default="generated_outputs", help="Directory containing .scad files to render")
    parser.add_argument("--output-dir", default="generated_outputs", help="Directory where output .stl/.json files should be saved")
    parser.add_argument("--validate-only", action="store_true", help="Only validate OpenSCAD configuration, do not render")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(__name__, level=log_level, log_file="logs/render_scad.log")
    
    try:
        # Load configuration
        config = get_config()
        
        # Validate OpenSCAD configuration
        detected_version = validate_openscad_config(config)
        print(f"OpenSCAD validation successful. Detected version: {detected_version}")
        
        if args.validate_only:
            print("Validation complete. Exiting as requested.")
            sys.exit(0)
        
        # --- Call the batch rendering function --- 
        print(f"\nStarting rendering process for files in '{args.input_dir}'...")
        results = render_all_scad(args.input_dir, args.output_dir, config)
        
        print(f"\nRendering complete. Processed {len(results)} files.")
        success_count = 0
        for i, result in enumerate(results):
             status = result.get('status', 'Unknown')
             scad_file = os.path.basename(result.get('scad_path', 'N/A'))
             duration = f"{result.get('duration', 0):.2f}s"
             print(f"  [{i+1}/{len(results)}] {scad_file}: {status} ({duration})")
             if status == "Success":
                 success_count += 1
                 if result.get('stl_path'):
                     print(f"      -> STL: {result['stl_path']}")
                 if result.get('summary_path'):
                     print(f"      -> Summary: {result['summary_path']}")
             elif result.get('error'):
                 print(f"      Error: {result['error']}")
        print(f"\nSummary: {success_count} successful, {len(results) - success_count} failed/skipped.")
            
    except (ConfigError, RenderError) as e:
        print(f"Error: {e}")
        logger.error(f"Stopping due to error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.exception("An unexpected error occurred") # Log traceback
        sys.exit(1) 