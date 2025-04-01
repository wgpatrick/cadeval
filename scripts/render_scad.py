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
from typing import Dict, Any, List, Optional, Tuple
from packaging import version

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


def validate_openscad_config(config: Config) -> str:
    """
    Validates the OpenSCAD configuration: executable path and version.
    
    Args:
        config: The loaded Config object.
        
    Returns:
        The detected OpenSCAD version string if validation passes.
        
    Raises:
        RenderError: If validation fails (path not found, version too old).
    """
    logger.info("Validating OpenSCAD configuration...")
    
    # 1. Validate executable path
    try:
        executable_path = config.get_required('openscad.executable_path')
    except ConfigError:
         raise RenderError("OpenSCAD executable path not defined in configuration ('openscad.executable_path')")

    # Resolve path relative to config if necessary (handled by get_required if path not absolute)
    # Use raw path from config first
    raw_path = config.get('openscad.executable_path')
    if not os.path.isabs(raw_path):
        config_dir = os.path.dirname(os.path.abspath(config.config_path))
        resolved_path = os.path.normpath(os.path.join(config_dir, raw_path))
    else:
        resolved_path = os.path.normpath(raw_path)

    logger.debug(f"Checking for OpenSCAD executable at: {resolved_path}")
    if not os.path.exists(resolved_path):
        raise RenderError(f"OpenSCAD executable not found at path: {resolved_path}")
    if not os.path.isfile(resolved_path):
         raise RenderError(f"Specified OpenSCAD path is not a file: {resolved_path}")
    if not os.access(resolved_path, os.X_OK):
        raise RenderError(f"Specified OpenSCAD path is not executable: {resolved_path}")
    
    logger.info(f"OpenSCAD executable found at: {resolved_path}")
    
    # 2. Validate OpenSCAD version
    try:
        result = subprocess.run([resolved_path, "--version"], capture_output=True, text=True, check=True, timeout=10)
        version_output = result.stderr.strip() # OpenSCAD often prints version to stderr
        if not version_output and result.stdout:
            version_output = result.stdout.strip() # Fallback to stdout
        
        logger.debug(f"OpenSCAD --version output: {version_output}")
        
        # Extract version number (e.g., "OpenSCAD version 2021.01")
        detected_version_str = None
        if "version" in version_output.lower():
             parts = version_output.split()
             for i, part in enumerate(parts):
                 if part.lower() == "version" and i + 1 < len(parts):
                     detected_version_str = parts[i+1]
                     break
        
        if not detected_version_str:
            raise RenderError(f"Could not parse OpenSCAD version from output: {version_output}")
        
        detected_version = version.parse(detected_version_str)
        logger.info(f"Detected OpenSCAD version: {detected_version}")
        
        # Compare with minimum required version
        try:
            minimum_version_str = config.get_required('openscad.minimum_version')
            minimum_version = version.parse(minimum_version_str)
        except ConfigError:
             raise RenderError("Minimum OpenSCAD version not defined in configuration ('openscad.minimum_version')")
        
        if detected_version < minimum_version:
            error_msg = f"Detected OpenSCAD version ({detected_version}) is older than minimum required ({minimum_version}). Please update OpenSCAD."
            logger.error(error_msg)
            raise RenderError(error_msg)
        else:
             logger.info(f"OpenSCAD version meets minimum requirement ({minimum_version}).")
             
        return str(detected_version)
             
    except FileNotFoundError:
        # Should be caught by path check, but safeguard
        raise RenderError(f"Failed to run OpenSCAD command. Path correct? {resolved_path}")
    except subprocess.TimeoutExpired:
        raise RenderError("Checking OpenSCAD version timed out.")
    except subprocess.CalledProcessError as e:
         raise RenderError(f"Error running OpenSCAD --version: {e.stderr}")
    except Exception as e:
        raise RenderError(f"Unexpected error during OpenSCAD version check: {e}")

# --- Placeholder for main rendering logic --- 
# (To be implemented next)

def render_scad_file(scad_path: str, output_dir: str, config: Config) -> Dict[str, Any]:
    """
    Renders a single .scad file to .stl and potentially .json summary.
    
    Args:
        scad_path: Absolute or relative path to the input .scad file.
        output_dir: Directory where output .stl and .json files should be saved.
        config: The loaded Config object.
        
    Returns:
        A dictionary containing the rendering status, output paths, duration, and error info.
    """
    start_time = time.time()
    scad_filename = os.path.basename(scad_path)
    base_name = os.path.splitext(scad_filename)[0]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    stl_path = os.path.join(output_dir, f"{base_name}.stl")
    summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
    
    result = {
        "scad_path": scad_path,
        "status": "Unknown", # Will be updated to Success, Compile Error, Timeout, or Failed
        "stl_path": None,
        "summary_path": None,
        "duration": 0,
        "return_code": None,
        "stdout": None,
        "stderr": None,
        "error": None
    }
    
    logger.info(f"Starting render for: {scad_filename}")
    logger.debug(f"  Input SCAD: {scad_path}")
    logger.debug(f"  Output STL: {stl_path}")
    logger.debug(f"  Output Summary: {summary_path}")

    try:
        # Get OpenSCAD configuration and build command
        command = _build_openscad_command(scad_path, stl_path, summary_path, config)
        timeout_seconds = config.get('openscad.render_timeout_seconds', 120)
        
        logger.debug(f"Executing OpenSCAD command: {' '.join(command)}")
        
        # Execute the command
        process_result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        result["return_code"] = process_result.returncode
        result["stdout"] = process_result.stdout
        result["stderr"] = process_result.stderr
        result["duration"] = time.time() - start_time
        
        # Determine status based on return code
        if process_result.returncode == 0:
            # Check if STL file was actually created (important sanity check)
            if os.path.exists(stl_path) and os.path.getsize(stl_path) > 0:
                result["status"] = "Success"
                result["stl_path"] = stl_path
                logger.info(f"Render successful for {scad_filename} in {result['duration']:.2f}s")
                # Check if summary file was created (it might not be if version is old or options wrong)
                if os.path.exists(summary_path) and os.path.getsize(summary_path) > 0:
                     result["summary_path"] = summary_path
                     logger.debug(f"  Summary file created: {summary_path}")
                else:
                     logger.warning(f"  Summary file NOT created/empty for {scad_filename}")
            else:
                 result["status"] = "Failed"
                 result["error"] = "OpenSCAD reported success (exit code 0) but output STL file is missing or empty."
                 logger.error(result["error"]) 
        else:
            result["status"] = "Compile Error"
            result["error"] = f"OpenSCAD failed with exit code {process_result.returncode}. Stderr: {process_result.stderr.strip()}"
            logger.error(f"Render failed for {scad_filename}. Stderr: {process_result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        result["status"] = "Timeout"
        result["error"] = f"OpenSCAD rendering timed out after {timeout_seconds} seconds."
        result["duration"] = time.time() - start_time # Update duration to reflect timeout
        logger.error(result["error"])
    except FileNotFoundError:
         result["status"] = "Failed"
         result["error"] = f"OpenSCAD executable not found at the specified path: {command[0]}"
         logger.error(result["error"]) 
    except ConfigError as e:
         result["status"] = "Failed"
         result["error"] = f"Configuration error during rendering setup: {e}"
         logger.error(result["error"])
    except Exception as e:
        result["status"] = "Failed"
        result["error"] = f"An unexpected error occurred during rendering: {e}"
        logger.exception("Unexpected rendering error") # Log traceback
        result["duration"] = time.time() - start_time # Update duration

    return result

def _build_openscad_command(scad_path: str, stl_path: str, summary_path: str, config: Config) -> List[str]:
    """Helper function to construct the OpenSCAD command list."""
    # Get OpenSCAD configuration
    executable_path = config.get_required('openscad.executable_path')
    # Re-resolve path relative to config if needed (ensure consistency)
    raw_exec_path = config.get('openscad.executable_path')
    if not os.path.isabs(raw_exec_path):
        config_dir = os.path.dirname(os.path.abspath(config.config_path))
        executable_path = os.path.normpath(os.path.join(config_dir, raw_exec_path))
    else:
            executable_path = os.path.normpath(raw_exec_path)
    
    export_format = config.get('openscad.export_format', 'asciistl')
    backend = config.get('openscad.backend', None) # Default to None if not specified
    summary_options = config.get('openscad.summary_options', 'all') 
    
    # Construct the command parts
    command_parts = [
        executable_path,
        "-q", # Quiet mode
        "--export-format", export_format,
    ]
    
    # Add backend only if specified in config and not empty
    if backend:
        command_parts.extend(["--backend", backend])
        
    # Add summary options
    command_parts.extend(["--summary", summary_options])
    command_parts.extend(["--summary-file", summary_path])
        
    # Add output and input files
    command_parts.extend(["-o", stl_path])
    command_parts.append(scad_path)
    
    return command_parts

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