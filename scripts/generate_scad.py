#!/usr/bin/env python3
"""
SCAD Generation Script for CadEval

This module implements the logic to generate OpenSCAD code from task descriptions
using various Large Language Models (LLMs).
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.logger_setup import get_logger, setup_logger
from scripts.config_loader import get_config, ConfigError
from scripts.task_loader import load_tasks, TaskLoadError
from scripts.llm_clients import create_llm_client, LLMError

# Initialize logger for this module
logger = get_logger(__name__)
config = get_config()

class ScadGenerationError(Exception):
    """Exception raised for errors during OpenSCAD code generation."""
    pass


def format_prompt(description: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Format a task description into a suitable prompt for LLM.
    
    Args:
        description: The task description text
        parameters: Optional parameters to customize the prompt
    
    Returns:
        The formatted prompt string
    """
    # Default parameters (could be moved to config)
    params = {
        "include_header": True,
        "reminder_single_part": True,
        "specific_reminders": True,
    }
    
    # Override defaults with any provided parameters
    if parameters:
        params.update(parameters)
    
    # Create prompt components
    components = []
    
    # Header - main request
    if params["include_header"]:
        components.append(
            "Create a valid OpenSCAD script that models the following 3D object, described below. "
            "The code should define a complete 3D model that closely matches the description."
        )
    
    # The actual description
    components.append(description)
    
    # Add specific reminders if requested
    if params["reminder_single_part"]:
        components.append(
            "IMPORTANT: Create only a SINGLE part/model in this script. "
            "Do not include multiple unconnected components."
        )
    
    if params["specific_reminders"]:
        components.append(
            "Reminders:\n"
            "- Ensure the model is manifold and watertight\n"
            "- Use appropriate operations as needed\n"
            "- Define the model at the origin (0,0,0)\n"
            "- Include helpful comments in your code\n"
            "- Do not include example usage or test code\n"
            "- Do not include explanations outside of code comments"
        )
    
    # Join the components with double newlines for visual separation
    return "\n\n".join(components)


def generate_scad_for_task(
    task: Dict[str, Any], 
    model_config: Dict[str, Any],
    output_dir: str,
    prompt_params: Optional[Dict[str, Any]] = None,
    replicate_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate OpenSCAD code for a specific task using a specific LLM.
    
    Args:
        task: The task data dictionary
        model_config: The model configuration dictionary
        output_dir: Directory where SCAD files should be saved
        prompt_params: Optional parameters to customize the prompt
        replicate_id: Optional identifier for the replicate run
    
    Returns:
        A dictionary with information about the generation process and results
    """
    # Get necessary information from task
    task_id = task.get("task_id")
    description = task.get("description")
    
    # Get model information
    provider = model_config.get("provider", "").lower()
    model_name = model_config.get("name", "")
    # Sanitize provider and model name for filename usage
    provider_safe = provider.replace("/", "_")
    model_name_safe = model_name.replace("/", "_")
    model_identifier_for_filename = f"{provider_safe}_{model_name_safe}"
    
    # Prepare result dictionary
    result = {
        "task_id": task_id,
        "model": f"{provider}/{model_name}", # Use original names for result ID
        "model_config_used": model_config,
        "replicate_id": replicate_id, # Add replicate_id to result
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "output_path": None,
        "error": None,
        "duration_seconds": None
    }
    
    # Create output path including replicate_id if provided
    base_filename = f"{task_id}_{model_identifier_for_filename}"
    if replicate_id is not None:
        base_filename += f"_rep{replicate_id}"
    output_path = os.path.join(output_dir, f"{base_filename}.scad")
    result["output_path"] = output_path
    
    # Format the prompt
    try:
        prompt = format_prompt(description, prompt_params)
        logger.info(f"Formatted prompt for task '{task_id}' using {provider}/{model_name}")
        logger.debug(f"Prompt: {prompt}")
        result["prompt_used"] = prompt
    except Exception as e:
        error_msg = f"Error formatting prompt for task '{task_id}': {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
    
    # Create LLM client
    try:
        client = create_llm_client(model_config)
    except LLMError as e:
        error_msg = f"Failed to create LLM client for {provider}/{model_name}: {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
    
    # Generate OpenSCAD code
    start_time = time.time()
    try:
        generated_code = client.generate_text(prompt)
        
        # Check if the response is empty
        if not generated_code:
            error_msg = f"Received empty response from {provider}/{model_name}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result
        
        # Extract SCAD code from response if needed
        # Some LLMs might wrap the code in markdown code blocks or add explanations
        scad_code = extract_scad_code(generated_code)
        
        # Save the SCAD code to file
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(scad_code)
            
            logger.info(f"Successfully saved SCAD code to {output_path}")
            result["success"] = True
        except Exception as e:
            error_msg = f"Error saving SCAD code to file {output_path}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result
        
    except LLMError as e:
        error_msg = f"Error generating SCAD code from {provider}/{model_name}: {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
    except Exception as e:
        error_msg = f"Unexpected error during SCAD generation: {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
    finally:
        # Record duration regardless of success/failure
        result["duration_seconds"] = time.time() - start_time
    
    return result


def extract_scad_code(response: str) -> str:
    """
    Extract OpenSCAD code from an LLM response.
    
    This handles cases where LLMs wrap code in markdown code blocks
    or add explanations before/after the code.
    
    Args:
        response: The full text response from an LLM
    
    Returns:
        The extracted OpenSCAD code
    """
    # First, check if the response contains markdown code blocks
    if "```" in response:
        # Look for OpenSCAD code blocks
        code_blocks = []
        lines = response.split('\n')
        in_code_block = False
        scad_block = False
        block_content = []
        
        for line in lines:
            if line.strip().startswith("```"):
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    # Check if this is explicitly marked as OpenSCAD
                    scad_block = "scad" in line.strip().lower() or "openscad" in line.strip().lower()
                    block_content = []
                else:
                    # End of code block
                    if scad_block or not code_blocks:
                        # Either it's a SCAD block or we have no blocks yet
                        code_blocks.append("\n".join(block_content))
                    in_code_block = False
                    scad_block = False
            elif in_code_block:
                block_content.append(line)
        
        # If we found at least one block, return the first SCAD block or the first block
        if code_blocks:
            return code_blocks[0]
    
    # If no code blocks found or no blocks with OpenSCAD content,
    # return the whole response (which might be pure code without markdown)
    return response


def generate_all_scad(
    tasks_dir: str = "tasks",
    output_dir: str = "generated_outputs",
    model_filter: Optional[List[str]] = None,
    task_filter: Optional[List[str]] = None,
    prompt_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate OpenSCAD code for all tasks using all configured LLMs.
    
    Args:
        tasks_dir: Directory containing task YAML files
        output_dir: Directory where SCAD files should be saved
        model_filter: Optional list of model identifiers to restrict generation to
        task_filter: Optional list of task IDs to restrict generation to
        prompt_params: Optional parameters to customize the prompts
    
    Returns:
        A list of dictionaries, each containing information about a generation attempt
    """
    results = []
    
    # Load configuration
    try:
        # Get LLM models from config
        models = config.get("llm.models", [])
        if not models:
            error_msg = "No LLM models configured in config.yaml"
            logger.error(error_msg)
            raise ScadGenerationError(error_msg)
        
        # Filter models if requested
        if model_filter:
            filtered_models = []
            for model in models:
                provider = model.get("provider", "").lower()
                name = model.get("name", "")
                identifier = f"{provider}_{name}"
                if identifier in model_filter or provider in model_filter or name in model_filter:
                    filtered_models.append(model)
            models = filtered_models
            
            if not models:
                error_msg = f"No models match the provided filter: {model_filter}"
                logger.error(error_msg)
                raise ScadGenerationError(error_msg)
        
        # Get or resolve output directory
        if output_dir in config.get("directories", {}):
            output_dir = config.resolve_path(f"directories.{output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
    except (ConfigError, ScadGenerationError) as e:
        logger.error(f"Error during configuration: {str(e)}")
        raise ScadGenerationError(f"Configuration error: {str(e)}")
    
    # Load tasks
    try:
        tasks = load_tasks(tasks_dir)
        if not tasks:
            error_msg = f"No tasks found in {tasks_dir}"
            logger.error(error_msg)
            raise ScadGenerationError(error_msg)
        
        # Filter tasks if requested
        if task_filter:
            tasks = [task for task in tasks if task.get("task_id") in task_filter]
            if not tasks:
                error_msg = f"No tasks match the provided filter: {task_filter}"
                logger.error(error_msg)
                raise ScadGenerationError(error_msg)
        
    except TaskLoadError as e:
        logger.error(f"Error loading tasks: {str(e)}")
        raise ScadGenerationError(f"Task loading error: {str(e)}")
    
    # For each task and model, generate SCAD code
    for task in tasks:
        task_id = task.get("task_id", "UNKNOWN")
        logger.info(f"Processing task: {task_id}")
        
        for model_config in models:
            provider = model_config.get("provider", "UNKNOWN")
            model_name = model_config.get("name", "UNKNOWN")
            logger.info(f"Generating SCAD for task '{task_id}' using {provider}/{model_name}")
            
            # Generate SCAD code for this task and model
            try:
                result = generate_scad_for_task(task, model_config, output_dir, prompt_params)
                results.append(result)
                
                if result["success"]:
                    logger.info(f"Successfully generated SCAD for task '{task_id}' using {provider}/{model_name}")
                else:
                    logger.warning(f"Failed to generate SCAD for task '{task_id}' using {provider}/{model_name}: {result['error']}")
                
            except Exception as e:
                logger.error(f"Unexpected error generating SCAD for task '{task_id}' using {provider}/{model_name}: {str(e)}")
                results.append({
                    "task_id": task_id,
                    "model": f"{provider}/{model_name}",
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "output_path": None,
                    "error": f"Unexpected error: {str(e)}",
                    "duration_seconds": None
                })
    
    # Log summary
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"SCAD generation completed. Successful: {success_count}/{len(results)}")
    
    return results


if __name__ == "__main__":
    # Example usage when run directly
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate OpenSCAD code from task descriptions using LLMs")
    parser.add_argument("--tasks-dir", default="tasks", help="Directory containing task YAML files")
    parser.add_argument("--output-dir", default="generated_outputs", help="Directory where SCAD files should be saved")
    parser.add_argument("--models", nargs="+", help="Specific models to use (provider_name or provider or name)")
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(__name__, level=log_level, log_file="logs/generate_scad.log")
    
    try:
        # Run SCAD generation
        results = generate_all_scad(
            tasks_dir=args.tasks_dir,
            output_dir=args.output_dir,
            model_filter=args.models,
            task_filter=args.tasks
        )
        
        # Print summary
        success_count = sum(1 for r in results if r["success"])
        print(f"\nSCAD generation completed: {success_count}/{len(results)} successful")
        
        # Print details for each result
        for i, result in enumerate(results):
            task_id = result["task_id"]
            model = result["model"]
            status = "SUCCESS" if result["success"] else "FAILED"
            duration = f"{result['duration_seconds']:.2f}s" if result["duration_seconds"] else "N/A"
            
            print(f"\n[{i+1}] Task: {task_id} | Model: {model} | Status: {status} | Duration: {duration}")
            if result["output_path"] and result["success"]:
                print(f"    Output: {result['output_path']}")
            if not result["success"]:
                print(f"    Error: {result['error']}")
        
    except ScadGenerationError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1) 