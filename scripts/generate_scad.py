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
from scripts.llm_clients import create_llm_client, LLMError, BaseLLMClient

# Initialize logger for this module
logger = get_logger(__name__)
config = get_config()

class ScadGenerationError(Exception):
    """Exception raised for errors during OpenSCAD code generation."""
    pass


def generate_scad_for_task(
    task: Dict[str, Any], 
    model_config: Dict[str, Any],
    output_dir: str,
    replicate_id: Optional[int] = None,
    prompt_key: str = 'default'
) -> Dict[str, Any]:
    """
    Generate OpenSCAD code for a specific task using a specific LLM.
    
    Args:
        task: The task data dictionary
        model_config: The model configuration dictionary
        output_dir: Directory where SCAD files should be saved
        replicate_id: Optional identifier for the replicate run
        prompt_key: Key to identify which prompt template to use from config (default: 'default')
    
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
        "prompt_key": prompt_key,  # Add prompt key to result
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "output_path": None,
        "error": None,
        "llm_duration_seconds": None,
        "prompt_tokens": None,
        "completion_tokens": None
    }
    
    # Create output path including replicate_id and prompt_key
    base_filename = f"{task_id}_{model_identifier_for_filename}_{prompt_key}"
    if replicate_id is not None:
        base_filename += f"_rep{replicate_id}"
    output_path = os.path.join(output_dir, f"{base_filename}.scad")
    result["output_path"] = output_path
    result["output_scad_path"] = output_path  # For compatibility with existing code
    
    # Get prompt template from config and format it with task description
    try:
        # Get the prompt template from config using prompt_key
        prompt_template = config.get_prompt(prompt_key)
        if not prompt_template:
            error_msg = f"Prompt key '{prompt_key}' not found in configuration"
            logger.error(error_msg)
            result["error"] = error_msg
            return result
            
        # Format the prompt template with the task description
        prompt = prompt_template.format(description=description)
        logger.info(f"Using prompt template '{prompt_key}' for task '{task_id}' using {provider}/{model_name}")
        logger.debug(f"Prompt: {prompt}")
        result["prompt_used"] = prompt
    except Exception as e:
        error_msg = f"Error formatting prompt for task '{task_id}': {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
    
    # Create LLM client
    client: Optional[BaseLLMClient] = None # Type hint
    try:
        client = create_llm_client(model_config)
    except (LLMError, ValueError) as e: # Catch ValueError from create_llm_client too
        error_msg = f"Failed to create LLM client for {provider}/{model_name}: {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
    
    # Generate OpenSCAD code
    try:
        start_time = time.time()
        generated_code = client.generate_text(prompt)
        end_time = time.time()
        llm_duration = end_time - start_time
        result["llm_duration_seconds"] = llm_duration
        
        # --- Extract Token Usage --- Start ---
        prompt_tokens = None
        completion_tokens = None
        raw_response = client.last_response # Client stores the raw response

        try:
            if client.provider == 'openai':
                # Always use /v1/responses API structure: input/output tokens
                try:
                    if hasattr(raw_response, 'usage') and raw_response.usage:
                        prompt_tokens = getattr(raw_response.usage, 'input_tokens', None)
                        completion_tokens = getattr(raw_response.usage, 'output_tokens', None)
                        if prompt_tokens is None or completion_tokens is None:
                             logger.debug(f"OpenAI /v1/responses usage field lacks input/output tokens for {client.model_name}.")
                    else:
                         logger.debug(f"OpenAI /v1/responses object lacks usage data (or is empty) for {client.model_name}.")
                except AttributeError:
                    logger.warning(f"Token usage attribute ('usage') is missing in the response structure for OpenAI model {client.model_name} via /v1/responses API.")
            elif client.provider == 'anthropic':
                 # Uses input/output tokens
                 try:
                     if hasattr(raw_response, 'usage') and raw_response.usage:
                        prompt_tokens = getattr(raw_response.usage, 'input_tokens', None)
                        completion_tokens = getattr(raw_response.usage, 'output_tokens', None)
                        if prompt_tokens is None or completion_tokens is None:
                             logger.warning(f"Could not find populated input/output tokens in usage object for Anthropic model {client.model_name}.")
                     else:
                         logger.warning(f"Could not find populated usage information in response object for Anthropic model {client.model_name}.")
                 except AttributeError:
                     logger.warning(f"Token usage attribute ('usage') is missing in the response structure for Anthropic model {client.model_name}.")
            elif client.provider == 'google':
                 # Uses usage_metadata with prompt_token_count / candidates_token_count
                 try:
                     if hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                        prompt_tokens = getattr(raw_response.usage_metadata, 'prompt_token_count', None)
                        completion_tokens = getattr(raw_response.usage_metadata, 'candidates_token_count', None)
                        if prompt_tokens is None or completion_tokens is None:
                             logger.warning(f"Could not find populated prompt/candidates token counts in usage_metadata for Google model {client.model_name}.")
                     else:
                         logger.warning(f"Could not find populated usage_metadata in response object for Google model {client.model_name}.")
                 except AttributeError:
                     logger.warning(f"Token usage attribute ('usage_metadata') is missing in the response structure for Google model {client.model_name}.")

            # --- Add DeepSeek Token Extraction --- Start ---
            elif client.provider == 'deepseek':
                 # Uses standard OpenAI 'usage' structure
                 try:
                     if hasattr(raw_response, 'usage') and raw_response.usage:
                         prompt_tokens = getattr(raw_response.usage, 'prompt_tokens', None)
                         completion_tokens = getattr(raw_response.usage, 'completion_tokens', None)
                         if prompt_tokens is None or completion_tokens is None:
                              logger.warning(f"Could not find populated prompt/completion tokens in usage object for DeepSeek model {client.model_name}.")
                     else:
                          logger.warning(f"Could not find populated usage information in response object for DeepSeek model {client.model_name}.")
                 except AttributeError:
                      logger.warning(f"Token usage attribute ('usage') is missing in the response structure for DeepSeek model {client.model_name}.")
            # --- Add DeepSeek Token Extraction --- End ---

            result["prompt_tokens"] = prompt_tokens
            result["completion_tokens"] = completion_tokens
            if prompt_tokens is not None or completion_tokens is not None:
                 logger.debug(f"Token Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
            else:
                 logger.debug(f"Token Usage information was not available or found for {client.provider}/{client.model_name}.")

        except Exception as e:
            # Catch any other unexpected errors during token extraction
            logger.warning(f"Unexpected error extracting token usage for {client.provider}/{client.model_name}: {e}", exc_info=True)
        # --- Extract Token Usage --- End ---
        
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
                result = generate_scad_for_task(task, model_config, output_dir)
                results.append(result)
                
                # Log success/failure
                if result["success"]:
                    logger.info(f"  Successfully generated: {os.path.basename(result['output_path'])}")
                else:
                    logger.error(f"  Failed to generate: {result['error']}")
                
            except Exception as e:
                logger.error(f"Unexpected error generating SCAD for {task_id} with {model_config.get('name')}: {e}")
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
            duration = f"{result['llm_duration_seconds']:.2f}s" if result["llm_duration_seconds"] else "N/A"
            
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