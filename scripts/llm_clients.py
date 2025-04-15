#!/usr/bin/env python3
"""
LLM Client Wrappers for CadEval

This module provides client wrappers for interacting with various Large Language Models
(LLMs) from OpenAI, Anthropic, and Google.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import requests
from requests.exceptions import RequestException, Timeout
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.logger_setup import get_logger
from scripts.config_loader import get_config, ConfigError

# Initialize logger for this module
logger = get_logger(__name__)
config = get_config()

class LLMError(Exception):
    """Exception raised for errors when interacting with LLMs."""
    pass


class BaseLLMClient:
    """
    Base class for all LLM client wrappers.
    """
    
    def __init__(self, provider: str, model_name: str, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            provider: The LLM provider name ('openai', 'anthropic', 'google')
            model_name: The specific model to use
            **kwargs: Additional model-specific parameters
        """
        self.provider = provider
        self.model_name = model_name
        self.max_retries = kwargs.get('max_retries', 3)
        self.parameters = kwargs
        self.last_response = None # Initialize last_response attribute
        
        # Logging initialization
        logger.info(f"Initializing {provider} client with model: {model_name}")
        
        # Initialize the client with appropriate API key
        try:
            self.api_key = config.get_api_key(provider)
            # Specific initialization will be done in subclasses
        except ConfigError as e:
            logger.error(f"Failed to initialize {provider} client: {e}")
            raise LLMError(f"Failed to initialize {provider} client: {e}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional parameters specific to this request
                      (overrides default parameters if provided)
        
        Returns:
            The generated text response
            
        Raises:
            LLMError: If text generation fails
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement generate_text method")
    
    def _log_request(self, prompt: str, params: Dict[str, Any]) -> None:
        """
        Log details of the LLM request.
        
        Args:
            prompt: The input prompt
            params: The parameters being used for the request
        """
        # Truncate prompt for logging if too long
        max_log_length = 1000
        truncated_prompt = (prompt[:max_log_length] + "...") if len(prompt) > max_log_length else prompt
        
        logger.debug(f"LLM Request: {self.provider} | {self.model_name}")
        logger.debug(f"Parameters: {json.dumps(params, indent=2)}")
        logger.debug(f"Prompt: {truncated_prompt}")
    
    def _log_response(self, response: str, duration: float) -> None:
        """
        Log details of the LLM response.
        
        Args:
            response: The generated text
            duration: Time taken to generate the response (in seconds)
        """
        # Truncate response for logging if too long
        max_log_length = 1000
        truncated_response = (response[:max_log_length] + "...") if len(response) > max_log_length else response
        
        logger.debug(f"LLM Response (after {duration:.2f}s): {truncated_response}")


class OpenAIClient(BaseLLMClient):
    """
    Client wrapper for OpenAI API.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name: The specific OpenAI model to use
            **kwargs: Additional model-specific parameters
        """
        super().__init__('openai', model_name, **kwargs)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Set default parameters - store the intended max length generically
        self.default_params = {
            'temperature': kwargs.get('temperature', 0.7),
            '_max_output_length': kwargs.get('max_tokens', 1000), # Use internal name
            'top_p': kwargs.get('top_p', 1.0),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
            'presence_penalty': kwargs.get('presence_penalty', 0.0),
        }
        # Add the correct max length parameter name based on model
        if 'o1-' in self.model_name: # Heuristic check for o1 models
             self.default_params['max_completion_tokens'] = self.default_params.pop('_max_output_length')
        else:
             self.default_params['max_tokens'] = self.default_params.pop('_max_output_length')
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError, Timeout)),
        reraise=True
    )
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional parameters for this specific request
        
        Returns:
            The generated text response
            
        Raises:
            LLMError: If text generation fails
        """
        # Merge default parameters with request-specific ones
        params = {**self.default_params, **kwargs}
        
        # Ensure the correct max length parameter is used based on the model
        # (This handles overrides via kwargs as well)
        max_len_param_name = 'max_completion_tokens' if 'o1-' in self.model_name else 'max_tokens'
        if max_len_param_name == 'max_completion_tokens' and 'max_tokens' in params:
            params['max_completion_tokens'] = params.pop('max_tokens')
        elif max_len_param_name == 'max_tokens' and 'max_completion_tokens' in params:
             params['max_tokens'] = params.pop('max_completion_tokens')

        # Log the request (using the final parameters)
        self._log_request(prompt, params)
        
        start_time = time.time()
        try:
            # Pass parameters dynamically
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params # Pass the prepared parameters dictionary
            )
            
            generated_text = response.choices[0].message.content
            self.last_response = response # Store the raw response
            duration = time.time() - start_time
            
            # Log the response
            self._log_response(generated_text, duration)
            
            return generated_text
            
        except openai.BadRequestError as e:
             # Specific handling for bad requests which might include parameter issues
             logger.error(f"OpenAI API BadRequestError: {str(e)}")
             # Re-raise as LLMError for consistent handling upstream
             raise LLMError(f"OpenAI API BadRequestError: {str(e)}")
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMError(f"OpenAI API error: {str(e)}")


class OpenAIReasoningClient(BaseLLMClient):
    """
    Client wrapper for OpenAI Reasoning models (o1, o3) using the Responses API.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the OpenAI Reasoning client.
        
        Args:
            model_name: The specific OpenAI reasoning model to use (e.g., o1, o3-mini)
            **kwargs: Additional model-specific parameters (effort, max_output_tokens)
        """
        super().__init__('openai', model_name, **kwargs)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Set default parameters for the Responses API
        self.default_params = {
            'reasoning': {'effort': kwargs.get('effort', 'medium')},
            # Note: max_output_tokens limits TOTAL tokens (reasoning + output)
            'max_output_tokens': kwargs.get('max_output_tokens', kwargs.get('max_tokens', 10000)) # Use higher default if not specified
        }
        # Remove max_tokens if it was passed via kwargs from config designed for chat models
        if 'max_tokens' in self.parameters:
             del self.parameters['max_tokens']

    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError, Timeout)),
        reraise=True
    )
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI Responses API (for o1/o3 models).
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional parameters for this specific request (effort, max_output_tokens)
        
        Returns:
            The generated text response
            
        Raises:
            LLMError: If text generation fails or is incomplete without output
        """
        # Merge default parameters with request-specific ones
        params = {**self.default_params, **kwargs}
        # Ensure nested 'reasoning' dict is handled correctly if passed in kwargs
        if 'effort' in kwargs:
            params['reasoning'] = {'effort': kwargs['effort']}
            del params['effort'] # Remove top-level if present
        
        # Format input for the Responses API
        input_messages = [{"role": "user", "content": prompt}]
        
        # Log the request (adjust logging if needed for different params)
        log_params_for_display = {k: v for k, v in params.items()}
        log_params_for_display['reasoning_effort'] = params.get('reasoning', {}).get('effort', 'default')
        if 'reasoning' in log_params_for_display: del log_params_for_display['reasoning']
        self._log_request(prompt, log_params_for_display)
        
        start_time = time.time()
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=input_messages,
                reasoning=params['reasoning'],
                max_output_tokens=params['max_output_tokens']
            )
            
            duration = time.time() - start_time
            
            # Check response status
            if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
                error_msg = "Generation stopped due to max_output_tokens limit."
                if response.output_text:
                    logger.warning(f"{error_msg} Partial output returned.")
                    # Log partial response
                    self._log_response(response.output_text, duration)
                    return response.output_text # Return partial text
                else:
                    error_msg += " No output generated (limit reached during reasoning)."
                    logger.error(error_msg)
                    raise LLMError(error_msg)
            elif response.status != "completed":
                 error_msg = f"Generation finished with unexpected status: {response.status}"
                 logger.error(error_msg)
                 raise LLMError(error_msg)

            generated_text = response.output_text
            self.last_response = response # Store the raw response
            
            # Log the successful response
            self._log_response(generated_text, duration)
            
            return generated_text
            
        except openai.BadRequestError as e:
            logger.error(f"OpenAI API BadRequestError: {str(e)}")
            raise LLMError(f"OpenAI API BadRequestError: {str(e)}")
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMError(f"OpenAI API error: {str(e)}")
        except Timeout as e:
            logger.error(f"Request to OpenAI API timed out: {str(e)}")
            raise LLMError(f"Request to OpenAI API timed out: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI Responses API: {str(e)}")
            raise LLMError(f"Unexpected error with OpenAI Responses API: {str(e)}")


class AnthropicClient(BaseLLMClient):
    """
    Client wrapper for Anthropic API.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Anthropic client.
        
        Args:
            model_name: The specific Anthropic model to use
            **kwargs: Additional model-specific parameters
        """
        super().__init__('anthropic', model_name, **kwargs)
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Set default parameters
        self.default_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1000),
            'top_p': kwargs.get('top_p', 1.0),
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError, Timeout)),
        reraise=True
    )
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional parameters for this specific request
        
        Returns:
            The generated text response
            
        Raises:
            LLMError: If text generation fails
        """
        # Merge default parameters with request-specific ones
        params = {**self.default_params, **kwargs}
        
        # Log the request
        self._log_request(prompt, params)
        
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params['temperature'],
                max_tokens=params['max_tokens'],
                top_p=params['top_p'],
            )
            
            generated_text = response.content[0].text
            self.last_response = response # Store the raw response
            duration = time.time() - start_time
            
            # Log the response
            self._log_response(generated_text, duration)
            
            return generated_text
            
        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise LLMError(f"Anthropic API error: {str(e)}")
        except Timeout as e:
            logger.error(f"Request to Anthropic API timed out: {str(e)}")
            raise LLMError(f"Request to Anthropic API timed out: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with Anthropic API: {str(e)}")
            raise LLMError(f"Unexpected error with Anthropic API: {str(e)}")


class GoogleAIClient(BaseLLMClient):
    """
    Client wrapper for Google AI (Gemini) API.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Google AI client.
        
        Args:
            model_name: The specific Google AI model to use
            **kwargs: Additional model-specific parameters
        """
        super().__init__('google', model_name, **kwargs)
        
        # Initialize Google AI client
        genai.configure(api_key=self.api_key)
        
        # Set default parameters
        self.default_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_output_tokens': kwargs.get('max_tokens', 1000),
            'top_p': kwargs.get('top_p', 1.0),
            'top_k': kwargs.get('top_k', 40),
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RequestException, Timeout)),
        reraise=True
    )
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Google AI (Gemini) API.
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional parameters for this specific request
        
        Returns:
            The generated text response
            
        Raises:
            LLMError: If text generation fails
        """
        # Merge default parameters with request-specific ones
        params = {**self.default_params, **kwargs}
        
        # Log the request
        self._log_request(prompt, params)
        
        start_time = time.time()
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    'temperature': params['temperature'],
                    'max_output_tokens': params['max_output_tokens'],
                    'top_p': params['top_p'],
                    'top_k': params['top_k'],
                }
            )
            
            response = model.generate_content(prompt)
            generated_text = response.text
            self.last_response = response # Store the raw response
            duration = time.time() - start_time
            
            # Log the response
            self._log_response(generated_text, duration)
            
            return generated_text
            
        except RequestException as e:
            logger.error(f"Google AI API request error: {str(e)}")
            raise LLMError(f"Google AI API request error: {str(e)}")
        except Timeout as e:
            logger.error(f"Request to Google AI API timed out: {str(e)}")
            raise LLMError(f"Request to Google AI API timed out: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with Google AI API: {str(e)}")
            raise LLMError(f"Unexpected error with Google AI API: {str(e)}")


def create_llm_client(model_config: Dict[str, Any]) -> BaseLLMClient:
    """
    Factory function to create an appropriate LLM client based on configuration.
    
    Args:
        model_config: Dictionary containing model configuration
                     Must include 'provider' and 'name' keys
    
    Returns:
        An instance of the appropriate LLM client
        
    Raises:
        LLMError: If the provider is not supported or configuration is invalid
    """
    if not isinstance(model_config, dict):
        raise LLMError("Model configuration must be a dictionary")
    
    provider = model_config.get('provider', '').lower()
    model_name = model_config.get('name', '')
    
    if not provider or not model_name:
        raise LLMError("Model configuration must include 'provider' and 'name'")
    
    # Extract parameters from config, excluding provider and name
    params = {k: v for k, v in model_config.items() if k not in ['provider', 'name']}
    
    # Create the appropriate client based on provider
    try:
        if provider == 'openai':
            # Check if it's a reasoning model (o1/o3)
            if model_name.startswith('o1') or model_name.startswith('o3'):
                 logger.info(f"Detected OpenAI reasoning model: {model_name}. Using Responses API client.")
                 return OpenAIReasoningClient(model_name, **params)
            else:
                 return OpenAIClient(model_name, **params) # Use standard ChatCompletion client
        elif provider == 'anthropic':
            return AnthropicClient(model_name, **params)
        elif provider == 'google':
            return GoogleAIClient(model_name, **params)
        else:
            raise LLMError(f"Unsupported LLM provider: {provider}")
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Failed to create LLM client for {provider}: {str(e)}")


if __name__ == "__main__":
    # Example usage when run directly
    import argparse
    from scripts.logger_setup import setup_logger
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Test LLM clients with a simple prompt")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google"], help="LLM provider to test")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--prompt", default="Write a haiku about programming in Python", help="Test prompt to send")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(__name__, level=log_level, log_file="logs/llm_clients_test.log")
    
    try:
        # If specific provider/model is provided, test just that one
        if args.provider and args.model:
            model_config = {
                'provider': args.provider,
                'name': args.model,
                'temperature': 0.7,
                'max_tokens': 150
            }
            
            client = create_llm_client(model_config)
            print(f"\nTesting {args.provider.capitalize()} with model {args.model}...")
            print(f"Prompt: {args.prompt}")
            response = client.generate_text(args.prompt)
            print(f"\nResponse:\n{response}")
            
        else:
            # If no specific provider/model, test all configured models from config
            print("\nTesting all configured LLM models...")
            
            models = config.get('llm.models', [])
            if not models:
                print("No LLM models configured in config.yaml")
                sys.exit(1)
            
            for model_config in models:
                try:
                    provider = model_config.get('provider', '').capitalize()
                    model_name = model_config.get('name', '')
                    
                    print(f"\n---------- Testing {provider} | {model_name} ----------")
                    print(f"Prompt: {args.prompt}")
                    
                    client = create_llm_client(model_config)
                    response = client.generate_text(args.prompt)
                    
                    print(f"\nResponse:\n{response}")
                    print("-" * 50)
                    
                except LLMError as e:
                    print(f"Error: {e}")
                    print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 