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
        logger.debug(f"Parameters: {json.dumps(params, indent=2, default=str)}")
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


# --- Remove OpenAIClient and OpenAIReasoningClient ---

# --- Add UnifiedOpenAIClient --- Start ---
class UnifiedOpenAIClient(BaseLLMClient):
    """
    Unified client wrapper for OpenAI API using the Responses API (/v1/responses).
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the unified OpenAI client.

        Args:
            model_name: The specific OpenAI model to use.
            **kwargs: Additional model-specific parameters from config.
        """
        init_kwargs = kwargs.copy()
        init_kwargs.pop('provider', None)
        init_kwargs.pop('name', None)
        super().__init__(provider='openai', model_name=model_name, **init_kwargs)
        self.client = OpenAI(api_key=self.api_key)
        self.default_params = {
            'max_output_tokens': kwargs.get('max_tokens', kwargs.get('max_output_tokens', 8000)),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 1.0),
            'reasoning': {
                'effort': kwargs.get('effort', 'medium')
            },
        }
        self.parameters.pop('max_tokens', None)
        self.parameters.pop('frequency_penalty', None)
        self.parameters.pop('presence_penalty', None)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError, Timeout)),
        reraise=True
    )
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI Responses API (/v1/responses).
        Conditionally includes reasoning parameters based on model name.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional parameters for this specific request.

        Returns:
            The generated text response.

        Raises:
            LLMError: If text generation fails.
        """
        request_params = {**self.default_params, **kwargs}
        if 'max_tokens' in kwargs:
            request_params['max_output_tokens'] = kwargs['max_tokens']
        request_params.pop('max_tokens', None)

        # --- Conditionally adjust parameters based on model type --- Start ---
        # Models starting with 'o' (e.g., o1, o3, o4) don't support temp/top_p via /v1/responses
        is_o_model = self.model_name.startswith('o')

        if is_o_model:
            # o1/o3 models DO support 'reasoning', but NOT temp/top_p via /v1/responses
            if 'temperature' in request_params:
                logger.debug(f"Model {self.model_name} does not support 'temperature' via /v1/responses. Removing it.")
                del request_params['temperature']
            if 'top_p' in request_params:
                logger.debug(f"Model {self.model_name} does not support 'top_p' via /v1/responses. Removing it.")
                del request_params['top_p']
        else:
            # Other models (e.g., gpt-4.1) support temp/top_p, but NOT 'reasoning' via /v1/responses
            if 'reasoning' in request_params:
                logger.debug(f"Model {self.model_name} does not support 'reasoning' parameter via /v1/responses. Removing it.")
                del request_params['reasoning']
        # --- Conditionally adjust parameters based on model type --- End ---

        self._log_request(prompt, request_params)

        start_time = time.time()
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[{"role": "user", "content": prompt}],
                **request_params
            )
            self.last_response = response # Store raw response immediately

            # --- Extract text - More Resilient --- Start ---
            generated_text = ""
            response_status = getattr(response, 'status', 'unknown')

            if response_status != "completed":
                logger.warning(f"OpenAI response status for {self.model_name} was '{response_status}', not 'completed'. Returning empty text.")
                if hasattr(response, 'error') and response.error:
                     logger.warning(f"Response error details: {response.error}")
            else:
                # Status is completed, attempt text extraction
                try:
                    # Attempt 1: Direct .output_text
                    if hasattr(response, 'output_text') and response.output_text:
                        generated_text = response.output_text
                        logger.debug(f"Extracted text using response.output_text for {self.model_name}")
                    # Attempt 2: Nested structure
                    elif (response.output and
                          isinstance(response.output, list) and len(response.output) > 0 and
                          hasattr(response.output[0], 'content') and
                          isinstance(response.output[0].content, list) and len(response.output[0].content) > 0 and
                          hasattr(response.output[0].content[0], 'text')):
                        generated_text = response.output[0].content[0].text
                        logger.debug(f"Extracted text using response.output[0].content[0].text for {self.model_name}")
                    else:
                        # Status completed, but text structure not found - DO NOT RAISE ERROR
                        logger.warning(f"OpenAI response status completed for {self.model_name}, but could not find text in expected structures (.output_text or .output[0].content[0].text). Returning empty text.")
                        # Ensure generated_text remains ""
                except (AttributeError, IndexError, TypeError) as e:
                     # Error accessing attributes during extraction - DO NOT RAISE ERROR
                     logger.warning(f"Error accessing text in completed OpenAI response structure for model {self.model_name}: {e}. Returning empty text.")
                     # Ensure generated_text remains ""

            # Log if empty text resulted
            if not generated_text:
                 logger.warning(f"Final extracted text is empty for OpenAI model {self.model_name}. Original response status was '{response_status}'.")
            # --- Extract text - More Resilient --- End ---

            duration = time.time() - start_time
            self._log_response(generated_text, duration)
            return generated_text

        except openai.BadRequestError as e:
             logger.error(f"OpenAI API BadRequestError using /v1/responses: {str(e)}")
             raise LLMError(f"OpenAI API BadRequestError: {str(e)}")
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenAI API error using /v1/responses: {str(e)}")
            raise # Reraise for tenacity retry
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI /v1/responses or processing its response: {str(e)}", exc_info=True)
            # This catch block might be hit if unexpected errors occur OUTSIDE text extraction
            raise LLMError(f"Unexpected error calling OpenAI /v1/responses or processing response: {str(e)}")
# --- Add UnifiedOpenAIClient --- End ---


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
        init_kwargs = kwargs.copy()
        init_kwargs.pop('provider', None)
        init_kwargs.pop('name', None)
        super().__init__(provider='anthropic', model_name=model_name, **init_kwargs)
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.default_params = {
            'max_tokens': kwargs.get('max_tokens', 4000),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 1.0),
        }
        self.parameters.pop('max_tokens', None) # Cleanup informational params

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError, Timeout)),
        reraise=True
    )
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic API.
        Uses the thinking parameter if enabled in config.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional parameters for this specific request

        Returns:
            The generated text response

        Raises:
            LLMError: If text generation fails
        """
        # Merge default params with request-specific ones, prioritize request-specific
        request_params = {**self.default_params, **kwargs}
        # Construct message format for Anthropic
        messages = [{"role": "user", "content": prompt}]

        # --- Check for Extended Thinking flags --- Start ---
        use_extended_thinking = self.parameters.get('anthropic_extended_thinking_enabled', False)
        thinking_budget = self.parameters.get('anthropic_thinking_budget_tokens')
        # --- Check for Extended Thinking flags --- End ---

        # Log request details (use the merged params)
        self._log_request(prompt, request_params)

        start_time = time.time()
        try:
            # --- Select API endpoint based on flags --- Start ---
            # Prepare call arguments
            api_call_args = {
                "model": self.model_name,
                "messages": messages,
                **request_params
            }

            # Conditionally add the thinking parameter
            if use_extended_thinking and thinking_budget:
                logger.debug(f"Adding 'thinking' parameter for {self.model_name}.")
                # Ensure incompatible parameters are removed when thinking is enabled
                api_call_args.pop('temperature', None) # Although API requires 1.0, removing might be safer than forcing 1.0 here
                api_call_args.pop('top_p', None)
                api_call_args.pop('top_k', None) # Remove top_k just in case it's added later

                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": int(thinking_budget) # Ensure it's an int
                }
                api_call_args["thinking"] = thinking_config
            else:
                if use_extended_thinking:
                    logger.warning(f"Extended thinking enabled for {self.model_name}, but budget_tokens or betas missing in config. Using standard endpoint.")

            # Always use the standard endpoint
            logger.debug(f"Calling client.messages.create for {self.model_name}.")
            response = self.client.messages.create(**api_call_args)
            # --- Select API endpoint based on flags --- End ---

            self.last_response = response # Store raw response

            # Extract text content (handle potential list structure)
            generated_text = ""
            if response.content:
                if isinstance(response.content, list):
                    for block in response.content:
                        if hasattr(block, 'text') and block.text:
                            generated_text += block.text
                elif hasattr(response.content, 'text') and response.content.text: # Should not happen based on docs, but safer
                    generated_text = response.content.text
            
            if not generated_text:
                 logger.warning(f"Anthropic response content was empty or text block not found for model {self.model_name}.")

            duration = time.time() - start_time
            self._log_response(generated_text, duration)
            return generated_text
        except anthropic.APIStatusError as e:
            logger.error(f"Anthropic API status error: {str(e)}")
            raise LLMError(f"Anthropic API error: {str(e)}") from e
        except anthropic.APIConnectionError as e:
            logger.error(f"Anthropic API connection error: {str(e)}")
            raise LLMError(f"Anthropic connection error: {str(e)}") from e
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic rate limit exceeded: {str(e)}")
            raise LLMError(f"Anthropic rate limit error: {str(e)}") from e
        except anthropic.APIError as e:
            logger.error(f"General Anthropic API error: {str(e)}")
            raise LLMError(f"Anthropic API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic call: {e}", exc_info=True)
            raise LLMError(f"Unexpected error during Anthropic call: {e}") from e


class GoogleAIClient(BaseLLMClient):
    """
    Client wrapper for Google AI API (Gemini).
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Google AI client.

        Args:
            model_name: The specific Google model to use
            **kwargs: Additional model-specific parameters
        """
        init_kwargs = kwargs.copy()
        init_kwargs.pop('provider', None)
        init_kwargs.pop('name', None)
        super().__init__(provider='google', model_name=model_name, **init_kwargs)
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name)
        self.default_config = genai.types.GenerationConfig(
            candidate_count=kwargs.get('candidate_count', 1),
            max_output_tokens=kwargs.get('max_tokens', 8192),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
        )
        self.parameters.pop('max_tokens', None)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # Use broader exceptions as specific Google AI API errors might vary
        retry=retry_if_exception_type((RequestException, Timeout, genai.types.generation_types.BlockedPromptException, genai.types.generation_types.StopCandidateException)),
        reraise=True
    )
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Google AI API.

        Args:
            prompt: The input prompt text
            **kwargs: Additional generation config parameters for this request

        Returns:
            The generated text response

        Raises:
            LLMError: If text generation fails
        """
        # Create generation config for this request
        # Start with defaults and update with kwargs
        request_config_dict = {
             'max_output_tokens': kwargs.get('max_tokens', self.default_config.max_output_tokens),
             'temperature': kwargs.get('temperature', self.default_config.temperature),
             'top_p': kwargs.get('top_p', self.default_config.top_p),
             'candidate_count': kwargs.get('candidate_count', self.default_config.candidate_count),
        }
        # Filter out None values before creating GenerationConfig
        filtered_config = {k: v for k, v in request_config_dict.items() if v is not None}
        request_config = genai.types.GenerationConfig(**filtered_config)


        # Log the request (approximate parameters)
        self._log_request(prompt, filtered_config)

        start_time = time.time()
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=request_config
            )

            # Handle potential safety blocks or empty responses
            if not response.candidates:
                 block_reason = "Unknown"
                 safety_ratings = []
                 try:
                     block_reason = response.prompt_feedback.block_reason.name
                     safety_ratings = response.prompt_feedback.safety_ratings
                 except AttributeError:
                     pass # Keep default reason
                 logger.error(f"Google AI generation blocked. Reason: {block_reason}, SafetyRatings: {safety_ratings}")
                 raise LLMError(f"Google AI generation blocked. Reason: {block_reason}")

            # Extract text, handling potential lack of text part
            generated_text = ""
            try:
                generated_text = response.text
            except ValueError as e:
                # This can happen if the response is stopped early due to safety/length
                logger.warning(f"Could not extract text from Google AI response (ValueError): {e}. Checking parts.")
                try:
                     if response.candidates and response.candidates[0].content.parts:
                          generated_text = response.candidates[0].content.parts[0].text
                          logger.info("Extracted text from response parts after initial ValueError.")
                     else:
                          logger.error("No usable text found in Google AI response parts after ValueError.")
                          raise LLMError("No usable text found in Google AI response parts.")
                except Exception as part_e:
                     logger.error(f"Error extracting text from Google AI response parts: {part_e}")
                     raise LLMError(f"Error extracting text from Google AI response parts: {part_e}")
            
            if not generated_text:
                 logger.warning(f"Extracted empty text from Google AI response for model {self.model_name}")

            self.last_response = response # Store the raw response
            duration = time.time() - start_time

            # Log the response
            self._log_response(generated_text, duration)

            return generated_text

        except (RequestException, Timeout) as e:
            logger.error(f"Network error calling Google AI API: {str(e)}")
            raise # Reraise for tenacity
        except genai.types.generation_types.BlockedPromptException as e:
             logger.error(f"Google AI prompt blocked: {str(e)}")
             raise LLMError(f"Google AI prompt blocked: {str(e)}")
        except genai.types.generation_types.StopCandidateException as e:
             logger.warning(f"Google AI generation stopped unexpectedly: {str(e)}")
             # Attempt to return partial text if available
             try:
                 if e.candidate and hasattr(e.candidate, 'text'): return e.candidate.text
                 elif e.candidate and e.candidate.content and e.candidate.content.parts:
                      return e.candidate.content.parts[0].text
                 else: raise LLMError(f"Google AI generation stopped with no recoverable text: {str(e)}")
             except Exception as recovery_e:
                  logger.error(f"Failed to recover partial text after StopCandidateException: {recovery_e}")
                  raise LLMError(f"Google AI generation stopped with no recoverable text: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error calling Google AI API: {str(e)}", exc_info=True)
            raise LLMError(f"Unexpected error calling Google AI API: {str(e)}")


# --- Add DeepSeekClient --- Start ---
class DeepSeekClient(BaseLLMClient):
    """
    Client wrapper for DeepSeek API (OpenAI compatible).
    Handles specifics like base_url and potentially reasoning_content.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the DeepSeek client.

        Args:
            model_name: The specific DeepSeek model to use (e.g., 'deepseek-chat', 'deepseek-coder', 'deepseek-reasoner', 'deepseek-r1').
            **kwargs: Additional model-specific parameters from config.
        """
        init_kwargs = kwargs.copy()
        provider = init_kwargs.pop('provider', 'deepseek') # Ensure provider is set correctly
        init_kwargs.pop('name', None) # Remove 'name' if present, use model_name
        
        super().__init__(provider=provider, model_name=model_name, **init_kwargs)
        
        try:
            # Ensure api_key is loaded by super() first
            if not self.api_key:
                 raise ConfigError("DeepSeek API key not found or empty.")
                 
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=kwargs.get('base_url', "https://api.deepseek.com"), # Use config base_url or default
                timeout=kwargs.get('timeout_seconds', 600.0) # Increase default timeout to 600s (10 mins)
            )
            logger.info(f"DeepSeekClient initialized for model {self.model_name} using base_url: {self.client.base_url} with timeout: {self.client.timeout}") # Log the actual timeout used

        except ConfigError as e:
            logger.error(f"Configuration error initializing DeepSeekClient: {e}")
            raise LLMError(f"Configuration error for DeepSeek: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing DeepSeekClient for model {self.model_name}: {e}", exc_info=True)
            raise LLMError(f"Failed to initialize DeepSeek client for {self.model_name}: {e}")

        # Store default parameters, removing unsupported ones for reasoning models if necessary
        self.default_params = {
            'max_tokens': kwargs.get('max_tokens', 4096), # Default based on DeepSeek docs examples
            # Deepseek-reasoner (and likely r1) doesn't support these according to docs
            # Conditionally add/remove them in generate_text based on model type?
            # 'temperature': kwargs.get('temperature', 0.7), 
            # 'top_p': kwargs.get('top_p', 1.0),
            # 'frequency_penalty': kwargs.get('frequency_penalty', 0),
            # 'presence_penalty': kwargs.get('presence_penalty', 0),
        }
        
        # Add supported params if present in config
        for param in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if param in kwargs:
                self.default_params[param] = kwargs[param]
                
        # Check if this model is a reasoning model to apply specific rules
        # For now, let's assume models containing 'reasoner' or 'r1' are reasoning models
        self.is_reasoning_model = 'reasoner' in self.model_name or 'r1' in self.model_name # Simple check for now
        if self.is_reasoning_model:
             logger.info(f"Model {self.model_name} identified as a reasoning model. Temperature/TopP etc. will be ignored by API.")
             # Keep params in default_params, API ignores them, allows using same config structure


    def generate_text(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """
        Generate text using DeepSeek API (chat completions endpoint).

        Handles both single string prompts and message lists.
        Strips 'reasoning_content' from assistant messages if present in the input list.

        Args:
            prompt: The input prompt text (string) or a list of messages (dicts).
            **kwargs: Additional parameters specific to this request
                      (overrides default parameters if provided).

        Returns:
            The generated text content.

        Raises:
            LLMError: If text generation fails.
        """
        request_params = {**self.default_params, **kwargs}

        # --- Prepare messages ---
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # IMPORTANT: DeepSeek reasoning models require removing 'reasoning_content' 
            # from previous assistant messages before sending.
            messages = []
            for msg in prompt:
                 # Shallow copy the message dict
                 new_msg = msg.copy() 
                 # Remove 'reasoning_content' if it exists (Deepseek API returns 400 otherwise)
                 if 'reasoning_content' in new_msg:
                     del new_msg['reasoning_content']
                     logger.debug("Removed 'reasoning_content' from assistant message before sending.")
                 messages.append(new_msg)
        else:
            raise LLMError("Invalid prompt type for DeepSeekClient. Must be str or List[Dict].")

        # --- Remove unsupported params for reasoning models (API ignores them, but good practice) ---
        # According to docs, reasoning models ignore temp, top_p, presence_penalty, frequency_penalty
        # The docs also state logprobs/top_logprobs cause errors. Ensure they aren't passed.
        params_to_remove_for_reasoning = ['temperature', 'top_p', 'presence_penalty', 'frequency_penalty']
        params_that_error = ['logprobs', 'top_logprobs']

        final_request_params = request_params.copy()
        
        for param in params_that_error:
            if param in final_request_params:
                logger.warning(f"Removing unsupported parameter '{param}' for DeepSeek model {self.model_name}.")
                del final_request_params[param]

        if self.is_reasoning_model:
            for param in params_to_remove_for_reasoning:
                if param in final_request_params:
                    # Don't strictly remove, API ignores them. Log if they were provided.
                     logger.debug(f"Parameter '{param}' provided for reasoning model {self.model_name}. API will ignore it.")
                     # del final_request_params[param] # Optionally remove if needed

        # --- Log and Make API Call ---
        self._log_request(messages[-1]['content'], final_request_params) # Log last user message content
        
        start_time = time.time()
        try:
            # Use standard chat completions endpoint
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False, # Assuming non-streaming for now based on other clients
                **final_request_params
            )
            self.last_response = response # Store raw response

            # --- Extract text ---
            if response.choices:
                message = response.choices[0].message
                generated_text = message.content or "" # Get content, default to empty string
                
                # Log if reasoning_content was present (informational)
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                     logger.debug(f"Received 'reasoning_content' from model {self.model_name}.")
                     # We are not currently using/storing this, just the main 'content'
                     # You could potentially store it like: self.last_reasoning_content = message.reasoning_content
                
                if not generated_text:
                     logger.warning(f"DeepSeek API returned an empty message content for model {self.model_name}.")

            else:
                logger.warning(f"DeepSeek API response had no choices for model {self.model_name}. Returning empty text.")
                generated_text = ""

            duration = time.time() - start_time
            self._log_response(generated_text, duration)
            return generated_text

        except openai.APIError as e:
            # Handle API-specific errors (e.g., 4xx, 5xx from DeepSeek)
            logger.error(f"DeepSeek API error: {e.status_code} - {e.message}", exc_info=True)
            raise LLMError(f"DeepSeek API error ({e.status_code}): {e.message}")
        except openai.APITimeoutError as e:
             logger.error(f"DeepSeek API request timed out: {e}", exc_info=True)
             raise LLMError(f"DeepSeek API request timed out: {e}")
        except openai.APIConnectionError as e:
             logger.error(f"DeepSeek API connection error: {e}", exc_info=True)
             raise LLMError(f"DeepSeek API connection error: {e}")
        except Exception as e:
            # Catch any other unexpected errors during the API call or processing
            # --- Log Raw Response on JSON Error --- Start ---
            raw_response_text = "<Could not read raw response>"
            if isinstance(e, json.JSONDecodeError) and hasattr(e, 'doc'):
                 raw_response_text = e.doc # The doc attribute holds the string that failed to parse
            elif hasattr(e, 'response') and hasattr(e.response, 'text'): # Fallback for other potential errors
                 try:
                     raw_response_text = e.response.text
                 except Exception:
                      pass
            logger.error(f"Unexpected error calling DeepSeek API for model {self.model_name}: {str(e)}\nRaw Response Text: {raw_response_text[:1000]}...", exc_info=True) # Log first 1000 chars
            # --- Log Raw Response on JSON Error --- End ---
            raise LLMError(f"Unexpected error interacting with DeepSeek API: {str(e)}")

# --- Add DeepSeekClient --- End ---

# --- Add QwenClient --- Start ---
class QwenClient(BaseLLMClient):
    """
    Client wrapper for Alibaba Qwen API (OpenAI compatible).
    Uses the DashScope API endpoint.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Qwen client.

        Args:
            model_name: The specific Qwen model to use (e.g., 'qwen-plus', 'qwen-turbo').
            **kwargs: Additional model-specific parameters from config.
        """
        init_kwargs = kwargs.copy()
        provider = init_kwargs.pop('provider', 'qwen') # Ensure provider is set correctly
        init_kwargs.pop('name', None) # Remove 'name' if present, use model_name
        
        super().__init__(provider=provider, model_name=model_name, **init_kwargs)
        
        try:
            if not self.api_key:
                 raise ConfigError("Qwen (DashScope) API key not found or empty.")
            
            # The base_url should be up to /v1, the openai library appends /chat/completions
            # For international users:
            qwen_base_url = kwargs.get('base_url', "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
            # For users in China (if applicable, could be made configurable):
            # qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

            self.client = OpenAI(
                api_key=self.api_key, # This should be your DASHSCOPE_API_KEY
                base_url=qwen_base_url,
                timeout=kwargs.get('timeout_seconds', 600.0) # Default to 10 minutes timeout
            )
            logger.info(f"QwenClient initialized for model {self.model_name} using base_url: {self.client.base_url} with timeout: {self.client.timeout}")

        except ConfigError as e:
            logger.error(f"Configuration error initializing QwenClient: {e}")
            raise LLMError(f"Configuration error for Qwen: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing QwenClient for model {self.model_name}: {e}", exc_info=True)
            raise LLMError(f"Failed to initialize Qwen client for {self.model_name}: {e}")

        # Default parameters, Qwen API might ignore some if not supported by the specific model
        self.default_params = {
            'max_tokens': kwargs.get('max_tokens', 2048), # A common default, adjust as needed
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 1.0),
        }
        # Add other common OpenAI params if present in config
        for param in ['frequency_penalty', 'presence_penalty']:
            if param in kwargs:
                self.default_params[param] = kwargs[param]

        # --- Add Qwen-specific reasoning/thinking parameters --- Start ---
        self.qwen_reasoning_enabled = kwargs.get('qwen_reasoning_enabled', False)
        self.qwen_thinking_budget_tokens = kwargs.get('qwen_thinking_budget_tokens', None)
        if self.qwen_reasoning_enabled:
            logger.info(f"Qwen model {self.model_name} configured with reasoning enabled.")
            if self.qwen_thinking_budget_tokens is not None:
                logger.info(f"Qwen model {self.model_name} configured with thinking budget: {self.qwen_thinking_budget_tokens} tokens.")
        # --- Add Qwen-specific reasoning/thinking parameters --- End ---

    @retry(
        stop=stop_after_attempt(3), # Or use self.max_retries
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError, Timeout)),
        reraise=True
    )
    def generate_text(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """
        Generate text using Qwen API (chat completions endpoint).

        Args:
            prompt: The input prompt text (string) or a list of messages (dicts).
            **kwargs: Additional parameters specific to this request.

        Returns:
            The generated text content.

        Raises:
            LLMError: If text generation fails.
        """
        request_params = {**self.default_params, **kwargs}

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt # Assume it's already in the correct format
        else:
            raise LLMError("Invalid prompt type for QwenClient. Must be str or List[Dict].")
        
        log_prompt_content = messages[-1]['content'] if messages else ""
        self._log_request(log_prompt_content, request_params) # Log original request_params before modification
        
        # --- Conditionally add Qwen reasoning/thinking parameters --- Start ---
        api_call_params = request_params.copy()
        extra_body_params = {} # Initialize

        if self.qwen_reasoning_enabled:
            extra_body_params["enable_thinking"] = True
            logger.debug(f"Qwen reasoning enabled via extra_body.")

            if self.qwen_thinking_budget_tokens is not None:
                extra_body_params["thinking_budget"] = int(self.qwen_thinking_budget_tokens)
                logger.debug(f"Qwen thinking_budget set to {self.qwen_thinking_budget_tokens} via extra_body.")
        else:
            extra_body_params["enable_thinking"] = False
            logger.debug(f"Qwen reasoning NOT enabled, explicitly setting enable_thinking=False via extra_body for non-streaming call.")
                
        # Determine if streaming should be used
        use_streaming = self.qwen_reasoning_enabled
        if use_streaming:
            logger.debug(f"Qwen API call will use streaming because reasoning is enabled.")
        else:
            logger.debug(f"Qwen API call will NOT use streaming because reasoning is not enabled.")

        start_time = time.time()
        try:
            if use_streaming:
                stream_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    extra_body=extra_body_params,
                    **api_call_params
                )
                generated_text = ""
                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        generated_text += chunk.choices[0].delta.content
                self.last_response = None # Streaming response object is not a simple last response
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    extra_body=extra_body_params,
                    **api_call_params
                )
                self.last_response = response
                if response.choices:
                    message = response.choices[0].message
                    generated_text = message.content or ""
                    if not generated_text:
                        logger.warning(f"Qwen API returned an empty message content for model {self.model_name} (non-streaming).")
                else:
                    logger.warning(f"Qwen API response had no choices for model {self.model_name} (non-streaming). Returning empty text.")
                    generated_text = ""

            duration = time.time() - start_time
            self._log_response(generated_text, duration)
            return generated_text

        except openai.APIError as e:
            logger.error(f"Qwen API error: {e.status_code} - {e.message}", exc_info=True)
            raw_body = e.body # Can contain more details from Qwen API
            logger.error(f"Qwen API error body: {raw_body}")
            raise LLMError(f"Qwen API error ({e.status_code}): {e.message}. Body: {raw_body}")
        except openai.APITimeoutError as e:
             logger.error(f"Qwen API request timed out: {e}", exc_info=True)
             raise LLMError(f"Qwen API request timed out: {e}")
        except openai.APIConnectionError as e:
             logger.error(f"Qwen API connection error: {e}", exc_info=True)
             raise LLMError(f"Qwen API connection error: {e}")
        except Exception as e:
            raw_response_text = "<Could not read raw response>"
            if isinstance(e, json.JSONDecodeError) and hasattr(e, 'doc'):
                 raw_response_text = e.doc
            elif hasattr(e, 'response') and hasattr(e.response, 'text'):
                 try:
                     raw_response_text = e.response.text
                 except Exception: pass
            logger.error(f"Unexpected error calling Qwen API for model {self.model_name}: {str(e)}\nRaw Response Text (partial): {raw_response_text[:1000]}", exc_info=True)
            raise LLMError(f"Unexpected error interacting with Qwen API: {str(e)}")

# --- Add QwenClient --- End ---


def create_llm_client(model_config: Dict[str, Any]) -> BaseLLMClient:
    """
    Factory function to create the appropriate LLM client based on provider.

    Args:
        model_config: Dictionary containing model configuration (name, provider, etc.)

    Returns:
        An instance of the appropriate BaseLLMClient subclass.

    Raises:
        ValueError: If the provider is unknown or unsupported, or config is invalid.
        LLMError: If client initialization fails.
    """
    if not isinstance(model_config, dict):
        raise ValueError("Model configuration must be a dictionary")

    provider = model_config.get('provider', '').lower()
    model_name = model_config.get('name')

    if not provider:
         raise ValueError("Model configuration must include a 'provider' key.")
    if not model_name:
        raise ValueError("Model configuration must include a 'name' key.")

    # Pass the whole model_config as kwargs
    params = model_config

    try:
        if provider == 'openai':
            # Use the unified client for all OpenAI models
            logger.info(f"Detected OpenAI provider. Using UnifiedOpenAIClient for model: {model_name}")
            return UnifiedOpenAIClient(model_name=model_name, **params)
        elif provider == 'anthropic':
            return AnthropicClient(model_name=model_name, **params)
        elif provider == 'google':
            return GoogleAIClient(model_name=model_name, **params)
        # --- Add DeepSeek --- Start ---
        elif provider == 'deepseek':
             logger.info(f"Detected DeepSeek provider. Using DeepSeekClient for model: {model_name}")
             return DeepSeekClient(model_name=model_name, **params)
        # --- Add DeepSeek --- End ---
        # --- Add Qwen --- Start ---
        elif provider == 'qwen':
             logger.info(f"Detected Qwen provider. Using QwenClient for model: {model_name}")
             return QwenClient(model_name=model_name, **params)
        # --- Add Qwen --- End ---
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except Exception as e:
        # Catch potential errors during client initialization (e.g., API key issues)
        logger.error(f"Failed to initialize LLM client for {provider}/{model_name}: {e}", exc_info=True)
        raise LLMError(f"Failed to initialize LLM client for {provider}/{model_name}: {e}")


if __name__ == "__main__":
    # Example usage when run directly
    import argparse
    from scripts.logger_setup import setup_logger
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Test LLM clients with a simple prompt")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google", "deepseek", "qwen"], help="LLM provider to test")
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