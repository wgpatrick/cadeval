#!/usr/bin/env python3
"""
List Available LLM Models

This script queries the APIs of configured LLM providers (OpenAI, Anthropic, Google)
to list available models that can be used for generation tasks.
"""

import os
import sys
import warnings

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Filter specific warnings from SDKs if they become noisy
warnings.filterwarnings("ignore", category=UserWarning, module='google.api_core.protobuf_helpers')

try:
    from scripts.logger_setup import get_logger, setup_logger
    from scripts.config_loader import get_config, ConfigError
    import openai
    from openai import OpenAI
    import anthropic
    import google.generativeai as genai
except ImportError as e:
    print(f"Error: Missing required libraries. Please ensure your environment is set up correctly. {e}")
    sys.exit(1)

# Initialize logger for this module
setup_logger(__name__, level="INFO")  # Set default level
logger = get_logger(__name__)


def list_openai_models(api_key: str):
    """Lists available OpenAI models."""
    print("--- OpenAI Models ---")
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        # Filter for models that might be suitable (e.g., contain 'gpt')
        # This is heuristic; official docs are better for capabilities
        gpt_models = sorted([m.id for m in models if 'gpt' in m.id.lower()])
        if gpt_models:
            for model_id in gpt_models:
                print(f"- {model_id}")
        else:
            print("No GPT models found or unable to filter.")
            print("Full list:", [m.id for m in models])
    except openai.AuthenticationError:
        print("  Error: Invalid OpenAI API key.")
    except Exception as e:
        print(f"  Error querying OpenAI models: {e}")
    print("")


def list_anthropic_models(api_key: str):
    """Lists known/recommended Anthropic models (API doesn't offer a list function)."""
    print("--- Anthropic Models ---")
    print("  Note: Anthropic API does not provide a dynamic model listing function.")
    print("  Listing known recommended models (check official docs for latest):")
    # List based on known models as of mid-2024
    known_models = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    # Basic check if the API key works (optional, might add cost/complexity)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Simple test - maybe list models from config if needed?
        # For now, just acknowledge the key seems present
        print(f"  (API Key for Anthropic found)")
        for model_id in known_models:
            print(f"- {model_id}")
    except anthropic.AuthenticationError:
         print("  Error: Invalid Anthropic API key.")
    except Exception as e:
        print(f"  Error during basic Anthropic client check: {e}")
    print("")


def list_google_models(api_key: str):
    """Lists available Google AI (Gemini) models."""
    print("--- Google AI (Gemini) Models ---")
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Filter for models supporting content generation
        generative_models = sorted([m.name for m in models if 'generateContent' in m.supported_generation_methods])
        if generative_models:
            # Further refine to common model names if desired
            preferred_order = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
            
            # Sort known models first, then others
            sorted_models = []
            remaining_models = []
            for model_name_full in generative_models:
                model_id = model_name_full.split('/')[-1] # Extract ID like 'gemini-1.5-pro'
                found = False
                for pref in preferred_order:
                    if pref in model_id:
                        sorted_models.append(model_id)
                        found = True
                        break
                if not found:
                    remaining_models.append(model_id)
            
            # Combine lists
            final_list = sorted_models + sorted(remaining_models)
            
            for model_id in final_list:
                 print(f"- {model_id}")
        else:
            print("No models supporting 'generateContent' found.")
    except Exception as e:
        # Catching broad exception as Google SDK errors can vary
        print(f"  Error querying Google AI models: {e}")
        if "API key not valid" in str(e):
             print("  Hint: Check if the Google AI API key is correct.")
    print("")


if __name__ == "__main__":
    print("Querying LLM providers for available models...")
    print("(Requires valid API keys in .env file)")
    print("="*40)
    
    try:
        config = get_config()
    except ConfigError as e:
        logger.error(f"Failed to load configuration: {e}")
        print(f"Error: Could not load configuration. {e}")
        sys.exit(1)
    
    # Check OpenAI
    try:
        openai_key = config.get_api_key('openai')
        list_openai_models(openai_key)
    except ConfigError:
        print("--- OpenAI Models ---")
        print("  Skipped: OpenAI API key not configured in .env file.")
    except Exception as e:
        print(f"An unexpected error occurred with OpenAI: {e}")

    # Check Anthropic
    try:
        anthropic_key = config.get_api_key('anthropic')
        list_anthropic_models(anthropic_key)
    except ConfigError:
        print("--- Anthropic Models ---")
        print("  Skipped: Anthropic API key not configured in .env file.")
    except Exception as e:
         print(f"An unexpected error occurred with Anthropic: {e}")

    # Check Google
    try:
        google_key = config.get_api_key('google')
        list_google_models(google_key)
    except ConfigError:
        print("--- Google AI (Gemini) Models ---")
        print("  Skipped: Google AI API key not configured in .env file.")
    except Exception as e:
         print(f"An unexpected error occurred with Google: {e}")

    print("\n" + "="*40)
    print("Model listing complete.") 