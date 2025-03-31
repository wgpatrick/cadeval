#!/usr/bin/env python3
"""
Test script for LLM clients.

This script tests the LLM clients by sending a simple prompt to each configured LLM
and verifying that a response is received.
"""

import os
import sys
import unittest
from typing import Dict, Any

# Add the parent directory to the path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.llm_clients import create_llm_client, LLMError
from scripts.config_loader import get_config
from scripts.logger_setup import setup_logger

# Set up logging
setup_logger(__name__, level="INFO")


class TestLLMClients(unittest.TestCase):
    """Test cases for LLM clients."""

    def setUp(self):
        """Set up test environment."""
        self.config = get_config()
        self.test_prompt = "Write a single sentence haiku about 3D printing."
    
    def test_openai_client(self):
        """Test OpenAI client."""
        self._test_provider("openai")
    
    def test_anthropic_client(self):
        """Test Anthropic client."""
        self._test_provider("anthropic")
    
    def test_google_client(self):
        """Test Google client."""
        self._test_provider("google")
    
    def _test_provider(self, provider: str):
        """Helper method to test a specific provider."""
        # Find a model config for the given provider
        models = self.config.get('llm.models', [])
        model_config = None
        
        for config in models:
            if config.get('provider', '').lower() == provider.lower():
                model_config = config
                break
        
        if not model_config:
            self.skipTest(f"No {provider} model configured in config.yaml")
        
        try:
            # Create client and generate text
            client = create_llm_client(model_config)
            response = client.generate_text(self.test_prompt)
            
            # Basic verification
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)
            
            print(f"\n{provider.capitalize()} response: {response}\n")
            
        except LLMError as e:
            self.fail(f"LLM client error: {e}")
        except Exception as e:
            self.fail(f"Unexpected error: {e}")


if __name__ == "__main__":
    unittest.main() 