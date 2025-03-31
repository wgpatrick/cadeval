#!/usr/bin/env python3
"""
Unit tests for the generate_scad module.
"""

import os
import sys
import unittest
from unittest import mock

# Add the parent directory to the path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Mock the imports that require config.yaml
sys.modules['scripts.config_loader'] = mock.MagicMock()
sys.modules['scripts.llm_clients'] = mock.MagicMock()

# Now import the functions we want to test
from scripts.generate_scad import format_prompt, extract_scad_code

# Disable logging for tests to keep output clean
import logging
logging.disable(logging.CRITICAL)


class TestPromptFormatting(unittest.TestCase):
    """Test cases for prompt formatting."""
    
    def test_default_formatting(self):
        """Test prompt formatting with default parameters."""
        description = "Create a simple cube with side length 10mm."
        prompt = format_prompt(description)
        
        # Check that the prompt contains the expected components
        self.assertIn("Create a valid OpenSCAD script", prompt)
        self.assertIn(description, prompt)
        self.assertIn("IMPORTANT: Create only a SINGLE part/model", prompt)
        self.assertIn("Ensure the model is manifold and watertight", prompt)
    
    def test_custom_parameters(self):
        """Test prompt formatting with custom parameters."""
        description = "Create a cylinder with height 20mm and radius 5mm."
        custom_params = {
            "include_header": False,
            "reminder_single_part": True,
            "specific_reminders": False
        }
        
        prompt = format_prompt(description, custom_params)
        
        # Header should not be included
        self.assertNotIn("Create a valid OpenSCAD script", prompt)
        
        # Description should be included
        self.assertIn(description, prompt)
        
        # Single part reminder should be included
        self.assertIn("SINGLE part/model", prompt)
        
        # Specific reminders should not be included
        self.assertNotIn("manifold and watertight", prompt)
    
    def test_all_components_off(self):
        """Test prompt formatting with all optional components turned off."""
        description = "Create a sphere with radius 15mm."
        custom_params = {
            "include_header": False,
            "reminder_single_part": False,
            "specific_reminders": False
        }
        
        prompt = format_prompt(description, custom_params)
        
        # Only the description should be included
        self.assertEqual(prompt, description)


class TestCodeExtraction(unittest.TestCase):
    """Test cases for OpenSCAD code extraction from LLM responses."""
    
    def test_extract_from_markdown_code_block(self):
        """Test extracting code from a markdown code block."""
        response = """
Here's the OpenSCAD code for a simple cube:

```openscad
// Simple cube model
cube([10, 10, 10]);
```

I hope this helps!
"""
        code = extract_scad_code(response)
        self.assertEqual(code.strip(), "// Simple cube model\ncube([10, 10, 10]);")
    
    def test_extract_from_multiple_code_blocks(self):
        """Test extracting code when there are multiple code blocks."""
        response = """
Here's a partial solution:

```openscad
// First part
cube([10, 10, 10]);
```

And here's another part:

```
// Second part
sphere(r=5);
```

You can combine them like this:

```openscad
// Combined model
union() {
    cube([10, 10, 10]);
    translate([5, 5, 10]) sphere(r=5);
}
```
"""
        code = extract_scad_code(response)
        self.assertEqual(code.strip(), "// First part\ncube([10, 10, 10]);")
    
    def test_extract_from_unmarked_code_block(self):
        """Test extracting code from an unmarked code block."""
        response = """
Here's the OpenSCAD code:

```
// Simple model
cube([10, 10, 10]);
```
"""
        code = extract_scad_code(response)
        self.assertEqual(code.strip(), "// Simple model\ncube([10, 10, 10]);")
    
    def test_extract_from_raw_code(self):
        """Test extracting code when it's not in a code block."""
        response = """// Simple model
cube([10, 10, 10]);"""
        
        code = extract_scad_code(response)
        self.assertEqual(code.strip(), "// Simple model\ncube([10, 10, 10]);")


if __name__ == "__main__":
    unittest.main() 