#!/usr/bin/env python3
"""
Update the config.yaml file to match the expected format for config_loader.py
"""

import yaml
import os

# New config content
config = {
    "openscad": {
        "executable_path": "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
        "minimum_version": "2021.01",
        "render_timeout_seconds": 120,
        "export_format": "asciistl",
        "backend": "Manifold",
        "summary_options": "all"
    },
    "directories": {
        "tasks": "./tasks",
        "reference": "./reference",
        "generated_outputs": "./generated_outputs",
        "results": "./results",
        "output": "./generated_outputs"  # Add output directory
    },
    "llm": {
        "models": [
            {
                "name": "gpt-4o-mini",
                "provider": "openai",
                "temperature": 0.2,
                "max_tokens": 1500
            },
            {
                "name": "claude-3-5-sonnet-20240620",
                "provider": "anthropic",
                "temperature": 0.2,
                "max_tokens": 1500
            },
            {
                "name": "gemini-1.5-pro-latest",
                "provider": "google",
                "temperature": 0.2,
                "max_tokens": 1500
            }
        ]
    },
    "geometry_check": {
        "bounding_box_tolerance_mm": 0.5,
        "icp_fitness_threshold": 1.0
    }
}

# Write the config to file
with open("config.yaml", "w") as f:
    # Add a header comment
    f.write("# CadEval Configuration\n")
    # Write the YAML content
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Updated config.yaml with new format") 