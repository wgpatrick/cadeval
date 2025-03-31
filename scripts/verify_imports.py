#!/usr/bin/env python3
"""
Script to verify that key packages are properly installed and can be imported.
This is part of the testing process for the CadEval project.
"""

import sys
import importlib

def check_import(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✅ Successfully imported {package_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {e}")
        return False

if __name__ == "__main__":
    # List of packages to check
    packages = [
        # LLM API clients
        "openai",
        "anthropic",
        "google.generativeai",
        
        # Geometry processing
        "trimesh",
        "numpy",
        "scipy",
        
        # Data handling & config
        "yaml",
        "jsonschema",
        
        # Utilities
        "dotenv",
        "tqdm",
        "click",
        "colorama"
    ]
    
    # Note about open3d
    print("⚠️ Note: 'open3d' is currently commented out in requirements.txt and will need to be installed separately.")
    
    # Check all imports
    failed = 0
    for package in packages:
        if not check_import(package):
            failed += 1
    
    # Summary
    total = len(packages)
    succeeded = total - failed
    print(f"\nSummary: {succeeded}/{total} packages imported successfully.")
    
    if failed == 0:
        print("🎉 All necessary imports are working correctly!")
        sys.exit(0)
    else:
        print(f"⚠️ {failed} package(s) failed to import. Please check your environment setup.")
        sys.exit(1) 