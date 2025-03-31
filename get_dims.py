import trimesh
import numpy as np # For formatting output
import argparse # Import argparse for command-line arguments
import os # To check if the file exists

# Set up argument parser
parser = argparse.ArgumentParser(description="Calculate the bounding box dimensions of an STL file.")
parser.add_argument("stl_path", type=str, help="Path to the input STL file.")

# Parse arguments
args = parser.parse_args()
stl_path = args.stl_path

try:
    # Check if the provided file path exists
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"File not found at the specified path: {stl_path}")

    mesh = trimesh.load_mesh(stl_path)

    # mesh.extents gives the dimensions [Length, Width, Height]
    dimensions = mesh.extents

    # Format for easy copying into YAML
    formatted_dims = np.round(dimensions, 4) # Round to reasonable precision
    print(f"STL Path: {stl_path}")
    print(f"Bounding Box [L, W, H]: [{formatted_dims[0]}, {formatted_dims[1]}, {formatted_dims[2]}]")

    # You can also print the raw bounds if needed
    # print(f"Raw Bounds (min_xyz, max_xyz): \n{mesh.bounds}")

except FileNotFoundError as e: # Catch specific FileNotFoundError
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error loading mesh: {e}")
    print(f"Please ensure the file at '{stl_path}' is a valid STL.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
