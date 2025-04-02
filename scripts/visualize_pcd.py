#!/usr/bin/env python3
# visualize_pcd.py
import open3d as o3d
import sys
import os

def visualize_point_clouds(file_paths):
    """Loads and visualizes multiple .ply point cloud files."""
    if not file_paths:
        print("Usage: python scripts/visualize_pcd.py <path_to_ply_file1> [path_to_ply_file2] ...")
        return

    point_clouds = []
    colors = [
        [1, 0.7, 0],  # Orange
        [0, 0.65, 0.9], # Blue
        [0.9, 0.3, 0.3], # Red
        [0.5, 0.9, 0.5], # Green
        [0.8, 0.3, 0.8], # Purple
        [1, 1, 0],  # Yellow
    ]
    
    print("Loading point clouds:")
    for i, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"Warning: File not found - {path}")
            continue
        try:
            pcd = o3d.io.read_point_cloud(path)
            if not pcd.has_points():
                print(f"Warning: No points found in {path}")
                continue

            # Assign a distinct color
            color_idx = i % len(colors)
            pcd.paint_uniform_color(colors[color_idx])
            
            point_clouds.append(pcd)
            print(f" - Loaded {path} ({len(pcd.points)} points) - Color {colors[color_idx]}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not point_clouds:
        print("No valid point clouds loaded to visualize.")
        return

    print("\nLaunching Open3D visualizer...")
    print("Close the window to exit.")
    o3d.visualization.draw_geometries(point_clouds, window_name="Point Cloud Visualization")

if __name__ == "__main__":
    # Adjust usage message if running from project root
    if os.path.basename(sys.argv[0]) == "visualize_pcd.py":
        pass # Use default usage
    else: # Likely run via python -m scripts.visualize_pcd
        print("Usage: python -m scripts.visualize_pcd <path_to_ply_file1> ...")
    
    visualize_point_clouds(sys.argv[1:]) 