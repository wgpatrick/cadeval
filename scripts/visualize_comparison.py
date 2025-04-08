#!/usr/bin/env python3
"""
Rewritten script to visualize reference and generated geometry as aligned
point clouds using Open3D. Uses alignment logic from geometry_check.py.
"""

import open3d as o3d
import numpy as np
import argparse
import os
import sys
import copy
import logging

# --- Setup Logger ---
# Using basicConfig for simplicity in a standalone script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function for Preprocessing (Downsample + FPFH) ---
# Copied directly from geometry_check.py (ensure this is kept in sync if geometry_check updates)
def preprocess(pcd, voxel_size, logger):
    """Preprocesses point cloud: downsamples, estimates normals, computes FPFH."""
    logger.debug(f"    Preprocessing: Downsampling with voxel size {voxel_size}...")
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
        if not pcd_down.has_points():
             logger.warning("    Preprocessing: Downsampling resulted in zero points.")
             return None, None
        logger.debug(f"    Preprocessing: Estimating normals...")
        # Using params from geometry_check.py
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=35))
        if not pcd_down.has_normals():
             logger.warning("    Preprocessing: Normal estimation failed.")
             # Attempt FPFH anyway
        logger.debug(f"    Preprocessing: Computing FPFH features...")
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                  pcd_down,
                  o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        if fpfh is None or not hasattr(fpfh, 'data') or fpfh.data.shape[1] == 0:
             logger.warning("    Preprocessing: FPFH computation failed or produced empty features.")
             return pcd_down, None # Return downsampled cloud even if features fail
        logger.debug(f"    Preprocessing: Done.")
        return pcd_down, fpfh
    except Exception as e:
        logger.error(f"    Preprocessing: Error during preprocessing: {e}", exc_info=True)
        return None, None

# --- Core Alignment Function ---
# Adapted directly from geometry_check.py (check_similarity function logic)
def load_and_align(ref_path, gen_path, n_points=50000):
    """
    Loads meshes, samples points, aligns using Global Reg + ICP from geometry_check.
    Returns original reference PCD and final aligned generated PCD.
    """
    logger.info(f"--- Starting Alignment Process ---")
    logger.info(f"Reference: {ref_path}")
    logger.info(f"Generated: {gen_path}")

    if not os.path.exists(ref_path):
        logger.error(f"Reference file not found: {ref_path}")
        return None, None, None, None, "Reference file not found."
    if not os.path.exists(gen_path):
        logger.error(f"Generated file not found: {gen_path}")
        return None, None, None, None, "Generated file not found."

    # --- Load Meshes ---
    try:
        logger.info("Loading meshes...")
        ref_mesh = o3d.io.read_triangle_mesh(ref_path)
        gen_mesh = o3d.io.read_triangle_mesh(gen_path)
        if not ref_mesh.has_triangles():
            return None, None, None, None, "Reference mesh is empty or invalid."
        if not gen_mesh.has_triangles():
            # Log warning but try to proceed, maybe sampling still works if minor issue
            logger.warning("Generated mesh loaded with no triangles.")
            # return None, None, None, None, "Generated mesh is empty or invalid."

        logger.info("Meshes loaded.")

        # Ensure normals are computed for sampling
        if not gen_mesh.has_vertex_normals(): gen_mesh.compute_vertex_normals()
        if not ref_mesh.has_vertex_normals(): ref_mesh.compute_vertex_normals()

    except Exception as e:
        logger.error(f"Error loading meshes: {e}", exc_info=True)
        return None, None, None, None, f"Error loading meshes: {e}"

    # --- Sample Point Clouds ---
    try:
        logger.info(f"Sampling {n_points} points from each mesh...")
        ref_pcd = ref_mesh.sample_points_uniformly(number_of_points=n_points)
        gen_pcd = gen_mesh.sample_points_uniformly(number_of_points=n_points)

        min_actual_points = 100
        actual_gen_points = len(gen_pcd.points)
        actual_ref_points = len(ref_pcd.points)
        logger.info(f"Actual points sampled: Gen={actual_gen_points}, Ref={actual_ref_points}")
        if actual_gen_points < min_actual_points or actual_ref_points < min_actual_points:
            logger.warning(f"Sampling produced very few points. Alignment might be inaccurate.")
            if actual_gen_points == 0 or actual_ref_points == 0:
                 return ref_pcd, gen_pcd, None, None, "Sampling produced zero points." # Return original PCDs

    except Exception as e:
         logger.error(f"Error sampling point clouds: {e}", exc_info=True)
         # Return None for aligned, maybe return original pcds if they exist?
         return None, None, None, None, f"Error sampling point clouds: {e}"

    # --- Global Alignment (FPFH + RANSAC) ---
    try:
        voxel_size = 5.0 # WP Changed from 5.0 to match geometry_check.py # User request to change back to 5.0
        logger.info(f"Starting Global Registration Preprocessing (Voxel Size: {voxel_size})...")
        gen_pcd_down, fpfh_gen = preprocess(gen_pcd, voxel_size, logger)
        ref_pcd_down, fpfh_ref = preprocess(ref_pcd, voxel_size, logger)

        if gen_pcd_down is None or fpfh_gen is None or ref_pcd_down is None or fpfh_ref is None:
            error_msg = "Preprocessing for RANSAC failed (check logs for details), skipping alignment."
            logger.error(error_msg)
            return ref_pcd, gen_pcd, None, None, error_msg # Return original PCDs

        distance_thresh = voxel_size * 1.5
        logger.info(f"Running RANSAC Global Registration (Distance Threshold: {distance_thresh})...")
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=gen_pcd_down, target=ref_pcd_down, # Flipped source/target to match geometry_check
            source_feature=fpfh_gen, target_feature=fpfh_ref,
            mutual_filter=True,
            max_correspondence_distance=distance_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        T_ransac = ransac_result.transformation
        logger.info(f"Global Registration RANSAC Fitness: {ransac_result.fitness:.6f}")
        logger.info(f"Global Registration RANSAC Inlier RMSE: {ransac_result.inlier_rmse:.6f}")

        gen_pcd_globally_aligned = copy.deepcopy(gen_pcd).transform(T_ransac)

    except Exception as e:
         logger.error(f"Error during global registration: {e}", exc_info=True)
         return ref_pcd, gen_pcd, None, None, f"Error during global registration: {e}" # Return original

    # --- ICP Refinement ---
    try:
        icp_refinement_threshold = 1.5 # Consistent with geometry_check.py
        logger.info(f"Performing ICP Refinement (Max Corr Dist: {icp_refinement_threshold})...")

        icp_result = o3d.pipelines.registration.registration_icp(
            source=gen_pcd_globally_aligned,
            target=ref_pcd,
            max_correspondence_distance=icp_refinement_threshold,
            init=np.identity(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        T_icp_refine = icp_result.transformation
        icp_refinement_fitness = icp_result.fitness
        icp_refinement_rmse = icp_result.inlier_rmse
        logger.info(f"ICP Refinement Fitness: {icp_refinement_fitness:.6f}, ICP Refinement Inlier RMSE: {icp_refinement_rmse:.6f}")

        gen_pcd_final_aligned = gen_pcd_globally_aligned.transform(T_icp_refine)

        logger.info("--- Alignment Process Completed Successfully ---")
        return ref_pcd, gen_pcd_final_aligned, icp_refinement_fitness, icp_refinement_rmse, None

    except Exception as e:
         logger.error(f"Error during ICP refinement: {e}", exc_info=True)
         # Return globally aligned if ICP failed? Or original? Let's return globally aligned for inspection
         return ref_pcd, gen_pcd_globally_aligned, None, None, f"Error during ICP refinement: {e}"


# --- Visualization Function ---
def visualize(pcd_list, title):
    """Displays a list of point clouds in an Open3D window."""
    if not pcd_list:
        logger.error("No point clouds provided for visualization.")
        return

    # Assign colors (simple alternating colors for now)
    colors = [[0.0, 0.6, 0.2], [0.8, 0.2, 0.2], [0.2, 0.2, 0.8]] # Green, Red, Blue
    colored_pcds = []
    for i, pcd in enumerate(pcd_list):
        if pcd is not None and pcd.has_points():
            pcd_copy = copy.deepcopy(pcd)
            pcd_copy.paint_uniform_color(colors[i % len(colors)])
            colored_pcds.append(pcd_copy)
        else:
            logger.warning(f"Skipping visualization of point cloud at index {i} because it's None or empty.")

    if not colored_pcds:
         logger.error("No valid point clouds left to visualize.")
         return

    logger.info(f"Displaying {len(colored_pcds)} point clouds... Close the window to exit.")
    o3d.visualization.draw_geometries(
        colored_pcds,
        window_name=title,
        width=1024,
        height=768
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, align (using geometry_check method), and visualize reference vs generated STLs.")
    parser.add_argument("--ref", required=True, help="Path to the reference STL file.")
    parser.add_argument("--gen", required=True, help="Path to the generated STL file.")
    parser.add_argument("--title", default="Aligned Comparison (Ref=Green, Gen=Red)", help="Optional title for the visualization window.")
    parser.add_argument("--points", type=int, default=50000, help="Number of points to sample from each mesh (default: 50000).")

    args = parser.parse_args()

    # Use absolute paths to avoid ambiguity
    ref_abs_path = os.path.abspath(args.ref)
    gen_abs_path = os.path.abspath(args.gen)

    # Load and align
    ref_pcd, gen_pcd_aligned, fitness, rmse, error_msg = load_and_align(ref_abs_path, gen_abs_path, args.points)

    # Visualize results
    if error_msg:
        logger.error(f"Alignment failed: {error_msg}")
        # Optionally visualize the original point clouds if they exist
        if ref_pcd and gen_pcd_aligned: # gen_pcd_aligned might be original or globally aligned on error
             logger.info("Visualizing point clouds *before* error occurred...")
             visualize([ref_pcd, gen_pcd_aligned], title=f"{args.title} (Alignment Error)")
        elif ref_pcd:
             logger.info("Visualizing only reference point cloud...")
             visualize([ref_pcd], title=f"{args.title} (Alignment Error - Gen Failed)")
        else:
             logger.error("Cannot visualize, reference point cloud failed to load/sample.")

    elif ref_pcd and gen_pcd_aligned:
        logger.info(f"Alignment successful. Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")
        visualize([ref_pcd, gen_pcd_aligned], title=args.title)
    else:
        logger.error("Alignment finished without error message, but point clouds are missing.")

    logger.info("Visualization script finished.") 