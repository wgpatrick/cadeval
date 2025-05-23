#!/usr/bin/env python3
"""
Geometry Check Script for CadEval

This module implements automated checks on generated STL files, comparing them
against reference models and task requirements.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
from tenacity import retry, stop_after_attempt, wait_fixed
import tempfile # For temporary directory
import copy # Add copy import

# Attempt imports for geometry libraries, provide guidance if missing
try:
    import trimesh
except ImportError:
    print("Error: Trimesh library not found. Please install using: pip install trimesh")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D library not found. Please ensure it's installed in your environment.")
    sys.exit(1)

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.logger_setup import get_logger, setup_logger
from scripts.config_loader import get_config, Config, ConfigError

# Initialize logger for this module
logger = get_logger(__name__)

# Define DEFAULT_CONFIG with default values for checks nested under 'geometry_check'
DEFAULT_CONFIG = {
    'geometry_check': {
        'bounding_box_tolerance_mm': 1.0, # Default tolerance in mm
        'similarity_threshold_mm': 1.0, # Default Chamfer distance threshold in mm
    }
}

# Environment variable to control debug output
SAVE_PCD_FOR_DEBUG = os.environ.get('SAVE_PCD_FOR_DEBUG', 'false').lower() == 'true'
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEBUG_PCD_DIR = os.path.join(PROJECT_ROOT, "debug_output", "similarity_pcd")

if SAVE_PCD_FOR_DEBUG:
    os.makedirs(DEBUG_PCD_DIR, exist_ok=True)
    print(f"--- DEBUG: Saving intermediate PCD files to {DEBUG_PCD_DIR} ---")

class GeometryCheckError(Exception):
    """Exception raised for errors during geometry checks."""
    pass

# --- Check Functions ---

def check_render_success(render_status: str) -> bool:
    """Check 1: Check if the rendering step was successful."""
    return render_status == "Success"

def clean_mesh_for_checks(mesh: o3d.geometry.TriangleMesh, logger: logging.Logger, filename: str) -> o3d.geometry.TriangleMesh:
    """Applies cleaning steps relevant for watertight/component checks."""
    try:
        initial_vertices = len(mesh.vertices)
        initial_triangles = len(mesh.triangles)
        mesh.merge_close_vertices(0.0001)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        final_vertices = len(mesh.vertices)
        final_triangles = len(mesh.triangles)
        if initial_vertices != final_vertices or initial_triangles != final_triangles:
            logger.debug(f"Cleaned mesh {filename}: Vertices {initial_vertices}->{final_vertices}, Triangles {initial_triangles}->{final_triangles}")
        return mesh
    except Exception as e:
        logger.warning(f"Error during mesh cleaning for {filename}: {e}")
        return mesh

def check_single_component(
    mesh_file: str,
    requirements: Dict[str, Any],
    logger: logging.Logger
) -> tuple[bool, str | None]:
    """Check 3: Check if the mesh consists of the expected number of connected components."""
    if not os.path.exists(mesh_file):
        error_msg = f"Mesh file not found: {mesh_file}"
        logger.error(error_msg)
        return False, error_msg

    expected_components = 1
    try:
        if 'topology_requirements' in requirements and \
           isinstance(requirements['topology_requirements'], dict) and \
           'expected_component_count' in requirements['topology_requirements']:
            expected_components = int(requirements['topology_requirements']['expected_component_count'])
            logger.debug(f"Expecting {expected_components} component(s) based on requirements.")
        else:
            logger.debug("No expected component count in requirements, defaulting to 1.")
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid format for expected_component_count in requirements ({e}), defaulting to 1.")
        expected_components = 1

    try:
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if not mesh.has_triangles():
            error_msg = f"Mesh file {mesh_file} is empty or could not be loaded."
            logger.warning(error_msg)
            return False, error_msg

        mesh = clean_mesh_for_checks(mesh, logger, os.path.basename(mesh_file))
        if not mesh.has_triangles():
             error_msg = f"Mesh {mesh_file} became empty after cleaning."
             logger.warning(error_msg)
             return False, error_msg

        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        num_components = len(cluster_n_triangles)
        logger.debug(f"File: {mesh_file} (Post-Clean), Found components: {num_components}")

        passes_check = num_components == expected_components

        if not passes_check:
            error_msg = f"Found {num_components} components, expected {expected_components}"
            logger.info(f"Component count check for {mesh_file}: FAILED ({error_msg})")
            return False, error_msg
        else:
            logger.info(f"Component count check for {mesh_file}: PASSED (Found {num_components}, Expected {expected_components})")
            return True, None
    except Exception as e:
        error_msg = f"Error checking single component for {mesh_file}: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def check_watertight(mesh_file: str, logger: logging.Logger) -> tuple[bool, str | None]:
    """Check 2: Check if the mesh is watertight (manifold)."""
    if not os.path.exists(mesh_file):
        error_msg = f"Mesh file not found: {mesh_file}"
        logger.error(error_msg)
        return False, error_msg

    try:
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if not mesh.has_triangles():
            error_msg = f"Mesh file {mesh_file} is empty or could not be loaded."
            logger.warning(error_msg)
            return False, error_msg

        mesh = clean_mesh_for_checks(mesh, logger, os.path.basename(mesh_file))
        if not mesh.has_triangles():
             error_msg = f"Mesh {mesh_file} became empty after cleaning."
             logger.warning(error_msg)
             return False, error_msg

        logger.debug(f"Checking manifold status for {mesh_file} (Post-Clean)...")
        is_manifold = mesh.is_watertight() # Open3D's watertight check primarily looks for non-manifold edges
        logger.debug(f"File: {mesh_file} (Post-Clean), o3d.is_watertight() result: {is_manifold}")

        if not is_manifold:
            try:
                # Attempt to get more info, but don't fail the check solely based on this
                edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
                logger.warning(f"Found {len(edges)} non-manifold edges in {mesh_file} (Post-Clean).")
            except Exception as edge_ex:
                 logger.warning(f"Could not get non-manifold edge info for {mesh_file} (Post-Clean): {edge_ex}")

            error_msg = "Mesh is not manifold (non-watertight edges found)"
            logger.info(f"Watertight check for {mesh_file}: FAILED ({error_msg})")
            return False, error_msg
        else:
            # Note: Open3D's is_watertight doesn't guarantee single component. That's handled by check_single_component.
            logger.info(f"Watertight (manifold edge) check for {mesh_file}: PASSED")
            return True, None
    except Exception as e:
        error_msg = f"Error checking watertightness for {mesh_file}: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

# --- Helper Function for Preprocessing (Downsample + FPFH) ---
# Copied from visualize_comparison.py
def preprocess(pcd, voxel_size, logger):
    """Preprocesses point cloud: downsamples, estimates normals, computes FPFH."""
    logger.debug(f"    Preprocessing: Downsampling with voxel size {voxel_size}...")
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
        if not pcd_down.has_points():
             logger.warning("    Preprocessing: Downsampling resulted in zero points.")
             return None, None
        logger.debug(f"    Preprocessing: Estimating normals...")
        # Increased radius/nn slightly for potentially sparse downsampled clouds
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=35))
        if not pcd_down.has_normals():
             logger.warning("    Preprocessing: Normal estimation failed.")
             # Attempt FPFH anyway, might work depending on Open3D version/implementation
             # return None, None # Stricter: Fail if normals fail
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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def check_similarity(
    generated_stl_path: str,
    reference_stl_path: str,
    threshold: float,
    logger: logging.Logger
) -> tuple[float | None, float | None, np.ndarray | None, float | None, float | None, str | None]:
    """
    Check 5: Performs alignment using Global Reg (FPFH+RANSAC) + ICP Refinement
    and calculates Chamfer distance, 95th and 99th percentile Hausdorff distances.
    Returns (chamfer_distance, icp_fitness, transformation_matrix, hausdorff_95p, hausdorff_99p, error_msg).
    """
    if not os.path.exists(generated_stl_path): return None, None, None, None, None, f"Generated file not found: {generated_stl_path}"
    if not os.path.exists(reference_stl_path): return None, None, None, None, None, f"Reference file not found: {reference_stl_path}"

    if os.path.samefile(generated_stl_path, reference_stl_path):
        logger.debug("Files are identical (same path), returning zero distance, perfect fitness, identity transform")
        identity_transform = np.identity(4)
        return 0.0, 1.0, identity_transform, 0.0, 0.0, None

    final_transformation_matrix = None # Initialize the final matrix
    icp_refinement_fitness = None # Fitness score specifically from ICP refinement
    hausdorff_95p = None # Initialize Hausdorff 95p
    hausdorff_99p = None # Initialize Hausdorff 99p

    try:
        run_id = f"{os.path.basename(generated_stl_path)}_vs_{os.path.basename(reference_stl_path)}".replace('.stl', '')
        logger.debug(f"Loading meshes for similarity check: {generated_stl_path} vs {reference_stl_path}")
        generated_mesh = o3d.io.read_triangle_mesh(generated_stl_path)
        reference_mesh = o3d.io.read_triangle_mesh(reference_stl_path)
        if not generated_mesh.has_triangles(): return None, None, None, None, None, "Generated mesh has no triangles"
        if not reference_mesh.has_triangles(): return None, None, None, None, None, "Reference mesh has no triangles"

        # Ensure normals are computed for sampling/potential later use
        if not generated_mesh.has_vertex_normals(): generated_mesh.compute_vertex_normals()
        if not reference_mesh.has_vertex_normals(): reference_mesh.compute_vertex_normals()

        if SAVE_PCD_FOR_DEBUG:
            o3d.io.write_triangle_mesh(os.path.join(DEBUG_PCD_DIR, f"{run_id}_ref_mesh.ply"), reference_mesh)
            o3d.io.write_triangle_mesh(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_mesh.ply"), generated_mesh)

        N_POINTS_TARGET = 50000 # Target point count for sampling
        logger.debug(f"Targeting {N_POINTS_TARGET} points for uniform sampling.")
        if not generated_mesh.has_triangles() or not reference_mesh.has_triangles():
            return None, None, None, None, None, "Cannot sample points from empty mesh"

        logger.debug(f"Sampling up to {N_POINTS_TARGET} points from each mesh...")
        gen_pcd = generated_mesh.sample_points_uniformly(number_of_points=N_POINTS_TARGET)
        ref_pcd = reference_mesh.sample_points_uniformly(number_of_points=N_POINTS_TARGET)

        min_actual_points = 100
        actual_gen_points = len(gen_pcd.points)
        actual_ref_points = len(ref_pcd.points)
        logger.info(f"Actual points sampled: Gen={actual_gen_points}, Ref={actual_ref_points}")
        if actual_gen_points < min_actual_points or actual_ref_points < min_actual_points:
            warning_msg = (f"Sampling produced very few points (Gen: {actual_gen_points}, Ref: {actual_ref_points}). "
                           f"Similarity check might be inaccurate.")
            logger.warning(warning_msg)
            if actual_gen_points == 0 or actual_ref_points == 0:
                 return None, None, None, None, None, "Sampling produced zero points."

        if SAVE_PCD_FOR_DEBUG:
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_ref_sampled.ply"), ref_pcd)
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_sampled.ply"), gen_pcd)

        # --- Global Alignment (FPFH + RANSAC) ---
        # NOTE: voxel_size is critical and scale-dependent. Needs tuning. 5.0 is a starting guess.
        voxel_size = 5.0 # WP Reduced from 5.0 # User request to change back to 5.0
        logger.debug(f"Starting Global Registration Preprocessing (Voxel Size: {voxel_size})...")
        gen_pcd_down, fpfh_gen = preprocess(gen_pcd, voxel_size, logger)
        ref_pcd_down, fpfh_ref = preprocess(ref_pcd, voxel_size, logger)

        # Check if preprocessing was successful enough
        if gen_pcd_down is None or fpfh_gen is None or ref_pcd_down is None or fpfh_ref is None:
            error_msg = "Preprocessing for RANSAC failed (check logs for details), skipping alignment."
            logger.error(error_msg)
            # Cannot proceed with alignment, but maybe can calculate distance on unaligned clouds?
            # For now, return error. Consider fallback later if needed.
            return None, None, None, None, None, error_msg

        distance_thresh = voxel_size * 1.5 # RANSAC correspondence threshold (updated based on new voxel_size)
        logger.debug(f"Running RANSAC Global Registration (Distance Threshold: {distance_thresh})...")
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=gen_pcd_down, target=ref_pcd_down,
            source_feature=fpfh_gen, target_feature=fpfh_ref,
            mutual_filter=True,
            max_correspondence_distance=distance_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), # Scale=False
            ransac_n=4, # Typically 3 or 4
            checkers=[], # Optional geometric checks can be added
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) # Iterations, Confidence
        )
        T_ransac = ransac_result.transformation # Initial transformation from RANSAC
        logger.info(f"Global Registration RANSAC Fitness: {ransac_result.fitness:.6f}")
        logger.info(f"Global Registration RANSAC Inlier RMSE: {ransac_result.inlier_rmse:.6f}")

        # Apply initial transform to the *original resolution* generated point cloud
        # Use deepcopy to avoid modifying the original gen_pcd
        gen_pcd_globally_aligned = copy.deepcopy(gen_pcd).transform(T_ransac)

        if SAVE_PCD_FOR_DEBUG:
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_ransac_aligned.ply"), gen_pcd_globally_aligned)

        # --- ICP Refinement ---
        icp_refinement_threshold = 1.5 # Threshold for *refinement* stage, can be smaller than RANSAC's
        logger.debug(f"Performing ICP Refinement (Max Corr Dist: {icp_refinement_threshold}, Max Iter: 200)...")

        # Refine the alignment between the globally aligned gen cloud and the original ref cloud
        icp_result = o3d.pipelines.registration.registration_icp(
            source=gen_pcd_globally_aligned, # Source = globally aligned cloud
            target=ref_pcd, # Target = original reference cloud
            max_correspondence_distance=icp_refinement_threshold,
            init=np.identity(4), # Initial transform for refinement is Identity (already globally aligned)
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        T_icp_refine = icp_result.transformation # Transformation from the ICP refinement step
        icp_refinement_fitness = icp_result.fitness # Capture fitness from this step
        logger.info(f"ICP Refinement Fitness: {icp_refinement_fitness:.6f}, ICP Refinement Inlier RMSE: {icp_result.inlier_rmse:.6f}")

        # Calculate the final combined transformation matrix
        # This matrix transforms the *original* gen_pcd to the *final* aligned position
        final_transformation_matrix = T_icp_refine @ T_ransac

        # Apply the *final* transformation to the *original* generated point cloud for distance calculation
        gen_pcd_final_aligned = copy.deepcopy(gen_pcd).transform(final_transformation_matrix)

        if SAVE_PCD_FOR_DEBUG:
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_final_aligned.ply"), gen_pcd_final_aligned)

        # Calculate distances for Chamfer and Hausdorff
        logger.debug("Calculating point cloud distances for Chamfer/Hausdorff...")
        dist_gen_to_ref = np.asarray(gen_pcd_final_aligned.compute_point_cloud_distance(ref_pcd))
        dist_ref_to_gen = np.asarray(ref_pcd.compute_point_cloud_distance(gen_pcd_final_aligned))

        # Handle potential empty distance arrays
        if dist_gen_to_ref.size == 0 or dist_ref_to_gen.size == 0:
             logger.warning("Distance calculation resulted in empty arrays. Cannot compute metrics.")
             chamfer_distance = np.inf
             hausdorff_95p = np.inf # Set 95p to inf
             hausdorff_99p = np.inf
        else:
             # Calculate Chamfer Distance (Average of Means)
             chamfer_distance = (np.mean(dist_gen_to_ref) + np.mean(dist_ref_to_gen)) / 2.0
             logger.info(f"Similarity check for {os.path.basename(generated_stl_path)}: Final Chamfer distance = {chamfer_distance:.6f} mm")

             # Calculate Hausdorff Distances (95th and 99th Percentile)
             all_distances = np.concatenate((dist_gen_to_ref, dist_ref_to_gen))
             hausdorff_95p = np.percentile(all_distances, 95) # Calculate 95p
             hausdorff_99p = np.percentile(all_distances, 99)
             logger.info(f"Similarity check for {os.path.basename(generated_stl_path)}: 95th Percentile Hausdorff = {hausdorff_95p:.6f} mm")
             logger.info(f"Similarity check for {os.path.basename(generated_stl_path)}: 99th Percentile Hausdorff = {hausdorff_99p:.6f} mm")

        logger.info(f"ICP Refinement Fitness = {icp_refinement_fitness:.6f}") # Log fitness separately

        error_msg = None
        # Note: error_msg now only relates to Chamfer threshold, Hausdorff pass/fail handled in perform_geometry_checks
        if chamfer_distance > threshold:
             error_msg = f"Chamfer distance {chamfer_distance:.4f} mm exceeds threshold {threshold} mm"
             logger.info(f"SimilarityCheck Note: {error_msg}") # Log exceedance as info

        # Return chamfer dist, ICP fitness, final transform, hausdorff 95p, hausdorff 99p, and potential threshold error msg
        return chamfer_distance, icp_refinement_fitness, final_transformation_matrix, hausdorff_95p, hausdorff_99p, error_msg

    except Exception as e:
        error_msg = f"Error during similarity check between {generated_stl_path} and {reference_stl_path}: {e}"
        logger.error(error_msg, exc_info=True)
        # Return None for metrics and the error message
        return None, None, None, None, None, error_msg # Return None for both 95p and 99p

def compare_aligned_bounding_boxes(
    generated_stl_path: str,
    reference_stl_path: str,
    final_transform: np.ndarray, # Renamed parameter to reflect it's the final one
    geometry_config: Dict[str, Any], # Expect geometry check config dictionary
    logger: logging.Logger
) -> Tuple[Optional[bool], Optional[List[float]], Optional[List[float]], Optional[str]]:
    """
    Check 4: Compares the bounding box of the fully aligned generated mesh against the reference mesh.
    Uses the final transformation matrix from the similarity check (RANSAC + ICP).
    Returns (is_accurate, ref_dims_sorted, aligned_gen_dims_sorted, error_message)
    """
    error_prefix = "AlignedBBoxCheck:"
    ref_dims_sorted = None
    aligned_gen_dims_sorted = None
    try:
        # --- Get tolerance from geometry_config dictionary --- 
        tolerance = float(geometry_config.get('bounding_box_tolerance_mm', 1.0))
        # ------------------------------------------------------
        logger.debug(f"  Tolerance for aligned bbox check: +/- {tolerance} mm")

        # Load reference mesh and get its sorted dimensions
        ref_mesh = trimesh.load(reference_stl_path, force='mesh')
        if not ref_mesh.vertices.size > 0: raise ValueError("Reference mesh empty")
        ref_dims_sorted = sorted(ref_mesh.bounding_box.extents)
        logger.debug(f"    Reference BBox Dims (Sorted): {[f'{d:.4f}' for d in ref_dims_sorted]}")

        # Load generated mesh
        gen_mesh = trimesh.load(generated_stl_path, force='mesh')
        if not gen_mesh.vertices.size > 0: raise ValueError("Generated mesh empty")

        # Apply the *final* combined transform and get its sorted dimensions
        gen_mesh.apply_transform(final_transform) # Use the final transform
        aligned_gen_dims_sorted = sorted(gen_mesh.bounding_box.extents)
        logger.debug(f"    Final Aligned Gen BBox Dims (Sorted): {[f'{d:.4f}' for d in aligned_gen_dims_sorted]}")

        # Compare dimensions (Reference vs Final Aligned Generated)
        is_accurate = True
        diffs = np.abs(np.array(aligned_gen_dims_sorted) - np.array(ref_dims_sorted))
        for i in range(3):
            if diffs[i] > tolerance:
                is_accurate = False
                logger.debug(f"    Dimension mismatch (Ref vs Final Aligned): ref={ref_dims_sorted[i]:.4f}, aligned_gen={aligned_gen_dims_sorted[i]:.4f}, diff={diffs[i]:.4f}, tol={tolerance}")
                break # No need to check further dims

        logger.debug(f"  Aligned bounding box accuracy result: {is_accurate}")
        if not is_accurate:
            error_msg = f"{error_prefix} Final Aligned vs Ref BBox Dims Diff (Sorted): {[f'{d:.4f}' for d in diffs.tolist()]} exceeds tolerance {tolerance}"
            return is_accurate, ref_dims_sorted, aligned_gen_dims_sorted, error_msg
        else:
            return is_accurate, ref_dims_sorted, aligned_gen_dims_sorted, None

    except ConfigError as e:
         # Specific error for missing tolerance config
         return None, ref_dims_sorted, aligned_gen_dims_sorted, f"{error_prefix} Missing/Invalid 'geometry_check.bounding_box_tolerance_mm' config: {e}"
    except FileNotFoundError as e:
        return None, ref_dims_sorted, aligned_gen_dims_sorted, f"{error_prefix} Mesh file not found: {e}"
    except ValueError as e: # Catch mesh loading/empty errors
         return None, ref_dims_sorted, aligned_gen_dims_sorted, f"{error_prefix} Error loading/processing mesh for bbox check: {e}"
    except Exception as e:
        logger.error(f"    Unhandled error during aligned bbox comparison: {e}", exc_info=True)
        return None, ref_dims_sorted, aligned_gen_dims_sorted, f"{error_prefix} Unhandled Exception: {e}"

# --- New Volume Check Function --- Start ---
def check_volume(
    generated_stl_path: str,
    reference_stl_path: str,
    volume_threshold_percent: float = 1.0,
    logger: logging.Logger = logger # Use module logger by default
) -> Tuple[Optional[bool], Optional[float], Optional[float], Optional[str]]:
    """
    Check 6: Compares the volume of the generated mesh against the reference mesh.
    Returns (is_within_threshold, reference_volume, generated_volume, error_message)
    """
    error_msg = None
    gen_vol, ref_vol = None, None
    passed = False # Default to False

    try:
        gen_mesh = trimesh.load(generated_stl_path, force='mesh')
        gen_vol = gen_mesh.volume
    except Exception as e:
        error_msg = f"Failed to load generated STL for volume check: {e}"
        logger.error(error_msg)
        # Return False, None volumes, and the error message
        return False, None, None, error_msg 

    try:
        ref_mesh = trimesh.load(reference_stl_path, force='mesh')
        ref_vol = ref_mesh.volume
    except Exception as e:
        error_msg = f"Failed to load reference STL for volume check: {e}"
        logger.error(error_msg)
        # Return False, potentially valid gen_vol, None ref_vol, and error
        return False, None, gen_vol, error_msg 

    # Check for watertightness (required for accurate volume)
    if not gen_mesh.is_watertight:
        error_msg = "Generated mesh is not watertight, volume comparison may be inaccurate."
        logger.warning(error_msg)
        # Return False as the check cannot be reliably performed
        return False, ref_vol, gen_vol, error_msg

    if not ref_mesh.is_watertight:
        error_msg = "Reference mesh is not watertight, volume comparison may be inaccurate."
        logger.warning(error_msg)
        # Return False as the check cannot be reliably performed
        return False, ref_vol, gen_vol, error_msg

    # Check for zero reference volume to avoid division by zero
    if ref_vol == 0:
        if gen_vol == 0:
             passed = True # Both are zero, consider it a pass
             error_msg = None
             logger.info("Both reference and generated volumes are zero.")
        else:
             passed = False
             error_msg = "Reference volume is zero, but generated volume is non-zero."
             logger.warning(error_msg)
        # Return result for zero ref volume case
        return passed, ref_vol, gen_vol, error_msg

    # Calculate percentage difference
    vol_diff = abs(gen_vol - ref_vol)
    percent_diff = (vol_diff / abs(ref_vol)) * 100 # Use abs(ref_vol) just in case

    passed = percent_diff <= volume_threshold_percent
    if not passed:
         logger.info(f"Volume check failed: Diff={percent_diff:.2f}% > Threshold={volume_threshold_percent:.2f}% (Ref: {ref_vol:.2f}, Gen: {gen_vol:.2f})")
         # Keep error_msg as None, failure is indicated by passed=False
         error_msg = None 
    else:
         logger.info(f"Volume check passed: Diff={percent_diff:.2f}% <= Threshold={volume_threshold_percent:.2f}%")
         error_msg = None

    # Explicit final return
    return passed, ref_vol, gen_vol, error_msg

# --- Main Orchestration Function --- Updated ---
def perform_geometry_checks(
    generated_stl_path: str,
    reference_stl_path: str,
    task_requirements: Dict[str, Any],
    rendering_info: Optional[Dict[str, Any]],
    geometry_config: Dict[str, Any] # Changed to expect a dictionary
) -> Dict[str, Any]:
    """
    Perform all geometry checks for a given generated STL file.
    """
    logger.info(f"----- Starting Geometry Checks for: {os.path.basename(generated_stl_path)} -----")

    # --- Get thresholds from geometry_config dictionary --- 
    bbox_tolerance = float(geometry_config.get('bounding_box_tolerance_mm', 1.0))
    chamfer_threshold = float(geometry_config.get('chamfer_threshold_mm', 1.0))
    hausdorff_threshold = float(geometry_config.get('hausdorff_threshold_mm', 1.0))
    icp_fitness_threshold = float(geometry_config.get('icp_fitness_threshold', 0.99))
    expected_component_count = task_requirements.get('topology_requirements', {}).get('expected_component_count', 1)
    # Retrieve volume threshold as a percentage (e.g., 2.0 for 2%)
    volume_threshold_percent = float(geometry_config.get('volume_threshold_percent', 5.0))
    # --- 

    # Initialize result structure
    results = {
        "generated_stl_path": generated_stl_path,
        "reference_stl_path": reference_stl_path,
        "checks": {
            "check_render_successful": None, # Assume None initially
            "check_is_watertight": None,
            "check_is_single_component": None,
            "check_bounding_box_accurate": None,
            "check_volume_passed": None,
            "check_hausdorff_passed": None,
            "check_chamfer_passed": None # Added this key
        },
        "geometric_similarity_distance": None, # Changed from chamfer_distance
        "icp_fitness_score": None,
        "hausdorff_95p_distance": None, # New field for 95p
        "hausdorff_99p_distance": None, # Keep 99p for info
        "reference_volume_mm3": None, # New
        "generated_volume_mm3": None, # New
        "reference_bbox_mm": None, # New
        "generated_bbox_aligned_mm": None, # New
        "error": None,
        "check_errors": [] # List to store specific check failures
    }

    # --- Populate Render Info (If available) --- Start ---
    if rendering_info:
        results["checks"]["check_render_successful"] = rendering_info.get("status") == "Success"
        # If render failed, we might skip some checks or note it
        if not results["checks"]["check_render_successful"]:
            logger.warning(f"Render failed or info missing for {os.path.basename(generated_stl_path)}. Skipping checks that depend on render.")
            # Decide if we should return early or just mark checks as N/A
            # For now, let's continue but checks requiring render success won't proceed
    else:
        # This case applies to providers like zoo_cli that generate STL directly
        results["checks"]["check_render_successful"] = None # Indicate render was not applicable/run
    # --- Populate Render Info (If available) --- End ---

    # --- Early Exit if Generated STL is Missing ---
    if not generated_stl_path or not os.path.exists(generated_stl_path):
        error_msg = f"Generated STL not found or path is invalid: {generated_stl_path}"
        logger.error(error_msg)
        results["error"] = error_msg
        # Set all checks to None/False as they cannot be run
        for check_key in results["checks"]:
            results["checks"][check_key] = None if check_key == "check_render_successful" else False # Render status might be known
        logger.info(f"----- Finished Geometry Checks (STL Missing) for: {os.path.basename(generated_stl_path)} -----")
        return results
    # --- End Early Exit ---

    # --- Check 2: Watertight Check ---
    logger.debug("Attempting watertight check...")
    try:
        is_watertight, watertight_error = check_watertight(generated_stl_path, logger)
        results["checks"]["check_is_watertight"] = is_watertight
        if watertight_error:
            results["check_errors"].append(f"WatertightCheck: {watertight_error}")
    except Exception as e:
        logger.error(f"Unhandled error during watertight check: {e}", exc_info=True)
        results["checks"]["check_is_watertight"] = False
        results["check_errors"].append(f"WatertightCheck: Unhandled Exception - {e}")
    logger.debug("Watertight check finished.")

    # --- Check 3: Single Component Check ---
    logger.debug("Attempting single component check...")
    try:
        is_single_component, component_error = check_single_component(generated_stl_path, task_requirements, logger)
        results["checks"]["check_is_single_component"] = is_single_component
        if component_error:
            results["check_errors"].append(f"SingleComponentCheck: {component_error}")
    except Exception as e:
        logger.error(f"Unhandled error during single component check: {e}", exc_info=True)
        results["checks"]["check_is_single_component"] = False
        results["check_errors"].append(f"SingleComponentCheck: Unhandled Exception - {e}")
    logger.debug("Single component check finished.")

    final_transform = None # Variable to store transform from Similarity Check

    # --- Checks Requiring Reference STL --- Start ---
    if not reference_stl_path or not os.path.exists(reference_stl_path):
        ref_error_msg = f"Reference STL not found ({reference_stl_path}), skipping Similarity, Hausdorff, Chamfer, Aligned BBox, and Volume checks."
        results["check_errors"].append(ref_error_msg)
        logger.warning(ref_error_msg)
        # Set dependent checks to None/False
        results["checks"]["check_bounding_box_accurate"] = None
        results["checks"]["check_hausdorff_passed"] = None
        results["checks"]["check_chamfer_passed"] = None # Set Chamfer to None too
        results["checks"]["check_volume_passed"] = None
    else:
        # --- Check 5: Similarity (Hausdorff/Chamfer) and Alignment ---
        logger.debug("Attempting similarity check (ICP/Chamfer/Hausdorff)...")
        try:
            # Note: check_similarity now returns:
            # (chamfer_distance, icp_fitness, transformation_matrix, hausdorff_95p, hausdorff_99p, error_msg)
            chamfer_dist, icp_fitness, final_transform, hausdorff_95p, hausdorff_99p, sim_error = check_similarity(
                 generated_stl_path,
                 reference_stl_path,
                 chamfer_threshold, # Pass threshold for potential internal use/logging
                 logger
            )
            # Store raw values
            results["geometric_similarity_distance"] = chamfer_dist # Store chamfer distance
            results["icp_fitness_score"] = icp_fitness
            results["hausdorff_95p_distance"] = hausdorff_95p # Store 95p
            results["hausdorff_99p_distance"] = hausdorff_99p # Store 99p

            # Perform Hausdorff check using 95p
            if hausdorff_95p is not None and not np.isinf(hausdorff_95p):
                is_hausdorff_passed = hausdorff_95p <= hausdorff_threshold
                results["checks"]["check_hausdorff_passed"] = is_hausdorff_passed
                if not is_hausdorff_passed:
                    results["check_errors"].append(f"HausdorffCheck: 95p distance {hausdorff_95p:.4f} mm exceeds threshold {hausdorff_threshold} mm")
            else:
                results["checks"]["check_hausdorff_passed"] = False # Treat None/inf as failure
                if hausdorff_95p is None:
                     results["check_errors"].append(f"HausdorffCheck: 95p calculation failed.")
                else: # is inf
                     results["check_errors"].append(f"HausdorffCheck: 95p distance is infinite.")

            # Perform Chamfer check using chamfer_dist
            if chamfer_dist is not None and not np.isinf(chamfer_dist):
                is_chamfer_passed = chamfer_dist <= chamfer_threshold
                results["checks"]["check_chamfer_passed"] = is_chamfer_passed
                if not is_chamfer_passed:
                    results["check_errors"].append(f"ChamferCheck: Distance {chamfer_dist:.4f} mm exceeds threshold {chamfer_threshold} mm")
            else:
                results["checks"]["check_chamfer_passed"] = False # Treat None/inf as failure
                if chamfer_dist is None:
                    results["check_errors"].append(f"ChamferCheck: Calculation failed.")
                else: # is inf
                    results["check_errors"].append(f"ChamferCheck: Distance is infinite.")

            # Handle similarity check errors (e.g., alignment failure, file loading)
            if sim_error:
                 results["check_errors"].append(f"SimilarityCheck Error: {sim_error}")
                 # If alignment failed, don't trust the transform for BBox check
                 if final_transform is None:
                     logger.warning("Similarity check reported an error and alignment transform is missing.")

        except Exception as e:
            logger.error(f"Unhandled error calling check_similarity: {e}", exc_info=True)
            error_msg = f"SimilarityCheck: Unhandled Exception - {e}"
            results["check_errors"].append(error_msg)
            # Set all related results to indicate failure/missing data
            results["geometric_similarity_distance"] = None
            results["icp_fitness_score"] = None
            results["hausdorff_95p_distance"] = None
            results["hausdorff_99p_distance"] = None
            results["checks"]["check_hausdorff_passed"] = False
            results["checks"]["check_chamfer_passed"] = False # Also fail chamfer check
            final_transform = None # Ensure transform is None on error

        # --- Check 4: Bounding Box (Aligned vs Reference) ---
        logger.debug("Attempting aligned bounding box check...")
        if final_transform is not None:
             try:
                 is_bbox_accurate, ref_dims, aligned_dims, bbox_error = compare_aligned_bounding_boxes(
                     generated_stl_path,
                     reference_stl_path,
                     final_transform,
                     geometry_config, # Pass the geometry_config dictionary
                     logger
                 )
                 results["checks"]["check_bounding_box_accurate"] = is_bbox_accurate
                 results["reference_bbox_mm"] = ref_dims
                 results["generated_bbox_aligned_mm"] = aligned_dims
                 if bbox_error:
                     results["check_errors"].append(bbox_error)

             except Exception as e:
                logger.error(f"Unhandled error during aligned bounding box check: {e}", exc_info=True)
                results["checks"]["check_bounding_box_accurate"] = False
                results["check_errors"].append(f"AlignedBBoxCheck: Unhandled Exception - {e}")
        else:
             logger.warning("Skipping aligned bounding box check because alignment transform is missing.")
             results["checks"]["check_bounding_box_accurate"] = None
             results["check_errors"].append("AlignedBBoxCheck: Skipped due to missing alignment transform.")
        logger.debug("Aligned bounding box check finished.")

        # --- Check 6: Volume Check ---
        logger.debug("Attempting volume check...")
        try:
            is_volume_passed, ref_vol, gen_vol, vol_error = check_volume(
                 generated_stl_path,
                 reference_stl_path,
                 volume_threshold_percent, # Pass the percentage value
                 logger
            )
            results["checks"]["check_volume_passed"] = is_volume_passed
            results["reference_volume_mm3"] = ref_vol
            results["generated_volume_mm3"] = gen_vol
            if vol_error:
                results["check_errors"].append(vol_error)

        except Exception as e:
            logger.error(f"Unhandled error during volume check: {e}", exc_info=True)
            results["checks"]["check_volume_passed"] = False
            results["check_errors"].append(f"VolumeCheck: Unhandled Exception - {e}")
        logger.debug("Volume check finished.")

    # --- End of checks requiring reference STL ---

    # --- Convert NumPy bools to Python bools for JSON serialization --- Start ---
    try:
        # Check if 'checks' key exists and is a dictionary
        checks_dict = results.get("checks")
        if isinstance(checks_dict, dict):
            for check_key in checks_dict:
                check_value = checks_dict[check_key]
                # Check if it's a NumPy boolean
                if isinstance(check_value, np.bool_):
                    # Convert to Python boolean
                    results["checks"][check_key] = bool(check_value)
                    # Optional: Add debug logging if needed
                    # logger.debug(f"Converted np.bool_ to bool for check key: {check_key}")
    except Exception as e:
        # Log error but don't crash the whole check process
        logger.warning(f"Error converting boolean types for JSON serialization: {e}")
    # --- Convert NumPy bools to Python bools for JSON serialization --- End ---

    # --- Final summary log --- Start ---
    if results.get("error"):
        logger.error(f"Geometry checks failed for {os.path.basename(generated_stl_path)} due to error: {results['error']}")
    elif results.get("check_errors"):
         logger.warning(f"Geometry checks completed for {os.path.basename(generated_stl_path)} with issues: {'; '.join(results['check_errors'])}")
    else:
        logger.info(f"Geometry checks completed successfully for {os.path.basename(generated_stl_path)}")
    # --- Final summary log --- End ---

    logger.info(f"----- Finished Geometry Checks for: {os.path.basename(generated_stl_path)} -----")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perform geometry checks on STL files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # Add other args if needed for direct testing
    args = parser.parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    log_dir = os.path.join(PROJECT_ROOT, "logs") # Place logs in project logs dir
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(__name__, level=log_level, log_file=os.path.join(log_dir, "geometry_check_standalone.log"))

    logger.info("Running geometry_check.py directly...")
    print("Standalone execution placeholder. Implement test calls as needed.")
    