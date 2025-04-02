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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def check_similarity(
    generated_stl_path: str,
    reference_stl_path: str,
    threshold: float,
    logger: logging.Logger
) -> tuple[float | None, float | None, np.ndarray | None, str | None]:
    """
    Check 5: Performs ICP alignment and calculates Chamfer distance.
    Returns (chamfer_distance, icp_fitness, transformation_matrix, error_msg).
    """
    if not os.path.exists(generated_stl_path): return None, None, None, f"Generated file not found: {generated_stl_path}"
    if not os.path.exists(reference_stl_path): return None, None, None, f"Reference file not found: {reference_stl_path}"

    if os.path.samefile(generated_stl_path, reference_stl_path):
        logger.debug("Files are identical (same path), returning zero distance, perfect fitness, identity transform")
        identity_transform = np.identity(4)
        return 0.0, 1.0, identity_transform, None

    transformation_matrix = None
    try:
        run_id = f"{os.path.basename(generated_stl_path)}_vs_{os.path.basename(reference_stl_path)}".replace('.stl', '')
        logger.debug(f"Loading meshes for similarity check: {generated_stl_path} vs {reference_stl_path}")
        generated_mesh = o3d.io.read_triangle_mesh(generated_stl_path)
        reference_mesh = o3d.io.read_triangle_mesh(reference_stl_path)
        if not generated_mesh.has_triangles(): return None, None, None, "Generated mesh has no triangles"
        if not reference_mesh.has_triangles(): return None, None, None, "Reference mesh has no triangles"

        generated_mesh.compute_vertex_normals()
        reference_mesh.compute_vertex_normals()

        if SAVE_PCD_FOR_DEBUG:
            o3d.io.write_triangle_mesh(os.path.join(DEBUG_PCD_DIR, f"{run_id}_ref_mesh.ply"), reference_mesh)
            o3d.io.write_triangle_mesh(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_mesh.ply"), generated_mesh)

        N_POINTS_TARGET = 50000 # Target point count for sampling
        logger.debug(f"Targeting {N_POINTS_TARGET} points for uniform sampling.")
        if not generated_mesh.has_triangles() or not reference_mesh.has_triangles():
            return None, None, None, "Cannot sample points from empty mesh"

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
                 return None, None, None, "Sampling produced zero points."

        if SAVE_PCD_FOR_DEBUG:
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_ref_sampled.ply"), ref_pcd)
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_sampled.ply"), gen_pcd)

        # ICP parameters
        icp_correspondence_threshold = 1.0 # Threshold distance for correspondences
        logger.debug(f"Using ICP correspondence threshold: {icp_correspondence_threshold}")
        logger.debug(f"Using Chamfer distance similarity threshold: {threshold} mm")

        # Initial alignment (center alignment)
        center_gen = gen_pcd.get_center()
        center_ref = ref_pcd.get_center()
        initial_transform = np.identity(4)
        initial_transform[0:3, 3] = center_ref - center_gen

        if SAVE_PCD_FOR_DEBUG:
             gen_pcd_initial_aligned = o3d.geometry.PointCloud(gen_pcd) # Create copy
             gen_pcd_initial_aligned.transform(initial_transform)
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_initial_aligned.ply"), gen_pcd_initial_aligned)

        logger.debug(f"Performing ICP registration...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source=gen_pcd, # Point cloud to be transformed
            target=ref_pcd, # Reference point cloud
            max_correspondence_distance=icp_correspondence_threshold,
            init=initial_transform, # Initial guess
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        icp_fitness_score = reg_p2p.fitness
        transformation_matrix = reg_p2p.transformation
        logger.debug(f"ICP Result: Fitness={icp_fitness_score:.6f}, InlierRMSE={reg_p2p.inlier_rmse:.6f}")

        # Apply final transformation to generated point cloud for distance calc
        gen_pcd_transformed = o3d.geometry.PointCloud(gen_pcd) # Create copy
        gen_pcd_transformed.transform(transformation_matrix)

        if SAVE_PCD_FOR_DEBUG:
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_final_aligned.ply"), gen_pcd_transformed)

        # Calculate Chamfer distance
        logger.debug("Calculating Chamfer distance...")
        dist_gen_to_ref = np.asarray(gen_pcd_transformed.compute_point_cloud_distance(ref_pcd))
        dist_ref_to_gen = np.asarray(ref_pcd.compute_point_cloud_distance(gen_pcd_transformed))
        chamfer_distance = (np.mean(dist_gen_to_ref) + np.mean(dist_ref_to_gen)) / 2.0
        logger.info(f"Similarity check for {os.path.basename(generated_stl_path)}: Final Chamfer distance = {chamfer_distance:.6f} mm, ICP Fitness = {icp_fitness_score:.6f}")

        error_msg = None
        if chamfer_distance > threshold:
             error_msg = f"Chamfer distance {chamfer_distance:.4f} mm exceeds threshold {threshold} mm"
             # This is just a note, not necessarily a failure of the check itself

        return chamfer_distance, icp_fitness_score, transformation_matrix, error_msg
    except Exception as e:
        error_msg = f"Error during similarity check between {generated_stl_path} and {reference_stl_path}: {e}"
        logger.error(error_msg, exc_info=True)
        return None, None, None, error_msg

def compare_aligned_bounding_boxes(
    generated_stl_path: str,
    reference_stl_path: str,
    icp_transform: np.ndarray,
    config: Config,
    logger: logging.Logger
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Check 4: Compares the bounding box of the ICP-aligned generated mesh against the reference mesh.
    Returns (is_accurate, error_message)
    """
    error_prefix = "AlignedBBoxCheck:"
    try:
        tolerance = config.get_required('geometry_check.bounding_box_tolerance_mm')
        tolerance = float(tolerance)
        logger.debug(f"  Tolerance for aligned bbox check: +/- {tolerance} mm")

        # Load reference mesh and get its sorted dimensions
        ref_mesh = trimesh.load(reference_stl_path, force='mesh')
        if not ref_mesh.vertices.size > 0: raise ValueError("Reference mesh empty")
        ref_dims_sorted = sorted(ref_mesh.bounding_box.extents)
        logger.debug(f"    Reference BBox Dims (Sorted): {[f'{d:.4f}' for d in ref_dims_sorted]}")

        # Load generated mesh
        gen_mesh = trimesh.load(generated_stl_path, force='mesh')
        if not gen_mesh.vertices.size > 0: raise ValueError("Generated mesh empty")

        # Apply transform and get its sorted dimensions
        gen_mesh.apply_transform(icp_transform)
        aligned_gen_dims_sorted = sorted(gen_mesh.bounding_box.extents)
        logger.debug(f"    Aligned Gen BBox Dims (Sorted): {[f'{d:.4f}' for d in aligned_gen_dims_sorted]}")

        # Compare dimensions (Reference vs Aligned Generated)
        is_accurate = True
        diffs = np.abs(np.array(aligned_gen_dims_sorted) - np.array(ref_dims_sorted))
        for i in range(3):
            if diffs[i] > tolerance:
                is_accurate = False
                logger.debug(f"    Dimension mismatch (Ref vs Aligned): ref={ref_dims_sorted[i]:.4f}, aligned_gen={aligned_gen_dims_sorted[i]:.4f}, diff={diffs[i]:.4f}, tol={tolerance}")
                break # No need to check further dims

        logger.debug(f"  Aligned bounding box accuracy result: {is_accurate}")
        if not is_accurate:
            error_msg = f"{error_prefix} Aligned vs Ref BBox Dims Diff (Sorted): {[f'{d:.4f}' for d in diffs.tolist()]} exceeds tolerance {tolerance}"
            return is_accurate, error_msg
        else:
            return is_accurate, None

    except ConfigError as e:
         # Specific error for missing tolerance config
         return None, f"{error_prefix} Missing/Invalid 'geometry_check.bounding_box_tolerance_mm' config: {e}"
    except FileNotFoundError as e:
        return None, f"{error_prefix} Mesh file not found: {e}"
    except ValueError as e: # Catch mesh loading/empty errors
         return None, f"{error_prefix} Error loading/processing mesh for bbox check: {e}"
    except Exception as e:
        logger.error(f"    Unhandled error during aligned bbox comparison: {e}", exc_info=True)
        return None, f"{error_prefix} Unhandled Exception: {e}"

# --- Main Orchestration Function ---

def perform_geometry_checks(
    generated_stl_path: str,
    reference_stl_path: str,
    task_requirements: Dict[str, Any],
    rendering_info: Dict[str, Any],
    config: Config
) -> Dict[str, Any]:
    """
    Perform all geometry checks for a given generated STL file.
    """
    logger.info(f"Performing geometry checks for: {os.path.basename(generated_stl_path)}")

    check_results = {
        "check_render_successful": None,
        "check_is_watertight": None,
        "check_is_single_component": None,
        "check_bounding_box_accurate": None, # This will now reflect aligned check
        "geometric_similarity_distance": None,
        "icp_fitness_score": None,
        "check_errors": [] # Store errors encountered during checks
    }

    icp_transform = None # Variable to store transform from Check 5

    # --- Execute Checks ---

    # Check 1: Render Success
    render_status = rendering_info.get("status", "Unknown")
    check_results["check_render_successful"] = check_render_success(render_status)
    if not check_results["check_render_successful"]:
        logger.warning("Skipping further geometry checks as rendering was not successful.")
        check_results["check_errors"].append("Rendering failed, subsequent checks skipped.")
        return check_results

    # Check if generated STL path exists
    if not generated_stl_path or not os.path.exists(generated_stl_path):
        logger.error(f"Generated STL file not found at {generated_stl_path}, cannot perform geometry checks.")
        check_results["check_errors"].append(f"Generated STL not found ({generated_stl_path}).")
        # Set subsequent checks to None or indicate failure due to missing input
        check_results["check_is_watertight"] = None
        check_results["check_is_single_component"] = None
        check_results["check_bounding_box_accurate"] = None
        return check_results

    # Check 2: Watertight
    is_watertight = None # Initialize for Check 5 prerequisite check
    try:
        is_watertight, watertight_error = check_watertight(generated_stl_path, logger)
        check_results["check_is_watertight"] = is_watertight
        if watertight_error:
            check_results["check_errors"].append(f"WatertightCheck: {watertight_error}")
    except Exception as e:
        logger.error(f"Unhandled error calling check_watertight: {e}", exc_info=True)
        check_results["check_is_watertight"] = False
        check_results["check_errors"].append(f"WatertightCheck: Unhandled Exception - {e}")

    # Check 3: Single Component
    try:
        is_single, single_error = check_single_component(
            generated_stl_path,
            task_requirements, # Pass requirements here
            logger
        )
        check_results["check_is_single_component"] = is_single
        if single_error:
            check_results["check_errors"].append(f"SingleComponentCheck: {single_error}")
    except Exception as e:
        logger.error(f"Unhandled error calling check_single_component: {e}", exc_info=True)
        check_results["check_is_single_component"] = False
        check_results["check_errors"].append(f"SingleComponentCheck: Unhandled Exception - {e}")

    # Check 5: Geometric Similarity (Runs before Check 4)
    if not reference_stl_path or not os.path.exists(reference_stl_path):
        check_results["check_errors"].append(f"SimilarityCheck: Reference STL not found ({reference_stl_path}), skipping Similarity and Aligned BBox checks.")
        check_results["check_bounding_box_accurate"] = None # Cannot perform aligned bbox check without reference
    elif is_watertight is not False: # Run if watertight check didn't explicitly fail
        try:
            # Standardize config access for similarity threshold
            similarity_key = 'geometry_check.similarity_threshold_mm'
            default_sim_thresh = DEFAULT_CONFIG.get('geometry_check', {}).get('similarity_threshold_mm', 1.0)
            similarity_threshold_value = config.get(similarity_key, default_sim_thresh)
            similarity_threshold_value = float(similarity_threshold_value)

            chamfer_dist, icp_fitness, icp_transform, sim_error = check_similarity(
                 generated_stl_path,
                 reference_stl_path,
                 similarity_threshold_value, # Pass the float threshold
                 logger
            )
            check_results["geometric_similarity_distance"] = chamfer_dist
            check_results["icp_fitness_score"] = icp_fitness

            if sim_error:
                # Only record as error if it's not just exceeding threshold
                if "exceeds threshold" not in sim_error:
                     check_results["check_errors"].append(f"SimilarityCheck Error: {sim_error}")
                     icp_transform = None # Don't trust transform if similarity calc failed
                else:
                    # Log threshold exceedance as info/debug, not error that stops bbox check
                    logger.info(f"SimilarityCheck Note: {sim_error}")

        except (ConfigError, ValueError, TypeError) as e:
             logger.error(f"Invalid configuration for key '{similarity_key}': {e}", exc_info=True)
             check_results["check_errors"].append(f"SimilarityCheck: Invalid threshold config - {e}")
             icp_transform = None # Cannot proceed
        except Exception as e:
            logger.error(f"Unhandled error calling check_similarity: {e}", exc_info=True)
            check_results["check_errors"].append(f"SimilarityCheck: Unhandled Exception - {e}")
            check_results["geometric_similarity_distance"] = None
            check_results["icp_fitness_score"] = None
            icp_transform = None # Ensure transform is None on error

    else: # Watertight check failed explicitly
        logger.warning("Skipping similarity check and aligned bbox check due to prior watertight/load failure.")
        check_results["check_errors"].append("SimilarityCheck/AlignedBBoxCheck: Skipped due to prior watertight/load failure.")
        check_results["check_bounding_box_accurate"] = None

    # Check 4: Bounding Box (Aligned vs Reference)
    # Runs only if ICP transform was successfully obtained in Check 5
    if icp_transform is not None:
         try:
             is_bbox_accurate, bbox_error = compare_aligned_bounding_boxes(
                 generated_stl_path,
                 reference_stl_path,
                 icp_transform,
                 config,
                 logger
             )
             check_results["check_bounding_box_accurate"] = is_bbox_accurate
             # Add error message only if the check explicitly failed (result False)
             if is_bbox_accurate is False and bbox_error:
                 check_results["check_errors"].append(bbox_error)
             elif bbox_error: # Log other errors (e.g., config/file not found)
                 check_results["check_errors"].append(bbox_error)

         except Exception as e: # Catch potential unhandled errors from the helper
            logger.error(f"Unhandled error during aligned bounding box check: {e}", exc_info=True)
            check_results["check_bounding_box_accurate"] = None # Indicate failure/error
            check_results["check_errors"].append(f"AlignedBBoxCheck: Unhandled Exception - {e}")
    # Else: check_results["check_bounding_box_accurate"] remains None because icp_transform was None

    logger.info(f"Geometry checks completed for: {os.path.basename(generated_stl_path)}")
    # Add check_results_valid flag? Or rely on None values? Let's rely on None for now.
    return check_results

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
    