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
    # Or potentially add to conda environment
    sys.exit(1)

try:
    import open3d as o3d
    # Check for legacy pipelines if needed, adjust based on Open3D version
    # has_legacy_pipelines = hasattr(o3d, 'legacy')
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

# Define DEFAULT_CONFIG with default values for checks
DEFAULT_CONFIG = {
    'bounding_box_tolerance': 1.0, # Default tolerance in mm
    'similarity_threshold': 1.0, # Default Chamfer distance threshold
    # Add other default config values as needed
}

# Environment variable to control debug output
SAVE_PCD_FOR_DEBUG = os.environ.get('SAVE_PCD_FOR_DEBUG', 'false').lower() == 'true'
# --- Define output directory within the project ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) # Assumes script is in a subdir
DEBUG_PCD_DIR = os.path.join(PROJECT_ROOT, "debug_output", "similarity_pcd")
# ---

if SAVE_PCD_FOR_DEBUG:
    os.makedirs(DEBUG_PCD_DIR, exist_ok=True)
    print(f"--- DEBUG: Saving intermediate PCD files to {DEBUG_PCD_DIR} ---") # Updated path

class GeometryCheckError(Exception):
    """Exception raised for errors during geometry checks."""
    pass


# --- Placeholder Check Functions --- 

def check_render_success(render_status: str) -> bool:
    """Check 1: Check if the rendering step was successful."""
    # Simple check based on the status string from the render step
    return render_status == "Success"

def clean_mesh_for_checks(mesh: o3d.geometry.TriangleMesh, logger: logging.Logger, filename: str) -> o3d.geometry.TriangleMesh:
    """Applies cleaning steps relevant for watertight/component checks."""
    try:
        initial_vertices = len(mesh.vertices)
        initial_triangles = len(mesh.triangles)
        # Merge close vertices - essential for fixing small gaps
        mesh.merge_close_vertices(0.0001)
        # Optional: remove other potential issues (can sometimes create problems)
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
        return mesh # Return original mesh if cleaning fails

def check_single_component(
    mesh_file: str,
    requirements: Dict[str, Any], # Add requirements input
    logger: logging.Logger
) -> tuple[bool, str | None]:
    """
    Check if the mesh consists of the expected number of connected components.
    Defaults to expecting 1 component if not specified in requirements.
    Returns (bool, error_msg | None).
    """
    if not os.path.exists(mesh_file):
        error_msg = f"Mesh file not found: {mesh_file}"
        logger.error(error_msg)
        return False, error_msg

    # --- Determine expected component count ---
    expected_components = 1 # Default to 1
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
    # ---

    try:
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if not mesh.has_triangles():
            error_msg = f"Mesh file {mesh_file} is empty or could not be loaded."
            logger.warning(error_msg)
            return False, error_msg

        # --- Attempt cleaning ---
        mesh = clean_mesh_for_checks(mesh, logger, os.path.basename(mesh_file))
        if not mesh.has_triangles(): # Re-check after cleaning
             error_msg = f"Mesh {mesh_file} became empty after cleaning."
             logger.warning(error_msg)
             return False, error_msg
        # --- End cleaning ---

        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        num_components = len(cluster_n_triangles)
        logger.debug(f"File: {mesh_file} (Post-Clean), Found components: {num_components}")

        # --- Compare against expected count ---
        passes_check = num_components == expected_components
        # ---

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
    """
    Check if the mesh is watertight (manifold) AND consists of a single connected component.
    Returns (bool, error_msg | None).
    """
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

        # --- Attempt cleaning ---
        mesh = clean_mesh_for_checks(mesh, logger, os.path.basename(mesh_file))
        if not mesh.has_triangles(): # Re-check after cleaning
             error_msg = f"Mesh {mesh_file} became empty after cleaning."
             logger.warning(error_msg)
             return False, error_msg
        # --- End cleaning ---

        # Check 1: Manifold edges (Open3D's definition)
        logger.debug(f"Checking manifold status for {mesh_file} (Post-Clean)...")
        is_manifold = mesh.is_watertight()
        logger.debug(f"File: {mesh_file} (Post-Clean), o3d.is_watertight() result: {is_manifold}")
        if not is_manifold:
            try:
                edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
                logger.warning(f"Found {len(edges)} non-manifold edges in {mesh_file} (Post-Clean).")
            except Exception as edge_ex:
                 logger.warning(f"Could not get non-manifold edge info for {mesh_file} (Post-Clean): {edge_ex}")
            error_msg = "Mesh is not manifold (non-watertight edges found)"
            logger.info(f"Watertight (manifold edge) check for {mesh_file}: FAILED")
            return False, error_msg

        # Check 2: Single connected component
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        num_components = len(cluster_n_triangles)
        logger.debug(f"File: {mesh_file} (Post-Clean), Found components: {num_components}")
        is_single_component = num_components == 1

        if not is_single_component:
             error_msg = f"Mesh has multiple components ({num_components})"
             logger.info(f"Watertight (single component) check for {mesh_file}: FAILED ({error_msg})")
             return False, error_msg

        logger.info(f"Watertight check (manifold and single component) for {mesh_file}: PASSED")
        return True, None
    except Exception as e:
        error_msg = f"Error checking watertightness for {mesh_file}: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def check_bounding_box(
    generated_stl_path: str, 
    summary_json_path: Optional[str], 
    requirements: Dict[str, Any],
    config: Config
) -> Tuple[Optional[bool], Optional[str]]:
    """Check 4: Check if the bounding box dimensions match requirements within tolerance."""
    logger.debug(f"Checking bounding box for: {generated_stl_path}")
    error_prefix = "BoundingBoxCheck:"

    # Get target dimensions from requirements
    try:
        target_bbox = requirements['bounding_box']
        if not isinstance(target_bbox, list) or len(target_bbox) != 3:
             raise ValueError("bounding_box in requirements must be a list of 3 numbers.")
        target_dims_sorted = sorted([float(d) for d in target_bbox])
        logger.debug(f"  Target dimensions (sorted): {target_dims_sorted}")
    except KeyError:
        return None, f"{error_prefix} Missing 'bounding_box' in task requirements."
    except (ValueError, TypeError) as e:
         return None, f"{error_prefix} Invalid format for 'bounding_box' in task requirements: {e}"

    # Get tolerance from config
    try:
        tolerance = config.get_required('geometry_check.bounding_box_tolerance_mm')
        tolerance = float(tolerance)
        logger.debug(f"  Tolerance: +/- {tolerance} mm")
    except ConfigError:
        return None, f"{error_prefix} Missing 'geometry_check.bounding_box_tolerance_mm' in config."
    except (ValueError, TypeError):
        return None, f"{error_prefix} Invalid tolerance value in config."

    calculated_dims = None
    source = None

    # 1. Try reading from summary JSON first
    if summary_json_path and os.path.exists(summary_json_path):
        logger.debug(f"  Attempting to read bbox from summary file: {summary_json_path}")
        try:
            with open(summary_json_path, 'r') as f:
                summary_data = json.load(f)
            
            # Find bounding box data (structure might vary)
            # Common structures: data['geometry']['boundingbox'], data['boundingbox']
            bbox_data = None
            if isinstance(summary_data.get('geometry'), dict):
                bbox_data = summary_data['geometry'].get('boundingbox')
            elif 'boundingbox' in summary_data:
                 bbox_data = summary_data['boundingbox']
                 
            if isinstance(bbox_data, list) and len(bbox_data) == 6:
                 # Format: [minX, minY, minZ, maxX, maxY, maxZ]
                 min_coords = bbox_data[:3]
                 max_coords = bbox_data[3:]
                 calculated_dims = [max_coords[i] - min_coords[i] for i in range(3)]
                 source = "summary JSON"
                 logger.debug(f"  Got dims from summary JSON: {calculated_dims}")
            else:
                logger.warning("  Bounding box not found or invalid format in summary JSON.")
                
        except json.JSONDecodeError as e:
            logger.warning(f"  Failed to parse summary JSON: {e}")
        except Exception as e:
            logger.warning(f"  Error reading bounding box from summary JSON: {e}")

    # 2. Fallback to Trimesh if summary didn't provide dims
    if calculated_dims is None:
        logger.debug(f"  Falling back to Trimesh calculation for: {generated_stl_path}")
        if not os.path.exists(generated_stl_path):
             return None, f"{error_prefix} Generated STL file not found for Trimesh calculation: {generated_stl_path}"
        try:
            mesh = trimesh.load(generated_stl_path, force='mesh')
            if not mesh.vertices.size > 0:
                 raise ValueError("Mesh loaded by Trimesh has no vertices.")
            min_coords, max_coords = mesh.bounds
            calculated_dims = max_coords - min_coords
            source = "Trimesh"
            logger.debug(f"  Got dims from Trimesh: {calculated_dims}")
        except ValueError as e:
            return None, f"{error_prefix} Failed to load mesh for bbox check using Trimesh: {e}"
        except Exception as e:
            return None, f"{error_prefix} Unexpected error during Trimesh bbox calculation: {e}"

    # 3. Compare dimensions
    if calculated_dims is not None:
        calculated_dims_sorted = sorted(calculated_dims)
        logger.debug(f"  Calculated dims (sorted, from {source}): {calculated_dims_sorted}")
        
        is_accurate = True
        for i in range(3):
            if abs(calculated_dims_sorted[i] - target_dims_sorted[i]) > tolerance:
                is_accurate = False
                logger.debug(f"    Dimension mismatch: target={target_dims_sorted[i]}, calc={calculated_dims_sorted[i]}, diff={abs(calculated_dims_sorted[i] - target_dims_sorted[i]):.3f}, tol={tolerance}")
                break # No need to check further dims
        
        logger.debug(f"  Bounding box accuracy result: {is_accurate}")
        if not is_accurate:
            error_msg = f"BoundingBoxCheck: Extent diff {np.abs(np.array(calculated_dims_sorted) - np.array(target_dims_sorted))} exceeds tolerance {tolerance}"
            return is_accurate, error_msg
        else:
            return is_accurate, None
    else:
        error_msg = f"{error_prefix} Failed to obtain calculated dimensions."
        return False, error_msg

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def check_similarity(generated_stl_path: str, reference_stl_path: str, threshold: float, logger: logging.Logger) -> tuple[float | None, float | None, str | None]:
    """
    Performs ICP alignment and calculates Chamfer distance.
    
    Note on implementation choices (Post-Debugging):
    - Dense point cloud sampling (~50k points) is crucial for accurate ICP on simple shapes.
    - Point-to-Plane ICP estimation generally performs well with dense clouds.
    - Chamfer distance calculation requires careful handling of point correspondences.
    - Comparing identical file paths is handled as a special case (returns 0.0 distance).
    
    Retries on failure. Returns (chamfer_distance: float | None, icp_fitness: float | None, error_msg: str | None).
    """
    # Ensure file existence checks are done early
    if not os.path.exists(generated_stl_path): return None, None, f"Generated file not found: {generated_stl_path}"
    if not os.path.exists(reference_stl_path): return None, None, f"Reference file not found: {reference_stl_path}"
    
    # Special case - if the files are identical (same path), we can skip ICP and report near-zero distance
    if os.path.samefile(generated_stl_path, reference_stl_path):
        logger.debug(f"Files are identical (same path), returning zero distance and perfect fitness")
        return 0.0, 1.0, None # Return 0 distance, perfect fitness (1.0)

    try:
        run_id = f"{os.path.basename(generated_stl_path)}_vs_{os.path.basename(reference_stl_path)}".replace('.stl', '')

        logger.debug(f"Loading meshes for similarity check: {generated_stl_path} vs {reference_stl_path}")
        generated_mesh = o3d.io.read_triangle_mesh(generated_stl_path)
        reference_mesh = o3d.io.read_triangle_mesh(reference_stl_path)
        if not generated_mesh.has_triangles(): return None, None, "Generated mesh has no triangles"
        if not reference_mesh.has_triangles(): return None, None, "Reference mesh has no triangles"

        generated_mesh.compute_vertex_normals()
        reference_mesh.compute_vertex_normals()
        logger.debug("Normals computed.")

        # --- Save Original Meshes if Debugging ---
        if SAVE_PCD_FOR_DEBUG:
            o3d.io.write_triangle_mesh(os.path.join(DEBUG_PCD_DIR, f"{run_id}_ref_mesh.ply"), reference_mesh)
            o3d.io.write_triangle_mesh(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_mesh.ply"), generated_mesh)
        # ---

        # --- Determine Number of Points for Sampling ---
        # Increase the target significantly
        N_POINTS_TARGET = 50000 # Let's target 50,000 points
        logger.debug(f"Targeting {N_POINTS_TARGET} points for uniform sampling.")

        # Check if meshes are valid before sampling
        if not generated_mesh.has_triangles() or not reference_mesh.has_triangles():
            return None, None, "Cannot sample points from empty mesh"
        # ---

        logger.debug(f"Sampling up to {N_POINTS_TARGET} points from each mesh...")
        # Sample points uniformly.
        gen_pcd = generated_mesh.sample_points_uniformly(number_of_points=N_POINTS_TARGET)
        ref_pcd = reference_mesh.sample_points_uniformly(number_of_points=N_POINTS_TARGET)

        # Check if sampling actually produced enough points
        min_actual_points = 100
        actual_gen_points = len(gen_pcd.points)
        actual_ref_points = len(ref_pcd.points)
        logger.info(f"Actual points sampled: Gen={actual_gen_points}, Ref={actual_ref_points}")

        if actual_gen_points < min_actual_points or actual_ref_points < min_actual_points:
            warning_msg = (f"Sampling produced very few points (Gen: {actual_gen_points}, Ref: {actual_ref_points}). "
                           f"Similarity check might be inaccurate.")
            logger.warning(warning_msg)
            if actual_gen_points == 0 or actual_ref_points == 0:
                 return None, None, "Sampling produced zero points."
            # Proceed even with few points, but warning issued.

        # --- Save Sampled PCDs if Debugging ---
        if SAVE_PCD_FOR_DEBUG:
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_ref_sampled.ply"), ref_pcd)
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_sampled.ply"), gen_pcd)
        # ---

        # --- Determine ICP Threshold ---
        icp_correspondence_threshold = 1.0 # Fixed value for now
        logger.debug(f"Using ICP correspondence threshold: {icp_correspondence_threshold}")
        logger.debug(f"Final Chamfer distance threshold: {threshold}")
        # --- End ICP Threshold Determination ---

        # --- Initial Alignment ---
        center_gen = gen_pcd.get_center()
        center_ref = ref_pcd.get_center()
        initial_transform = np.identity(4)
        initial_transform[0:3, 3] = center_ref - center_gen
        logger.debug(f"Initial transformation (translation): {initial_transform[0:3, 3]}")
        gen_pcd_initial_aligned = gen_pcd.transform(initial_transform)

        # --- Save Initially Aligned PCD if Debugging ---
        if SAVE_PCD_FOR_DEBUG:
             o3d.io.write_point_cloud(os.path.join(DEBUG_PCD_DIR, f"{run_id}_gen_initial_aligned.ply"), gen_pcd_initial_aligned)
        # ---

        logger.debug(f"Performing ICP registration with max_correspondence_distance={icp_correspondence_threshold}...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            gen_pcd, ref_pcd, # Use original gen_pcd for registration source
            max_correspondence_distance=icp_correspondence_threshold,
            init=initial_transform, # Provide the initial alignment
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        icp_fitness_score = reg_p2p.fitness # Store the fitness score
        logger.debug(f"ICP Result (PointToPlane): Fitness={icp_fitness_score:.6f}, InlierRMSE={reg_p2p.inlier_rmse:.6f}")

        gen_pcd_transformed = gen_pcd.transform(reg_p2p.transformation)

        logger.debug("Calculating Chamfer distance...")
        dist_gen_to_ref = np.asarray(gen_pcd_transformed.compute_point_cloud_distance(ref_pcd))
        dist_ref_to_gen = np.asarray(ref_pcd.compute_point_cloud_distance(gen_pcd_transformed))

        mean_gen_to_ref = np.mean(dist_gen_to_ref)
        mean_ref_to_gen = np.mean(dist_ref_to_gen)
        max_gen_to_ref = np.max(dist_gen_to_ref)
        max_ref_to_gen = np.max(dist_ref_to_gen)
        std_gen_to_ref = np.std(dist_gen_to_ref)
        std_ref_to_gen = np.std(dist_ref_to_gen)

        logger.debug(f"Distances Gen->Ref: mean={mean_gen_to_ref:.6f}, max={max_gen_to_ref:.6f}, std={std_gen_to_ref:.6f}")
        logger.debug(f"Distances Ref->Gen: mean={mean_ref_to_gen:.6f}, max={max_ref_to_gen:.6f}, std={std_ref_to_gen:.6f}")

        chamfer_distance = (mean_gen_to_ref + mean_ref_to_gen) / 2.0
        logger.info(f"Similarity check for {os.path.basename(generated_stl_path)}: Final Chamfer distance = {chamfer_distance:.6f}, ICP Fitness = {icp_fitness_score:.6f}")

        error_msg = None
        if chamfer_distance > threshold:
             error_msg = f"Chamfer distance {chamfer_distance:.4f} exceeds threshold {threshold}"
             logger.debug(f"--- Similarity Check Note: {error_msg} ---")

        return chamfer_distance, icp_fitness_score, error_msg # Return distance, fitness, and potential error message

    except Exception as e:
        error_msg = f"Error during similarity check between {generated_stl_path} and {reference_stl_path}: {e}"
        logger.error(error_msg, exc_info=True)
        logger.debug(f"--- Similarity Check ERRORED ---")
        return None, None, error_msg # Return None for values on error


# --- Main Orchestration Function --- 

def perform_geometry_checks(
    generated_stl_path: str,
    reference_stl_path: str,
    task_requirements: Dict[str, Any],
    rendering_info: Dict[str, Any], # Contains status, summary_path etc.
    config: Config
) -> Dict[str, Any]:
    """
    Perform all geometry checks for a given generated STL file.
    
    Args:
        generated_stl_path: Path to the generated STL file.
        reference_stl_path: Path to the reference STL file.
        task_requirements: Dictionary of requirements from the task YAML.
        rendering_info: Dictionary containing results from the rendering step.
        config: The loaded Config object.
    
    Returns:
        A dictionary containing the results of all checks, conforming to the 
        schema defined in Section 6 of the project plan.
    """
    logger.info(f"Performing geometry checks for: {os.path.basename(generated_stl_path)}")
    
    check_results = {
        # Check results (booleans or null)
        "check_render_successful": None,
        "check_is_watertight": None,
        "check_is_single_component": None,
        "check_bounding_box_accurate": None,
        # Similarity scores (floats or null)
        "geometric_similarity_distance": None,
        "icp_fitness_score": None,
        # Error messages specific to check phase
        "check_errors": [] 
    }

    # --- Execute Checks --- 
    
    # Check 1: Render Success (Prerequisite for most others)
    render_status = rendering_info.get("status", "Unknown")
    check_results["check_render_successful"] = check_render_success(render_status)
    
    if not check_results["check_render_successful"]:
        logger.warning("Skipping further geometry checks as rendering was not successful.")
        check_results["check_errors"].append("Rendering failed, subsequent checks skipped.")
        return check_results # Early exit if rendering failed
        
    # Check if generated STL path exists (required for all subsequent checks)
    if not generated_stl_path or not os.path.exists(generated_stl_path):
        logger.error(f"Generated STL file not found at {generated_stl_path}, cannot perform geometry checks.")
        check_results["check_errors"].append(f"Generated STL not found ({generated_stl_path}).")
        # Set subsequent checks to None or indicate failure due to missing input
        check_results["check_is_watertight"] = None
        check_results["check_is_single_component"] = None
        check_results["check_bounding_box_accurate"] = None
        return check_results # Cannot proceed without the generated STL
        
    # Check 2: Watertight
    is_watertight = None
    try:
        is_watertight, watertight_error = check_watertight(generated_stl_path, logger)
        check_results["check_is_watertight"] = is_watertight
        if watertight_error: # Add error message if it exists
            check_results["check_errors"].append(f"WatertightCheck: {watertight_error}")
    except Exception as e:
        logger.error(f"Unhandled error calling check_watertight: {e}", exc_info=True)
        check_results["check_is_watertight"] = False # Mark as failed on exception
        check_results["check_errors"].append(f"WatertightCheck: Unhandled Exception - {e}")

    # Check 3: Single Component
    try:
        # Pass task_requirements to the check function
        is_single, single_error = check_single_component(
            generated_stl_path,
            task_requirements, # Pass requirements here
            logger
        )
        check_results["check_is_single_component"] = is_single
        if single_error: # Add error message if it exists
            check_results["check_errors"].append(f"SingleComponentCheck: {single_error}")
    except Exception as e:
        logger.error(f"Unhandled error calling check_single_component: {e}", exc_info=True)
        check_results["check_is_single_component"] = False # Mark as failed on exception
        check_results["check_errors"].append(f"SingleComponentCheck: Unhandled Exception - {e}")

    # Check 4: Bounding Box
    summary_json_path = rendering_info.get("summary_path")
    try:
        is_bbox_accurate, bbox_error = check_bounding_box(
            generated_stl_path,
            summary_json_path,
            task_requirements,
            config
        )
        check_results["check_bounding_box_accurate"] = is_bbox_accurate
        if bbox_error:
            check_results["check_errors"].append(bbox_error)
    except Exception as e:
        logger.error(f"Unhandled error calling check_bounding_box: {e}", exc_info=True)
        check_results["check_bounding_box_accurate"] = False # Mark as failed

    # Check 5: Geometric Similarity
    if not reference_stl_path or not os.path.exists(reference_stl_path):
        check_results["check_errors"].append(f"SimilarityCheck: Reference STL not found ({reference_stl_path}).")
    elif is_watertight is not False: # Run if watertight is True or None (i.e., didn't explicitly fail)
        try:
            threshold = config.get('similarity_threshold', DEFAULT_CONFIG['similarity_threshold'])
            chamfer_dist, icp_fitness, sim_error = check_similarity(generated_stl_path, reference_stl_path, threshold, logger)
            check_results["geometric_similarity_distance"] = chamfer_dist
            check_results["icp_fitness_score"] = icp_fitness
            if sim_error: # Add error message if it exists
                check_results["check_errors"].append(f"SimilarityCheck: {sim_error}")
        except Exception as e:
            logger.error(f"Unhandled error calling check_similarity: {e}", exc_info=True)
            check_results["check_errors"].append(f"SimilarityCheck: Unhandled Exception - {e}")
            check_results["geometric_similarity_distance"] = None
            check_results["icp_fitness_score"] = None
    else:
        logger.warning("Skipping similarity check due to prior watertight/load failure.")
        check_results["check_errors"].append("SimilarityCheck: Skipped due to prior watertight/load failure.")
        
    logger.info(f"Geometry checks completed for: {os.path.basename(generated_stl_path)}")
    return check_results


if __name__ == "__main__":
    # Example usage when run directly (for testing individual functions)
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Perform geometry checks on STL files")
    # Add arguments if needed for standalone testing, e.g.:
    # parser.add_argument("--generated-stl", required=True, help="Path to the generated STL file")
    # parser.add_argument("--reference-stl", required=True, help="Path to the reference STL file")
    # parser.add_argument("--task-yaml", required=True, help="Path to the task definition YAML")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(__name__, level=log_level, log_file="logs/geometry_check.log")
    
    logger.info("Running geometry_check.py directly...")
    # Add example calls to check functions or perform_geometry_checks here for testing
    print("Standalone execution placeholder. Implement test calls as needed.")
    
    # Example: Test watertight check on a known file
    # try:
    #    test_stl = "path/to/some/test.stl" # Provide a real path for testing
    #    if os.path.exists(test_stl):
    #        is_watertight, error = check_watertight(test_stl)
    #        print(f"Watertight check on {test_stl}: Result={is_watertight}, Error={error}")
    #    else:
    #        print(f"Test STL file not found: {test_stl}")
    # except Exception as e:
    #     print(f"Error during standalone test: {e}")
    