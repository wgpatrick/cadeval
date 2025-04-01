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

class GeometryCheckError(Exception):
    """Exception raised for errors during geometry checks."""
    pass


# --- Placeholder Check Functions --- 

def check_render_success(render_status: str) -> bool:
    """Check 1: Check if the rendering step was successful."""
    # Simple check based on the status string from the render step
    return render_status == "Success"

def check_watertight(stl_path: str) -> Tuple[Optional[bool], Optional[str]]:
    """Check 2: Check if the mesh is watertight using Trimesh."""
    logger.debug(f"Checking watertightness for: {stl_path}")
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        is_watertight = bool(mesh.is_watertight)
        logger.debug(f"  Result: {is_watertight}")
        return is_watertight, None
    except ValueError as e:
        # Handle common trimesh loading errors (e.g., empty file, not mesh)
        error_msg = f"Failed to load mesh for watertight check: {e}"
        logger.warning(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during watertight check: {e}"
        logger.error(error_msg)
        return None, error_msg

def check_single_component(stl_path: str, requirements: Dict[str, Any]) -> Tuple[Optional[bool], Optional[str]]:
    """Check 3: Check if the mesh consists of a single connected component."""
    logger.debug(f"Checking single component for: {stl_path}")
    
    # Default assumption if not specified in requirements
    expected_count = requirements.get('topology_requirements', {}).get('expected_component_count', 1)
    logger.debug(f"  Expected component count: {expected_count}")

    try:
        mesh = trimesh.load(stl_path, force='mesh')
        # Split the mesh into potentially disconnected bodies
        components = mesh.split()
        body_count = len(components)
        logger.debug(f"  Detected component count: {body_count}")
        
        is_correct_count = (body_count == expected_count)
        return is_correct_count, None
        
    except ValueError as e:
        error_msg = f"Failed to load mesh for single component check: {e}"
        logger.warning(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during single component check: {e}"
        logger.error(error_msg)
        return None, error_msg

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
        return is_accurate, None
    else:
        # Should not happen if logic is correct, but safeguard
        return None, f"{error_prefix} Failed to obtain calculated dimensions."

def check_similarity(
    generated_stl_path: str, 
    reference_stl_path: str
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Check 5: Calculate geometric similarity using ICP and Chamfer Distance.

    Returns: 
        Tuple(icp_fitness_score, chamfer_distance, error_message)
    """
    logger.debug(f"Checking similarity between {generated_stl_path} and {reference_stl_path}")
    error_prefix = "SimilarityCheck:"
    icp_fitness = None
    chamfer_distance = None

    # Default parameters for ICP and Chamfer
    # Using default ICP parameters as specified in the plan
    icp_threshold = 1.0 # Default correspondence distance threshold for ICP
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000) # Default criteria
    number_of_points_for_chamfer = 5000 # Number of points to sample for Chamfer

    try:
        # 1. Load meshes
        try:
            logger.debug("  Loading reference mesh...")
            mesh_ref = o3d.io.read_triangle_mesh(reference_stl_path)
            if not mesh_ref.has_vertices() or not mesh_ref.has_triangles():
                 raise ValueError("Reference mesh is empty or invalid after loading.")
            logger.debug("  Reference mesh loaded.")
        except Exception as e:
            return None, None, f"{error_prefix} Failed to load reference mesh ({reference_stl_path}): {e}"
            
        try:
             logger.debug("  Loading generated mesh...")
             mesh_gen = o3d.io.read_triangle_mesh(generated_stl_path)
             if not mesh_gen.has_vertices() or not mesh_gen.has_triangles():
                 raise ValueError("Generated mesh is empty or invalid after loading.")
             logger.debug("  Generated mesh loaded.")
        except Exception as e:
            return None, None, f"{error_prefix} Failed to load generated mesh ({generated_stl_path}): {e}"

        # 2. Perform ICP Alignment (align generated mesh to reference mesh)
        logger.debug("  Performing ICP alignment...")
        # Ensure meshes have normals computed for robust ICP
        mesh_gen.compute_vertex_normals()
        mesh_ref.compute_vertex_normals()
        
        # Initial guess: identity matrix
        init_transform = np.identity(4)
        
        # Run ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            mesh_gen, mesh_ref, icp_threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            icp_criteria
        )
        
        icp_fitness = reg_p2p.fitness # Proportion of inliers
        # Lower rmse is generally better, but fitness score (0-1) is often used
        # Let's record fitness as per the plan, RMSE might also be useful: reg_p2p.inlier_rmse
        logger.debug(f"  ICP Fitness: {icp_fitness:.4f}")
        # Apply the transformation to align the generated mesh
        mesh_gen_aligned = mesh_gen.transform(reg_p2p.transformation)
        logger.debug("  Generated mesh aligned.")

        # 3. Calculate Chamfer Distance
        logger.debug(f"  Calculating Chamfer distance using {number_of_points_for_chamfer} sampled points...")
        # Sample points from both meshes
        pcd_ref = mesh_ref.sample_points_poisson_disk(number_of_points=number_of_points_for_chamfer)
        pcd_gen_aligned = mesh_gen_aligned.sample_points_poisson_disk(number_of_points=number_of_points_for_chamfer)
        
        if not pcd_ref.has_points() or not pcd_gen_aligned.has_points():
             raise ValueError("Failed to sample sufficient points for Chamfer distance calculation.")

        # Compute distances between point clouds
        dist_gen_to_ref = pcd_gen_aligned.compute_point_cloud_distance(pcd_ref)
        dist_ref_to_gen = pcd_ref.compute_point_cloud_distance(pcd_gen_aligned)
        
        # Calculate Chamfer distance (average of mean squared distances)
        chamfer_l1 = np.mean(dist_gen_to_ref) + np.mean(dist_ref_to_gen)
        # Or L2 (sum of mean squared distances) - L1 is more common as 'Chamfer'
        # chamfer_l2 = np.mean(np.square(dist_gen_to_ref)) + np.mean(np.square(dist_ref_to_gen))
        
        chamfer_distance = chamfer_l1 
        logger.debug(f"  Chamfer Distance (L1): {chamfer_distance:.4f}")

        return icp_fitness, chamfer_distance, None

    except Exception as e:
        error_msg = f"Unexpected error during similarity check: {e}"
        logger.error(error_msg)
        # Return current values (might have ICP fitness even if Chamfer fails)
        return icp_fitness, chamfer_distance, f"{error_prefix} {error_msg}"


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
    is_watertight, wt_error = check_watertight(generated_stl_path)
    check_results["check_is_watertight"] = is_watertight
    if wt_error: check_results["check_errors"].append(f"WatertightCheck: {wt_error}")
    can_load_mesh = (wt_error is None or "Failed to load mesh" not in wt_error)

    # Check 3: Single Component
    if can_load_mesh:
         is_single, sc_error = check_single_component(generated_stl_path, task_requirements)
         check_results["check_is_single_component"] = is_single
         if sc_error: check_results["check_errors"].append(f"SingleComponentCheck: {sc_error}")
    else:
         logger.warning("Skipping single component check due to mesh load failure.")
         check_results["check_errors"].append("SingleComponentCheck: Skipped due to mesh load error.")

    # Check 4: Bounding Box
    summary_json_path = rendering_info.get("summary_path") # Get path from render results
    is_bbox_accurate, bbox_error = check_bounding_box(
        generated_stl_path, 
        summary_json_path, 
        task_requirements, 
        config
    )
    check_results["check_bounding_box_accurate"] = is_bbox_accurate
    if bbox_error: check_results["check_errors"].append(bbox_error) # Error already prefixed

    # Check 5: Geometric Similarity
    if not reference_stl_path or not os.path.exists(reference_stl_path):
         logger.warning(f"Reference STL file not found at {reference_stl_path}, skipping similarity check.")
         check_results["check_errors"].append(f"SimilarityCheck: Reference STL not found ({reference_stl_path}).")
    elif can_load_mesh: # Check if generated mesh could be loaded
        icp_score, chamfer_dist, sim_error = check_similarity(generated_stl_path, reference_stl_path)
        check_results["icp_fitness_score"] = icp_score
        check_results["geometric_similarity_distance"] = chamfer_dist
        if sim_error: check_results["check_errors"].append(sim_error)
    else:
        logger.warning("Skipping similarity check due to generated mesh load failure.")
        check_results["check_errors"].append("SimilarityCheck: Skipped due to mesh load error.")
        
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
    