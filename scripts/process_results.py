#!/usr/bin/env python3
"""
Processes the results JSON file generated by run_evaluation.py to compute
meta-statistics and prepare data for a dashboard display.
"""

import json
import os
import logging
import argparse
import statistics
from collections import defaultdict
import sys
import numpy as np # Import numpy
import yaml
from typing import List, Dict, Any, Tuple

# --- Add project root to sys.path --- Start ---
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root (assuming scripts is one level below root)
project_root = os.path.dirname(script_dir)
# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")
# --- Add project root to sys.path --- End ---

try:
    from scripts.config_loader import get_config, Config
    from scripts.logger_setup import setup_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from the project root or the PYTHONPATH is set correctly.")
    sys.exit(1)

logger = logging.getLogger(__name__) # Use standard logger

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """Loads configuration from YAML file."""
    try:
        logger.info(f"Initializing configuration using path: {config_path}")
        return Config(config_path) # NEW: Pass path string
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise # Re-raise FileNotFoundError to be caught by Config or caller
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise # Re-raise YAMLError
    except Exception as e:
        logger.error(f"Unexpected error initializing configuration: {e}")
        raise # Re-raise other exceptions

# --- Results Loading ---
def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Loads results from a JSON file or directory of JSON files."""
    all_results = []
    if os.path.isdir(results_path):
        logger.info(f"Loading results from directory: {results_path}")
        for filename in os.listdir(results_path):
            if filename.endswith("_summary.json"):
                file_path = os.path.join(results_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # If the summary file contains a list, extend; otherwise append
                        if isinstance(data, list):
                            all_results.extend(data)
                        elif isinstance(data, dict):
                            all_results.append(data)
                        else:
                             logger.warning(f"Unexpected data type in {filename}: {type(data)}")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from file: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
    elif os.path.isfile(results_path) and results_path.endswith(".json"):
        logger.info(f"Loading results directly from file: {results_path}")
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                     all_results = data # Assume it's a list of result dicts
                     logger.info(f"Loaded {len(all_results)} result entries directly from list in {results_path}")
                else:
                     logger.error(f"Expected a list of results in {results_path}, found {type(data)}")
                     return [] # Return empty on unexpected format
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {results_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading file {results_path}: {e}")
            return []
    else:
        logger.error(f"Invalid results path provided: {results_path}")
        return []

    return all_results

# --- Threshold Helper ---
def get_threshold(config: Config, key: str, default: float) -> float:
    """Safely gets a float threshold from config, logs warning and uses default if missing/invalid."""
    try:
        # Use the get method with nested keys
        value = config.get(key)
        if value is None:
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default
        return float(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid value for configuration key '{key}' ({value}), using default: {default}. Error: {e}")
        return default
    except Exception as e:
         logger.error(f"Unexpected error getting threshold for key '{key}': {e}. Using default: {default}.", exc_info=True)
         return default

# --- Data Processing ---
def process_data_for_dashboard(results: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    """Processes the raw results data into the format needed for the dashboard."""
    dashboard_data = []

    # --- Re-enable Threshold reading --- Start ---
    bbox_tolerance = get_threshold(config, 'geometry_check.bounding_box_tolerance_mm', 1.0)
    hausdorff_threshold = get_threshold(config, 'geometry_check.hausdorff_threshold_mm', 0.5) # Use same default as geometry_check.py for 95p
    volume_threshold = get_threshold(config, 'geometry_check.volume_threshold_percent', 1.0)

    logger.info(f"Using thresholds for dashboard pass/fail determination:")
    logger.info(f"  - BBox Tolerance: +/- {bbox_tolerance} mm")
    logger.info(f"  - Hausdorff 95p Threshold: {hausdorff_threshold} mm")
    logger.info(f"  - Volume Threshold: {volume_threshold}% diff")
    # --- Re-enable Threshold reading --- End ---

    # Format numbers for display
    def fmt(val, precision=4):
        if val is None:
            return "N/A"
        try:
            f_val = float(val)
            if np.isinf(f_val): return "Inf"
            if np.isnan(f_val): return "NaN"
            return f"{f_val:.{precision}f}"
        except (ValueError, TypeError):
            # Return the original value if it can't be formatted as float
            return str(val)

    # Add threshold for Chamfer
    similarity_threshold = get_threshold(config, 'geometry_check.similarity_threshold_mm', 1.0)
    logger.info(f"  - Chamfer Threshold: {similarity_threshold} mm")

    for entry in results: # Iterating through raw replicate entries
        checks_data = entry.get("checks", {})
        if not isinstance(checks_data, dict):
            logger.warning(f"Skipping entry due to invalid 'checks' format (expected dict): {entry.get('task_id', 'N/A')}/{entry.get('model_name', 'N/A')}")
            checks_data = {} # Use empty dict to avoid errors below

        # --- Calculate boolean flags --- Start ---
        scad_gen_success = entry.get("generation_error") is None or entry.get("generation_error") == ""
        
        render_status = entry.get("render_status", "N/A")
        render_success = render_status == "Success"
        
        # Checks are considered run if render succeeded, allowing checks to execute.
        checks_run_attempted = render_success 

        watertight = checks_data.get("check_is_watertight", entry.get("check_is_watertight"))
        single_comp = checks_data.get("check_is_single_component", entry.get("check_is_single_component"))

        # BBox Check
        bbox_passed = None
        ref_bbox_val = entry.get("reference_bbox_mm")
        gen_bbox_val = entry.get("generated_bbox_aligned_mm")
        if render_success and ref_bbox_val is not None and gen_bbox_val is not None:
            try:
                ref_dims = sorted([float(d) for d in ref_bbox_val])
                gen_dims = sorted([float(d) for d in gen_bbox_val])
                diffs = np.abs(np.array(gen_dims) - np.array(ref_dims))
                bbox_passed = all(d <= bbox_tolerance for d in diffs)
            except Exception as e:
                logger.warning(f"Error calculating bbox diff for {entry.get('task_id')}/{entry.get('model_name')}/{entry.get('replicate_id')}: {e}")
                bbox_passed = False # Fail if calculation errors
        elif not render_success:
            bbox_passed = None # Cannot check if render failed
        elif checks_data.get("check_bounding_box_accurate") is not None:
             bbox_passed = checks_data.get("check_bounding_box_accurate") # Fallback
        else:
             bbox_passed = None # Indicate check couldn't be determined

        # Hausdorff Check
        hausdorff_passed = None
        hausdorff_95p = entry.get("hausdorff_95p_distance")
        if render_success and hausdorff_95p is not None:
            try:
                f_hausdorff_95p = float(hausdorff_95p)
                if not np.isinf(f_hausdorff_95p) and not np.isnan(f_hausdorff_95p):
                    hausdorff_passed = f_hausdorff_95p <= hausdorff_threshold
                # else: hausdorff_passed remains None if inf/nan
            except (ValueError, TypeError):
                 hausdorff_passed = False # Fail if not a number
        elif not render_success:
            hausdorff_passed = None
        elif checks_data.get("check_hausdorff_passed") is not None:
            hausdorff_passed = checks_data.get("check_hausdorff_passed") # Fallback
        else:
            hausdorff_passed = None

        # Volume Check
        volume_passed = None
        ref_vol = entry.get("reference_volume_mm3")
        gen_vol = entry.get("generated_volume_mm3")
        if render_success and ref_vol is not None and gen_vol is not None:
            try:
                 f_ref_vol = float(ref_vol)
                 f_gen_vol = float(gen_vol)
                 if abs(f_ref_vol) > 1e-9: # Avoid division by zero
                     percent_diff = (abs(f_gen_vol - f_ref_vol) / abs(f_ref_vol)) * 100
                     volume_passed = percent_diff <= volume_threshold
                 else:
                     volume_passed = abs(f_gen_vol) <= 1e-9 # Pass if both are essentially zero
            except (ValueError, TypeError):
                 volume_passed = False # Fail if not numbers
        elif not render_success:
            volume_passed = None
        elif checks_data.get("check_volume_passed") is not None:
             volume_passed = checks_data.get("check_volume_passed") # Fallback
        else:
            volume_passed = None

        # Chamfer Check
        chamfer_passed = None
        chamfer_dist = entry.get("geometric_similarity_distance")
        if render_success and chamfer_dist is not None:
            try:
                f_chamfer_dist = float(chamfer_dist)
                if not np.isinf(f_chamfer_dist) and not np.isnan(f_chamfer_dist):
                     chamfer_passed = f_chamfer_dist <= similarity_threshold
            except (ValueError, TypeError):
                 chamfer_passed = False # Fail if not a number
        elif not render_success:
            chamfer_passed = None
        # No fallback for chamfer needed currently as it's not in `checks` dict
        else:
            chamfer_passed = None

        # Recalculate Overall Passed - requires all checks that *could* run to be True
        # Checks that are None (couldn't run) don't count against it.
        required_checks = [
            scad_gen_success, # Must always succeed
            render_success,   # Must always succeed for checks to run
            watertight,       # Check result (True/False/None)
            single_comp,    # Check result (True/False/None)
            bbox_passed,      # Check result (True/False/None)
            volume_passed,    # Check result (True/False/None)
            hausdorff_passed, # Check result (True/False/None)
            chamfer_passed    # Check result (True/False/None)
        ]
        
        # Overall pass only if SCAD gen and render succeeded, AND
        # all subsequent checks that were *not* None resulted in True.
        if not scad_gen_success or not render_success:
            overall_passed = False
        else:
            # Filter out None values and check if all remaining are True
            applicable_checks = [c for c in required_checks[2:] if c is not None] # Skip first two, filter None
            overall_passed = all(applicable_checks) if applicable_checks else True # Pass if no checks applicable after render

        # --- Calculate boolean flags --- End ---

        dashboard_data.append({
            "task_id": entry.get("task_id", "N/A"),
            "model_name": entry.get("model_name", "N/A"),
            "prompt_key": entry.get("prompt_key_used", "default"),
            "replicate_id": entry.get("replicate_id", "N/A"),
            # --- Raw Replicate Data ---
            "render_status": render_status,
            "render_err": entry.get("render_error_message", "") or "",
            "gen_err": entry.get("generation_error", "") or "",
            "check_err": entry.get("check_error", "") or "",
            # --- Processed Boolean Flags ---
            "scad_generation_success": bool(scad_gen_success), # Explicit bool cast
            "checks_run_attempted": bool(checks_run_attempted), # Explicit bool cast
            "check_render_successful": bool(render_success), # Explicit bool cast
            # Checks below CAN be None if not run, so no bool() cast here
            "check_is_watertight": watertight,
            "check_is_single_component": single_comp,
            "check_bounding_box_accurate": bbox_passed,
            "check_volume_passed": volume_passed,
            "check_hausdorff_passed": hausdorff_passed,
            "chamfer_check_passed": chamfer_passed,
            "overall_passed": bool(overall_passed), # Explicit bool cast
            # --- Metrics (Formatted) ---
            "chamfer_dist": fmt(chamfer_dist),
            "haus_95p_dist": fmt(hausdorff_95p),
            "haus_99p_dist": fmt(entry.get("hausdorff_99p_distance")),
            "icp_fitness": fmt(entry.get("icp_fitness_score")),
            "ref_vol": fmt(ref_vol, 2),
            "gen_vol": fmt(gen_vol, 2),
            "render_duration": fmt(entry.get("render_duration_seconds"), 2),
            # --- Raw Details ---
            "ref_bbox": ref_bbox_val, # Use variable holding the value
            "gen_bbox": gen_bbox_val, # Use variable holding the value
            "scad_path": entry.get("output_scad_path", ""),
            "stl_path": entry.get("output_stl_path", ""),
            "ref_stl_path": entry.get("reference_stl_path", "")
        })

    logger.info(f"Processed {len(dashboard_data)} replicate entries for dashboard.")
    return dashboard_data

# --- New Aggregation Function --- Start ---
def aggregate_replicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregates results across replicates for each task/model combination."""
    logger.info(f"Aggregating results across {len(results)} replicate entries...")
    grouped_results = defaultdict(list)
    for entry in results:
        key = (entry.get("task_id"), entry.get("model_name"))
        if key[0] and key[1]: # Ensure both task_id and model_name are present
            grouped_results[key].append(entry)

    aggregated_data = []
    for (task_id, model_name), replicates in grouped_results.items():
        num_replicates = len(replicates)
        agg_entry = {
            "task_id": task_id,
            "model_name": model_name,
            "num_replicates": num_replicates,
            # Aggregate counts
            "gen_success_count": sum(1 for r in replicates if r.get("generation_error") is None),
            "render_success_count": sum(1 for r in replicates if r.get("render_status") == "Success"),
            "checks_run_count": sum(1 for r in replicates if r.get("render_status") == "Success"), # Checks only run if render succeeds
            "watertight_count": sum(1 for r in replicates if r["checks"].get("check_is_watertight") is True),
            "single_comp_count": sum(1 for r in replicates if r["checks"].get("check_is_single_component") is True),
            "bbox_pass_count": sum(1 for r in replicates if r["checks"].get("check_bounding_box_accurate") is True),
            "volume_pass_count": sum(1 for r in replicates if r["checks"].get("check_volume_passed") is True),
            "hausdorff_pass_count": sum(1 for r in replicates if r["checks"].get("check_hausdorff_passed") is True),
            "overall_pass_count": sum(1 for r in replicates if r.get("overall_passed") is True), # Assumes process_data added this
            # Store first entry's descriptive info (assuming it's the same for all reps)
            "task_description": replicates[0].get("task_description"),
            "reference_stl_path": replicates[0].get("reference_stl_path"),
            "llm_config": replicates[0].get("llm_config"),
            # Add lists to store numeric values for stats calculation
            "metrics": defaultdict(list)
        }

        # Collect numeric metrics across replicates where checks ran
        for r in replicates:
            # Only consider metrics if checks were attempted (render successful)
            if r.get("render_status") == "Success":
                # Helper to safely convert to float, return None if invalid/None
                def safe_float(val):
                    if val is None: return None
                    try: return float(val)
                    except (ValueError, TypeError): return None
                
                metrics_to_collect = [
                    "geometric_similarity_distance", "icp_fitness_score", 
                    "hausdorff_95p_distance", "hausdorff_99p_distance",
                    "reference_volume_mm3", "generated_volume_mm3",
                    "render_duration_seconds"
                ]
                for key in metrics_to_collect:
                    val = safe_float(r.get(key))
                    if val is not None:
                        agg_entry["metrics"][key].append(val)
                
                # Calculate volume diff % for stats
                ref_vol = safe_float(r.get("reference_volume_mm3"))
                gen_vol = safe_float(r.get("generated_volume_mm3"))
                if ref_vol is not None and gen_vol is not None and abs(ref_vol) > 1e-6:
                    vol_diff_pct = (abs(gen_vol - ref_vol) / abs(ref_vol)) * 100
                    agg_entry["metrics"]["volume_diff_percent"].append(vol_diff_pct)

        # Calculate statistics (mean, median, stdev) for collected metrics
        for key, values in agg_entry["metrics"].items():
            if values:
                try: agg_entry[f"avg_{key}"] = statistics.mean(values)
                except statistics.StatisticsError: agg_entry[f"avg_{key}"] = None
                try: agg_entry[f"median_{key}"] = statistics.median(values)
                except statistics.StatisticsError: agg_entry[f"median_{key}"] = None
                if len(values) > 1:
                    try: agg_entry[f"stdev_{key}"] = statistics.stdev(values)
                    except statistics.StatisticsError: agg_entry[f"stdev_{key}"] = None
                else:
                    agg_entry[f"stdev_{key}"] = 0.0 # Or None? Stdev is 0 for single point
            else:
                 agg_entry[f"avg_{key}"] = None
                 agg_entry[f"median_{key}"] = None
                 agg_entry[f"stdev_{key}"] = None

        # Remove the raw metrics list
        del agg_entry["metrics"]
        
        aggregated_data.append(agg_entry)

    logger.info(f"Aggregated into {len(aggregated_data)} task/model combinations.")
    return aggregated_data
# --- New Aggregation Function --- End ---

# --- New Summary Statistics Calculation Function --- Start ---
def calculate_summary_statistics(processed_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Calculates model-level (per prompt) and task-level summary statistics."""
    logger.info("Calculating summary statistics for models (per prompt) and tasks...")
    # Model stats need to be nested by model_name first, then prompt_key
    model_stats = defaultdict(lambda: defaultdict(lambda: {
        "total_replicates": 0,
        "scad_generation_success_count": 0,
        "render_success_count": 0,
        "checks_run_count": 0,
        "overall_pass_count": 0,
        "watertight_pass_count": 0,
        "single_comp_pass_count": 0,
        "bbox_acc_pass_count": 0,
        "volume_pass_count": 0,
        "hausdorff_pass_count": 0,
        "chamfer_pass_count": 0,
        "metrics": defaultdict(list) # Store valid numeric metrics
    }))
    task_stats = defaultdict(lambda: {
        "total_replicates": 0,
        "scad_generation_success_count": 0,
        "render_success_count": 0,
        "checks_run_count": 0,
        "overall_pass_count": 0,
        "watertight_pass_count": 0,
        "single_comp_pass_count": 0,
        "bbox_acc_pass_count": 0,
        "volume_pass_count": 0,
        "hausdorff_pass_count": 0,
        "chamfer_pass_count": 0,
        "metrics": defaultdict(list)
    })

    metrics_to_aggregate = ["chamfer_dist", "haus_95p_dist"] # Add others like "icp_fitness" if needed

    # --- Accumulate Counts and Metrics ---
    for entry in processed_data:
        model_name = entry.get("model_name")
        task_id = entry.get("task_id")
        prompt_key = entry.get("prompt_key", "default") # Get the prompt key used

        if not model_name or not task_id:
            logger.warning(f"Skipping entry with missing model_name or task_id: {entry}")
            continue

        # Get stats dict for the specific model AND prompt
        m_stat = model_stats[model_name][prompt_key]
        t_stat = task_stats[task_id] # Task stats don't need prompt key nesting

        # Increment total counts
        m_stat["total_replicates"] += 1
        t_stat["total_replicates"] += 1

        # Increment success counts based on boolean flags
        if entry.get("scad_generation_success") is True:
            m_stat["scad_generation_success_count"] += 1
            t_stat["scad_generation_success_count"] += 1
        if entry.get("check_render_successful") is True:
            m_stat["render_success_count"] += 1
            t_stat["render_success_count"] += 1
        if entry.get("overall_passed") is True:
            m_stat["overall_pass_count"] += 1
            t_stat["overall_pass_count"] += 1

        # Increment check-related counts only if checks were attempted
        checks_attempted = entry.get("checks_run_attempted") is True
        if checks_attempted:
            m_stat["checks_run_count"] += 1
            t_stat["checks_run_count"] += 1
            if entry.get("check_is_watertight") is True:
                m_stat["watertight_pass_count"] += 1
                t_stat["watertight_pass_count"] += 1
            if entry.get("check_is_single_component") is True:
                m_stat["single_comp_pass_count"] += 1
                t_stat["single_comp_pass_count"] += 1
            if entry.get("check_bounding_box_accurate") is True:
                m_stat["bbox_acc_pass_count"] += 1
                t_stat["bbox_acc_pass_count"] += 1
            if entry.get("check_volume_passed") is True:
                m_stat["volume_pass_count"] += 1
                t_stat["volume_pass_count"] += 1
            if entry.get("check_hausdorff_passed") is True:
                m_stat["hausdorff_pass_count"] += 1
                t_stat["hausdorff_pass_count"] += 1
            if entry.get("chamfer_check_passed") is True:
                m_stat["chamfer_pass_count"] += 1
                t_stat["chamfer_pass_count"] += 1

            # Collect numeric metrics if checks were attempted
            for metric_key in metrics_to_aggregate:
                val_str = entry.get(metric_key)
                if val_str is not None and val_str not in ["N/A", "Inf", "NaN"]:
                    try:
                        val_float = float(val_str)
                        m_stat["metrics"][metric_key].append(val_float)
                        t_stat["metrics"][metric_key].append(val_float)
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert metric '{metric_key}' value '{val_str}' to float for stats.")

    # --- Calculate Rates and Statistics --- Need to handle nested model_stats
    def calculate_final_stats(stat_dict, is_nested=False):
        final_stats_result = {}
        for primary_key, primary_value in stat_dict.items():
            if is_nested:
                # Process nested dictionary (model -> prompt -> stats)
                nested_final_stats = {}
                for prompt_key, stats in primary_value.items():
                    # Process the individual stat dict (same logic as before)
                    total_reps = stats["total_replicates"]
                    checks_run = stats["checks_run_count"]
                    final = stats.copy()
                    # Calculate rates
                    final["scad_generation_success_rate"] = (stats["scad_generation_success_count"] / total_reps * 100) if total_reps > 0 else 0
                    final["render_success_rate"] = (stats["render_success_count"] / total_reps * 100) if total_reps > 0 else 0
                    final["overall_pass_rate"] = (stats["overall_pass_count"] / total_reps * 100) if total_reps > 0 else 0
                    final["watertight_pass_rate"] = (stats["watertight_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                    final["single_comp_pass_rate"] = (stats["single_comp_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                    final["bbox_acc_pass_rate"] = (stats["bbox_acc_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                    final["volume_pass_rate"] = (stats["volume_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                    final["hausdorff_pass_rate"] = (stats["hausdorff_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                    final["chamfer_pass_rate"] = (stats["chamfer_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                    # Calculate metric statistics
                    for metric, values in stats["metrics"].items():
                        metric_name = metric.replace('_dist', '').replace('haus_','hausdorff_')
                        if values:
                            try: final[f"avg_{metric_name}"] = statistics.mean(values)
                            except statistics.StatisticsError: final[f"avg_{metric_name}"] = None
                            try: final[f"median_{metric_name}"] = statistics.median(values)
                            except statistics.StatisticsError: final[f"median_{metric_name}"] = None
                            if len(values) > 1:
                                try: final[f"stdev_{metric_name}"] = statistics.stdev(values)
                                except statistics.StatisticsError: final[f"stdev_{metric_name}"] = None
                            else:
                                final[f"stdev_{metric_name}"] = 0.0
                        else:
                             final[f"avg_{metric_name}"] = None
                             final[f"median_{metric_name}"] = None
                             final[f"stdev_{metric_name}"] = None
                    del final["metrics"]
                    nested_final_stats[prompt_key] = final
                final_stats_result[primary_key] = nested_final_stats
            else:
                # Process flat dictionary (task -> stats)
                stats = primary_value
                total_reps = stats["total_replicates"]
                checks_run = stats["checks_run_count"]
                final = stats.copy()
                # Calculate rates
                final["scad_generation_success_rate"] = (stats["scad_generation_success_count"] / total_reps * 100) if total_reps > 0 else 0
                final["render_success_rate"] = (stats["render_success_count"] / total_reps * 100) if total_reps > 0 else 0
                final["overall_pass_rate"] = (stats["overall_pass_count"] / total_reps * 100) if total_reps > 0 else 0
                final["watertight_pass_rate"] = (stats["watertight_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                final["single_comp_pass_rate"] = (stats["single_comp_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                final["bbox_acc_pass_rate"] = (stats["bbox_acc_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                final["volume_pass_rate"] = (stats["volume_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                final["hausdorff_pass_rate"] = (stats["hausdorff_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                final["chamfer_pass_rate"] = (stats["chamfer_pass_count"] / checks_run * 100) if checks_run > 0 else 0
                # Calculate metric statistics
                for metric, values in stats["metrics"].items():
                    metric_name = metric.replace('_dist', '').replace('haus_','hausdorff_')
                    if values:
                        try: final[f"avg_{metric_name}"] = statistics.mean(values)
                        except statistics.StatisticsError: final[f"avg_{metric_name}"] = None
                        try: final[f"median_{metric_name}"] = statistics.median(values)
                        except statistics.StatisticsError: final[f"median_{metric_name}"] = None
                        if len(values) > 1:
                            try: final[f"stdev_{metric_name}"] = statistics.stdev(values)
                            except statistics.StatisticsError: final[f"stdev_{metric_name}"] = None
                        else:
                            final[f"stdev_{metric_name}"] = 0.0
                    else:
                         final[f"avg_{metric_name}"] = None
                         final[f"median_{metric_name}"] = None
                         final[f"stdev_{metric_name}"] = None
                del final["metrics"]
                final_stats_result[primary_key] = final
        return final_stats_result

    # Pass is_nested=True for model stats
    final_model_stats = calculate_final_stats(model_stats, is_nested=True)
    final_task_stats = calculate_final_stats(task_stats, is_nested=False)

    logger.info(f"Finished calculating summary statistics for {len(final_model_stats)} models (across prompts) and {len(final_task_stats)} tasks.")
    return final_model_stats, final_task_stats
# --- New Summary Statistics Calculation Function --- End ---

# --- Saving Results ---
def save_dashboard_data(data: List[Dict[str, Any]], output_path: str):
    """Saves the processed data as a JSON file for the dashboard."""
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created dashboard output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create directory {output_dir}: {e}")
            return # Cannot save if directory cannot be created
            
    json_content = json.dumps(data, indent=4)
    try:
        with open(output_path, 'w') as f:
            f.write(json_content)
        logger.info(f"Dashboard data successfully saved to {output_path}")
    except IOError as e:
        logger.error(f"Error writing dashboard data to {output_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving dashboard data: {e}")

def main(args):
    # Load configuration
    config = load_config()

    # Load results
    logger.info(f"Loading results from: {args.results_path}")
    raw_results = load_results(args.results_path)
    if not raw_results:
        logger.error("No results found or failed to load results.")
        sys.exit(1)

    # --- Process for Dashboard (Individual Replicates) ---
    logger.info("Processing data for dashboard...")
    dashboard_data = process_data_for_dashboard(raw_results, config)
    if not dashboard_data:
        logger.warning("Processing for dashboard yielded no data.")
        # Still continue to try and create empty stats if needed
        
    # --- Calculate Summary Statistics --- Start ---
    meta_stats, task_stats = {}, {} # Initialize empty in case processing failed
    if dashboard_data:
        try:
            meta_stats, task_stats = calculate_summary_statistics(dashboard_data)
        except Exception as e:
             logger.error(f"Failed to calculate summary statistics: {e}", exc_info=True)
             # Keep meta_stats and task_stats as empty dicts
    # --- Calculate Summary Statistics --- End ---


    # --- Restructure data for dashboard JS --- Start ---
    logger.info("Restructuring data for the expected dashboard format...")
    results_by_model = defaultdict(list)
    # Group the already processed dashboard_data (flat list)
    for entry in dashboard_data:
        model_name = entry.get("model_name", "unknown_model")
        results_by_model[model_name].append(entry)
    
    # Try to extract run_id from the input path
    run_id = "unknown_run"
    try:
        path_part = os.path.basename(args.results_path)
        parent_dir_name = os.path.basename(os.path.dirname(args.results_path))
        # Prioritize parent dir name if it looks like a run ID (e.g., contains digits)
        if any(char.isdigit() for char in parent_dir_name) and parent_dir_name != 'results':
             run_id = parent_dir_name
        elif os.path.isdir(args.results_path):
             run_id = path_part # Use directory name as fallback
        else:
             base_name = os.path.splitext(path_part)[0]
             if base_name.startswith("results_"):
                  run_id = base_name.split("_", 1)[1]
             elif "_" in base_name and any(char.isdigit() for char in base_name.split("_")[0]):
                  run_id = base_name.split("_")[0] # Guess it's the first part if it looks like ID
             else:
                  run_id = base_name # Fallback
        logger.info(f"Extracted run_id: {run_id}")
    except Exception as e:
        logger.warning(f"Could not automatically extract run_id from path '{args.results_path}': {e}")

    final_dashboard_output = {
        "run_id": run_id,
        "meta_statistics": meta_stats, # Use calculated stats
        "task_statistics": task_stats, # Add calculated task stats
        "results_by_model": dict(results_by_model)
    }
    # --- Restructure data for dashboard JS --- End ---

    # --- Save Dashboard Data ---
    dashboard_json_path = "dashboard/dashboard_data.json"
    logger.info(f"Saving final structured dashboard data to: {dashboard_json_path}")
    save_dashboard_data(final_dashboard_output, dashboard_json_path)

    logger.info("Processing complete.")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process CadEval results for dashboard and stats.")
    parser.add_argument("results_path", help="Path to the results JSON file or directory.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level.")
    # Add other args if needed

    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(args)