# scripts/process_results.py

import json
import argparse
import os
import statistics
from collections import defaultdict
import re # For extracting run ID
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_individual_result(result_entry):
    """Processes a single entry from the results.json list."""
    processed = result_entry.copy() # Start with original data

    # Initialize flags and statuses
    scad_generation_success = False
    render_success = False
    geometry_check_orchestration_success = False
    all_individual_checks_passed = False
    overall_pipeline_success = False
    individual_check_statuses = {} # Use this to store True/False/None/N/A

    # 1. SCAD Generation Success
    scad_generation_success = bool(result_entry.get("output_scad_path")) and result_entry.get("generation_error") is None
    processed['scad_generation_success'] = scad_generation_success

    # 2. Render Success
    if scad_generation_success:
        render_status = result_entry.get("render_status")
        render_success = render_status == "Success"
        processed['render_success'] = render_success
        # Add render status for detail table
        processed['render_status_detail'] = render_status if render_status else "N/A"
    else:
        processed['render_success'] = False
        processed['render_status_detail'] = "N/A (SCAD Fail)"


    # 3. Geometry Check Orchestration Success
    # Checks should only run if rendering was successful and there wasn't a check orchestration error
    if render_success:
        check_error = result_entry.get("check_error")
        geometry_check_orchestration_success = check_error is None
        processed['geometry_check_orchestration_success'] = geometry_check_orchestration_success
        processed['geometry_check_error_detail'] = check_error # Store potential error message
    else:
        processed['geometry_check_orchestration_success'] = False
        processed['geometry_check_error_detail'] = None # No error if it didn't run

    # 4. Individual Geometry Checks Passed & Statuses
    checks_data = result_entry.get("checks", {})
    check_keys_expected = ["check_render_successful", "check_is_watertight", "check_is_single_component", "check_bounding_box_accurate"] # Define expected keys

    if geometry_check_orchestration_success and checks_data is not None:
        # If checks_data is None (JSON null), treat as failed orchestration
        if checks_data is None:
            logging.warning(f"Task {result_entry.get('task_id')}/{result_entry.get('model_name')}: 'checks' field is null despite successful render & no check_error.")
            processed['geometry_check_orchestration_success'] = False
            geometry_check_orchestration_success = False # Update local flag too
            all_individual_checks_passed = False
            # Set all expected checks to N/A due to this inconsistency
            for key in check_keys_expected:
                 individual_check_statuses[key] = "N/A (Null Checks)"

        elif isinstance(checks_data, dict):
            all_passed_flag = True
            found_any_check = False
            for key in check_keys_expected:
                check_result = checks_data.get(key) # Use .get() to handle missing keys gracefully
                individual_check_statuses[key] = check_result # Store True, False, or None
                if check_result is not None:
                    found_any_check = True
                # A check only passes if it's explicitly True
                if check_result is not True:
                    all_passed_flag = False
            # Only consider "all passed" if the orchestration succeeded AND we found at least one check result
            all_individual_checks_passed = all_passed_flag and found_any_check
        else:
             # checks_data is not a dict or None, indicates an issue
             logging.warning(f"Task {result_entry.get('task_id')}/{result_entry.get('model_name')}: 'checks' field has unexpected type: {type(checks_data)}. Treating as check failure.")
             processed['geometry_check_orchestration_success'] = False
             geometry_check_orchestration_success = False # Update local flag
             all_individual_checks_passed = False
             for key in check_keys_expected:
                  individual_check_statuses[key] = "N/A (Bad Type)"

    else: # Checks didn't run or orchestration failed
        all_individual_checks_passed = False
        status_reason = "N/A (Orchestration Fail)" if render_success else "N/A (Render Fail)"
        for key in check_keys_expected:
            individual_check_statuses[key] = status_reason # Indicate why they weren't evaluated

    processed['individual_geometry_checks_passed'] = all_individual_checks_passed
    processed['individual_check_statuses'] = individual_check_statuses

    # 5. Overall Pipeline Success (Recalculate based on potentially updated flags)
    overall_pipeline_success = (scad_generation_success and
                              render_success and
                              geometry_check_orchestration_success and
                              all_individual_checks_passed)
    processed['overall_pipeline_success'] = overall_pipeline_success

    # Format similarity distance
    sim_dist = processed.get('geometric_similarity_distance')
    processed['similarity_distance_detail'] = f"{sim_dist:.4f}" if sim_dist is not None else "N/A"


    return processed


def calculate_meta_statistics(processed_results):
    """Calculates aggregate statistics grouped by model."""
    stats_by_model = defaultdict(lambda: defaultdict(int))
    similarity_scores_by_model = defaultdict(list)

    # --- Tally results per model ---
    for result in processed_results:
        model_name = result.get("model_name", "Unknown Model")
        stats = stats_by_model[model_name]

        stats['total_tasks'] += 1
        if result.get('scad_generation_success'):
            stats['scad_gen_success_count'] += 1
        if result.get('render_success'):
            stats['render_success_count'] += 1
        if result.get('geometry_check_orchestration_success'):
            stats['geo_check_run_success_count'] += 1
        if result.get('individual_geometry_checks_passed'):
            stats['all_geo_checks_passed_count'] += 1
        if result.get('overall_pipeline_success'):
            stats['overall_pipeline_success_count'] += 1
            # Collect similarity distance only for fully successful runs
            distance = result.get('geometric_similarity_distance')
            if distance is not None:
                try:
                    similarity_scores_by_model[model_name].append(float(distance))
                except (ValueError, TypeError):
                    logging.warning(f"Model {model_name}, Task {result.get('task_id')}: Invalid similarity distance '{distance}'")


    # --- Calculate rates and averages ---
    final_meta_stats = {}
    for model_name, counts in stats_by_model.items():
        total = counts['total_tasks']
        scad_success = counts['scad_gen_success_count']
        render_success = counts['render_success_count']
        check_run_success = counts['geo_check_run_success_count']
        all_checks_passed = counts['all_geo_checks_passed_count']
        overall_success = counts['overall_pipeline_success_count']

        model_stats = counts.copy() # Start with counts

        model_stats['scad_gen_success_rate'] = (scad_success / total * 100) if total > 0 else 0
        # Rate relative to previous successful step
        model_stats['render_success_rate_rel'] = (render_success / scad_success * 100) if scad_success > 0 else 0
        model_stats['geo_check_run_success_rate_rel'] = (check_run_success / render_success * 100) if render_success > 0 else 0
        model_stats['all_geo_checks_passed_rate_rel'] = (all_checks_passed / check_run_success * 100) if check_run_success > 0 else 0
        # Overall rate relative to total tasks
        model_stats['overall_pipeline_success_rate'] = (overall_success / total * 100) if total > 0 else 0

        # Calculate average similarity distance for fully successful runs
        valid_distances = similarity_scores_by_model.get(model_name, [])
        if valid_distances:
            model_stats['average_similarity_distance'] = statistics.mean(valid_distances)
            model_stats['successful_similarity_count'] = len(valid_distances)
            model_stats['median_similarity_distance'] = statistics.median(valid_distances)
            if len(valid_distances) > 1:
                model_stats['stdev_similarity_distance'] = statistics.stdev(valid_distances)
            else:
                model_stats['stdev_similarity_distance'] = 0
        else:
            model_stats['average_similarity_distance'] = None
            model_stats['successful_similarity_count'] = 0
            model_stats['median_similarity_distance'] = None
            model_stats['stdev_similarity_distance'] = None


        # Add absolute success counts for display
        model_stats['render_success_count'] = render_success
        model_stats['geo_check_run_success_count'] = check_run_success
        model_stats['all_geo_checks_passed_count'] = all_checks_passed
        model_stats['overall_pipeline_success_count'] = overall_success


        final_meta_stats[model_name] = model_stats

    return final_meta_stats

def extract_run_id(input_path):
    """Extracts the Run ID (e.g., YYYYMMDD_HHMMSS) from the input filename."""
    # Try matching pattern like results_YYYYMMDD_HHMMSS.json or run_YYYYMMDD_HHMMSS.log
    match = re.search(r'(\d{8}_\d{6})', os.path.basename(input_path))
    if match:
        return match.group(1)
    # Fallback if pattern doesn't match
    logging.warning(f"Could not extract standard run ID from path: {input_path}. Using fallback.")
    base = os.path.basename(input_path)
    base = base.replace("results_", "").replace("run_", "").replace(".json", "").replace(".log", "")
    return base if base else "unknown_run"

def main():
    parser = argparse.ArgumentParser(description="Process CadEval results.json for dashboard visualization.")
    parser.add_argument("--input", required=True, help="Path to the input results_RUNID.json file.")
    parser.add_argument("--output", default="dashboard_data.json", help="Path for the output dashboard_data.json file.")
    args = parser.parse_args()

    logging.info(f"Processing results from: {args.input}")

    if not os.path.exists(args.input):
        logging.error(f"Input file not found at {args.input}")
        return

    try:
        with open(args.input, 'r') as f:
            raw_results = json.load(f)
        if not isinstance(raw_results, list):
            logging.error(f"Error: Expected a JSON list in {args.input}, but got {type(raw_results)}")
            return

    except json.JSONDecodeError as e:
        logging.error(f"Error: Could not decode JSON from {args.input}: {e}")
        return
    except Exception as e:
         logging.error(f"An unexpected error occurred reading the input file: {e}")
         return


    # Process each result entry
    processed_results_list = []
    for i, entry in enumerate(raw_results):
        try:
             processed_results_list.append(process_individual_result(entry))
        except Exception as e:
             task_id = entry.get('task_id', 'unknown')
             model_name = entry.get('model_name', 'unknown')
             logging.error(f"Error processing entry {i} (Task: {task_id}, Model: {model_name}): {e}", exc_info=True)
             # Optionally add a placeholder or skip the entry

    # Group processed results by model for the dashboard tables
    results_by_model = defaultdict(list)
    for processed_entry in processed_results_list:
        model_name = processed_entry.get("model_name", "Unknown Model")
        # Sort tasks within each model for consistent table order? Optional.
        results_by_model[model_name].append(processed_entry)

    # Optional: Sort tasks within each model by task_id
    for model_name in results_by_model:
         results_by_model[model_name].sort(key=lambda x: x.get('task_id', ''))

    # Calculate meta-statistics
    meta_stats = calculate_meta_statistics(processed_results_list)

    # Extract Run ID
    run_id = extract_run_id(args.input)

    # Combine into final dashboard data structure
    dashboard_data = {
        "run_id": run_id,
        "meta_statistics": meta_stats,
        "results_by_model": dict(results_by_model) # Convert defaultdict to dict for JSON
    }

    # Save the dashboard data
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Error creating output directory {output_dir}: {e}")
            return # Cant save if dir creation fails


    try:
        with open(args.output, 'w') as f:
            json.dump(dashboard_data, f, indent=4)
        logging.info(f"Dashboard data successfully saved to: {args.output}")
    except IOError as e:
        logging.error(f"Error saving dashboard data to {args.output}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred saving the output file: {e}")


if __name__ == "__main__":
    main()