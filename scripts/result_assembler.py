import os
import logging
from typing import List, Dict, Any, Optional

# --- Main Assembly Function ---

def assemble_final_results(
    generation_results: List[Dict[str, Any]], # SCAD generation results (non-Zoo)
    direct_stl_generation_results: List[Dict[str, Any]], # Direct STL results (Zoo)
    render_results: List[Dict[str, Any]], # Results from rendering SCAD files
    check_results_map: Dict[str, Dict[str, Any]], # Keyed by actual STL path
    # stl_path_to_gen_info: Dict[str, Dict[str, Any]], # REMOVED - No longer needed
    logger: logging.Logger,
    project_root: str # Added project_root argument
) -> List[Dict[str, Any]]:
    """
    Merges results from all stages (generation, rendering, checking) into the final schema,
    handling both standard LLM->SCAD->STL and direct Zoo->STL workflows.
    Iterates through original generation results to ensure all attempts are included.
    """
    logger.info("Assembling final results from original generation lists...")

    # --- Define make_relative helper function within scope ---
    def make_relative(path: str | None) -> str | None:
        if path and os.path.isabs(path):
            try: return os.path.relpath(path, project_root)
            except ValueError: return path # Return absolute if on different drive (Windows)
        return path # Return as is if already relative or None
    # ---

    final_results_list = []
    # REMOVED: processed_gen_indices = set()

    # --- Create simple lookup map for render results ---
    # Includes successful and failed renders
    render_results_map = {res.get("scad_path"): res for res in render_results if res.get("scad_path")}
    # ---

    # --- Process Non-Zoo (LLM -> SCAD -> STL) Results ---
    logger.info(f"Processing {len(generation_results)} non-Zoo generation results...")
    for gen_result in generation_results:
        # --- Extract common info ---
        task_id = gen_result.get("task_id")
        model_config_used = gen_result.get("model_config_used", {})
        model_name = model_config_used.get("name")
        provider = model_config_used.get("provider")
        prompt_key = gen_result.get("prompt_key_used")
        replicate_id = gen_result.get("replicate_id")
        task_data = gen_result.get("task_data", {}) # Contains reference_stl, description
        gen_success = gen_result.get("success")
        gen_error = gen_result.get("error")
        scad_path = gen_result.get("output_path") # SCAD path from generation
        llm_duration = gen_result.get("llm_duration_seconds")
        prompt_tokens = gen_result.get("prompt_tokens")
        completion_tokens = gen_result.get("completion_tokens")

        # --- Basic Validation ---
        if None in [task_id, model_name, provider, prompt_key, replicate_id]:
             logger.error(f"Cannot assemble entry due to missing key info in non-Zoo generation result: {gen_result}")
             continue

        # --- Initialize the final entry ---
        final_entry = {
            "task_id": task_id, "replicate_id": replicate_id, "model_name": model_name,
            "provider": provider, "prompt_key_used": prompt_key,
            "task_description": task_data.get("description"),
            "reference_stl_path": make_relative(task_data.get("reference_stl")),
            "prompt_used": gen_result.get("prompt_used"),
            "llm_config": {
                 "provider": provider, "name": model_name,
                 "temperature": model_config_used.get("temperature"),
                 "max_tokens": model_config_used.get("max_tokens"),
                 "cli_args": None, # Not applicable for non-Zoo
             },
            "timestamp_utc": gen_result.get("timestamp"),
            "output_scad_path": make_relative(scad_path) if gen_success else None, # Only if gen succeeded
            "output_stl_path": None,
            "output_summary_json_path": None,
            "render_status": None,
            "render_duration_seconds": None,
            "render_error_message": None,
            "checks": { key: None for key in [ # Initialize all checks to None
                "check_render_successful", "check_is_watertight", "check_is_single_component",
                "check_bounding_box_accurate", "check_volume_passed", "check_hausdorff_passed",
                "check_chamfer_passed"
            ]},
            "geometric_similarity_distance": None, "icp_fitness_score": None,
            "hausdorff_95p_distance": None, "hausdorff_99p_distance": None,
            "reference_volume_mm3": None, "generated_volume_mm3": None,
            "reference_bbox_mm": None, "generated_bbox_aligned_mm": None,
            "generation_error": gen_error,
            "check_error": None,
            "llm_duration_seconds": llm_duration,
            "generation_duration_seconds": None, # Not applicable for non-Zoo
            "render_duration_seconds": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost": None # Calculated below
        }

        # --- Handle Generation Failure ---
        if not gen_success:
            final_entry["render_status"] = "Not Run"
            final_entry["check_error"] = "Checks not run due to generation failure."
            # Cost calculation skipped for gen failures currently
            final_results_list.append(final_entry)
            continue # Move to next generation result

        # --- Handle Generation Success - Process Rendering & Checks ---
        render_info = render_results_map.get(scad_path)
        render_success = False
        stl_path = None

        if render_info:
            render_status = render_info.get("status")
            render_success = (render_status == "Success")
            final_entry["render_status"] = render_status
            final_entry["render_duration_seconds"] = render_info.get("duration")
            final_entry["render_error_message"] = render_info.get("error")
            final_entry["checks"]["check_render_successful"] = render_success
            if render_success:
                stl_path = render_info.get("stl_path") # Get the actual STL path if render succeeded
                final_entry["output_stl_path"] = make_relative(stl_path)
                final_entry["output_summary_json_path"] = make_relative(render_info.get("summary_path"))
            else:
                # Render failed, checks cannot run based on render output
                final_entry["check_error"] = f"Checks not run due to render status: {render_status}"
                # Ensure check booleans reflect non-execution (apart from render check)
                for key in final_entry["checks"]:
                     if key != "check_render_successful":
                         final_entry["checks"][key] = None # Or False? Let's use None for clarity
        else:
            # SCAD generated, but no render info found (unexpected)
            logger.warning(f"Missing render info for successfully generated SCAD: {scad_path}")
            final_entry["render_status"] = "Error"
            final_entry["render_error_message"] = "Render metadata missing (possible orchestration error)"
            final_entry["checks"]["check_render_successful"] = False
            final_entry["check_error"] = "Checks not run due to missing render metadata."
            # Ensure check booleans reflect non-execution
            for key in final_entry["checks"]:
                if key != "check_render_successful":
                    final_entry["checks"][key] = None

        # --- Populate Checks if Render Succeeded ---
        if render_success and stl_path:
            check_info = check_results_map.get(stl_path)
            if check_info:
                checks_sub_dict = check_info.get("checks", {})
                for check_key in final_entry["checks"]:
                     if check_key == "check_render_successful": continue # Keep render status
                     check_value = checks_sub_dict.get(check_key)
                     # Convert to bool if not None, otherwise keep None
                     final_entry["checks"][check_key] = bool(check_value) if check_value is not None else None

                # Populate metrics
                final_entry["geometric_similarity_distance"] = check_info.get("geometric_similarity_distance")
                final_entry["icp_fitness_score"] = check_info.get("icp_fitness_score")
                final_entry["hausdorff_95p_distance"] = check_info.get("hausdorff_95p_distance")
                final_entry["hausdorff_99p_distance"] = check_info.get("hausdorff_99p_distance")
                final_entry["reference_volume_mm3"] = check_info.get("reference_volume_mm3")
                final_entry["generated_volume_mm3"] = check_info.get("generated_volume_mm3")
                final_entry["reference_bbox_mm"] = check_info.get("reference_bbox_mm")
                final_entry["generated_bbox_aligned_mm"] = check_info.get("generated_bbox_aligned_mm")

                # Populate check errors
                check_run_error = check_info.get("error")
                check_internal_errors = check_info.get("check_errors", [])
                all_errors = []
                if check_run_error: all_errors.append(f"Check Run Error: {check_run_error}")
                if check_internal_errors: all_errors.extend([f"Check Detail Error: {e}" for e in check_internal_errors])
                if all_errors:
                    final_entry["check_error"] = "; ".join(all_errors)
                    logger.warning(f"Check issues for {os.path.basename(stl_path or 'UNKNOWN')}: {final_entry['check_error']}")
            else:
                 # Render OK but no check info found (unexpected)
                 logger.error(f"Render successful but no check info for {stl_path}. Checks did not run or failed to record.")
                 final_entry["check_error"] = "Check results missing (setup/orchestration error?)"
                 # Ensure check booleans reflect non-execution
                 for key in final_entry["checks"]:
                     if key != "check_render_successful":
                         final_entry["checks"][key] = None

        # --- Calculate LLM Cost (can happen regardless of render/check outcome) ---
        estimated_cost = None
        cost_input_M = model_config_used.get("cost_per_million_input_tokens")
        cost_output_M = model_config_used.get("cost_per_million_output_tokens")
        if prompt_tokens is not None and completion_tokens is not None and cost_input_M is not None and cost_output_M is not None:
            try:
                cost_input_f = float(cost_input_M) / 1_000_000.0
                cost_output_f = float(cost_output_M) / 1_000_000.0
                prompt_tokens_f = float(prompt_tokens)
                completion_tokens_f = float(completion_tokens)
                estimated_cost = (prompt_tokens_f * cost_input_f) + (completion_tokens_f * cost_output_f)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid cost or token parameter for LLM model '{model_name}': {e}. Cannot calculate cost.")
        # Don't log warning if cost params are just missing (handled elsewhere if needed)
        final_entry["estimated_cost"] = estimated_cost

        # --- Append the processed non-Zoo entry ---
        final_results_list.append(final_entry)

    # --- Process Zoo (Direct STL) Results ---
    logger.info(f"Processing {len(direct_stl_generation_results)} Zoo generation results...")
    for gen_result in direct_stl_generation_results:
        # --- Extract common info ---
        task_id = gen_result.get("task_id")
        model_config_used = gen_result.get("model_config_used", {})
        model_name = model_config_used.get("name") # Zoo uses name directly
        provider = model_config_used.get("provider")
        prompt_key = gen_result.get("prompt_key_used")
        replicate_id = gen_result.get("replicate_id")
        task_data = gen_result.get("task_data", {})
        gen_success = gen_result.get("success")
        gen_error = gen_result.get("error")
        stl_path = gen_result.get("output_stl_path") # Direct STL path
        generation_duration = gen_result.get("generation_duration_seconds")

        # --- Basic Validation ---
        if None in [task_id, model_name, provider, prompt_key, replicate_id]:
             logger.error(f"Cannot assemble entry due to missing key info in Zoo generation result: {gen_result}")
             continue

        # --- Initialize the final entry ---
        final_entry = {
            "task_id": task_id, "replicate_id": replicate_id, "model_name": model_name,
            "provider": provider, "prompt_key_used": prompt_key,
            "task_description": task_data.get("description"),
            "reference_stl_path": make_relative(task_data.get("reference_stl")),
            "prompt_used": gen_result.get("prompt_used"), # Zoo might have prompt used
             "llm_config": { # Populate from model_config_used
                 "provider": provider, "name": model_name,
                 "temperature": None, # Not applicable
                 "max_tokens": None, # Not applicable
                 "cli_args": model_config_used.get("cli_args"),
             },
            "timestamp_utc": gen_result.get("timestamp"),
            "output_scad_path": None, # Not applicable
            "output_stl_path": make_relative(stl_path) if gen_success else None,
            "output_summary_json_path": None, # Not applicable
            "render_status": "N/A", # Not applicable
            "render_duration_seconds": None, # Not applicable
            "render_error_message": None, # Not applicable
            "checks": { key: None for key in [ # Initialize checks to None
                "check_render_successful", "check_is_watertight", "check_is_single_component",
                "check_bounding_box_accurate", "check_volume_passed", "check_hausdorff_passed",
                "check_chamfer_passed"
            ]},
            "geometric_similarity_distance": None, "icp_fitness_score": None,
            "hausdorff_95p_distance": None, "hausdorff_99p_distance": None,
            "reference_volume_mm3": None, "generated_volume_mm3": None,
            "reference_bbox_mm": None, "generated_bbox_aligned_mm": None,
            "generation_error": gen_error,
            "check_error": None,
            "llm_duration_seconds": None, # Not applicable
            "generation_duration_seconds": generation_duration,
            "render_duration_seconds": None, # Not applicable
            "prompt_tokens": None, # Not applicable
            "completion_tokens": None, # Not applicable
            "estimated_cost": None # Calculated below
        }
        final_entry["checks"]["check_render_successful"] = None # Explicitly None for Zoo

        # --- Handle Generation Failure ---
        if not gen_success:
            final_entry["check_error"] = "Checks not run due to generation failure."
            # Cost calculation skipped for gen failures currently
            final_results_list.append(final_entry)
            continue

        # --- Handle Generation Success - Process Checks ---
        check_info = check_results_map.get(stl_path)
        if check_info:
            checks_sub_dict = check_info.get("checks", {})
            for check_key in final_entry["checks"]:
                 if check_key == "check_render_successful": continue # Keep None for Zoo
                 check_value = checks_sub_dict.get(check_key)
                 final_entry["checks"][check_key] = bool(check_value) if check_value is not None else None

            # Populate metrics
            final_entry["geometric_similarity_distance"] = check_info.get("geometric_similarity_distance")
            final_entry["icp_fitness_score"] = check_info.get("icp_fitness_score")
            final_entry["hausdorff_95p_distance"] = check_info.get("hausdorff_95p_distance")
            final_entry["hausdorff_99p_distance"] = check_info.get("hausdorff_99p_distance")
            final_entry["reference_volume_mm3"] = check_info.get("reference_volume_mm3")
            final_entry["generated_volume_mm3"] = check_info.get("generated_volume_mm3")
            final_entry["reference_bbox_mm"] = check_info.get("reference_bbox_mm")
            final_entry["generated_bbox_aligned_mm"] = check_info.get("generated_bbox_aligned_mm")

            # Populate check errors
            check_run_error = check_info.get("error")
            check_internal_errors = check_info.get("check_errors", [])
            all_errors = []
            if check_run_error: all_errors.append(f"Check Run Error: {check_run_error}")
            if check_internal_errors: all_errors.extend([f"Check Detail Error: {e}" for e in check_internal_errors])
            if all_errors:
                final_entry["check_error"] = "; ".join(all_errors)
                logger.warning(f"Check issues for {os.path.basename(stl_path or 'UNKNOWN')}: {final_entry['check_error']}")
        else:
             # Zoo Gen OK but no check info found (unexpected)
             logger.error(f"Zoo generation successful but no check info for {stl_path}. Checks did not run or failed to record.")
             final_entry["check_error"] = "Check results missing (setup/orchestration error?)"
             # Ensure check booleans reflect non-execution
             for key in final_entry["checks"]:
                 if key != "check_render_successful":
                     final_entry["checks"][key] = None

        # --- Calculate Zoo Cost ---
        estimated_cost = None
        if generation_duration is not None:
            cost_per_minute = model_config_used.get("cost_per_minute")
            free_tier_seconds = model_config_used.get("free_tier_seconds", 0)
            try:
                if cost_per_minute is not None:
                    cost_per_minute_f = float(cost_per_minute)
                    free_tier_seconds_f = float(free_tier_seconds)
                    generation_duration_f = float(generation_duration)
                    billable_seconds = max(0, generation_duration_f - free_tier_seconds_f)
                    estimated_cost = (billable_seconds / 60.0) * cost_per_minute_f
                else:
                     logger.debug(f"Missing 'cost_per_minute' in config for zoo_cli model '{model_name}'. Cannot calculate cost.") # Debug instead of warn
            except (ValueError, TypeError) as e:
                 logger.warning(f"Invalid cost parameter for zoo_cli model '{model_name}': {e}. Cannot calculate cost.")
        final_entry["estimated_cost"] = estimated_cost

        # --- Append the processed Zoo entry ---
        final_results_list.append(final_entry)

    # REMOVED: Loop checking processed_gen_indices - no longer needed

    logger.info(f"Assembled {len(final_results_list)} final result entries.")
    # Sort for consistent output order
    final_results_list.sort(key=lambda x: (
        x.get("task_id", ""),         # Default to empty string
        x.get("provider", ""),        # Default to empty string
        x.get("model_name", ""),       # Default to empty string
        x.get("prompt_key_used", ""), # Default to empty string
        x.get("replicate_id", 0)      # Default to 0
    ))
    return final_results_list
