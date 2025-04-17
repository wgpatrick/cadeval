import os
import logging
from typing import List, Dict, Any

def assemble_final_results(
    generation_results: List[Dict[str, Any]], # SCAD generation results (non-Zoo)
    direct_stl_generation_results: List[Dict[str, Any]], # Direct STL results (Zoo)
    render_results: List[Dict[str, Any]], # Results from rendering SCAD files
    check_results_map: Dict[str, Dict[str, Any]], # Keyed by actual STL path
    stl_path_to_gen_info: Dict[str, Dict[str, Any]], # Keyed by expected/actual STL path
    logger: logging.Logger,
    project_root: str # Added project_root argument
) -> List[Dict[str, Any]]:
    """
    Merges results from all stages (generation, rendering, checking) into the final schema,
    handling both standard LLM->SCAD->STL and direct Zoo->STL workflows.
    Uses list filtering instead of complex key maps for generation results.
    """
    logger.info("Assembling final results...")

    # --- Define make_relative helper function within scope ---
    def make_relative(path: str | None) -> str | None:
        if path and os.path.isabs(path):
            try: return os.path.relpath(path, project_root)
            except ValueError: return path # Return absolute if on different drive (Windows)
        return path # Return as is if already relative or None
    # ---

    final_results_list = []
    processed_gen_indices = set() # Track indices of processed gen results

    # --- Create simple lookup map for render results ---
    render_results_map = {res.get("scad_path"): res for res in render_results if res.get("scad_path")}
    # ---

    # Iterate through the unified map which links STL paths back to generation info
    # This map only contains entries for which STL generation/rendering was *attempted* and potentially successful
    for stl_path, gen_info in stl_path_to_gen_info.items():
        # --- Flexible Model Config Handling ---
        model_config_data = gen_info.get("model_config_used") # Parallel script key
        if model_config_data is None:
            model_config_data = gen_info.get("model_config") # Serial script key
        if model_config_data is None:
            model_config_data = {} # Default to empty dict if neither found
            logger.warning(f"Missing 'model_config_used' or 'model_config' in gen_info for STL {stl_path}. Cost/config info may be incomplete.")

        # --- Extract provider/model consistently from the found config dict ---
        task_id = gen_info.get("task_id")
        model_name = model_config_data.get("name")
        provider = model_config_data.get("provider")
        is_zoo_run = (provider == 'zoo_cli')

        # --- Flexible SCAD Path Handling ---
        scad_path_from_gen_info = gen_info.get("output_path") # Parallel script key
        if scad_path_from_gen_info is None:
            scad_path_from_gen_info = gen_info.get("scad_path") # Serial script key

        # --- Extract other keys (assuming these are consistent) ---
        prompt_key = gen_info.get("prompt_key_used")
        replicate_id = gen_info.get("replicate_id")
        task_data = gen_info.get("task_data", {})

        # --- Extract target keys for comparison (using the consistently extracted values) ---
        target_provider = provider
        target_model_name = model_name
        target_task_id = task_id
        target_prompt_key = prompt_key
        target_replicate_id = replicate_id

        # --- DEBUG: Print target identifiers --- Start ---
        target_key_tuple = (target_task_id, target_provider, target_model_name, target_prompt_key, target_replicate_id)
        logger.debug(f"Attempting to find match for target key: {target_key_tuple}")
        # --- DEBUG: Print target identifiers --- End ---

        # --- Find Corresponding Original Generation Result ---
        gen_result = None
        original_gen_list = direct_stl_generation_results if is_zoo_run else generation_results
        found_index = -1

        for i, res in enumerate(original_gen_list):
            # Match based on core identifiers
            res_model_cfg = res.get("model_config_used", {})
            # --- DEBUG: Print identifiers being checked --- Start ---
            check_task_id = res.get("task_id")
            check_provider = res_model_cfg.get("provider")
            check_model_name = res_model_cfg.get("name")
            check_prompt_key_val = res.get("prompt_key_used") # Consistently use this key
            check_rep_id = res.get("replicate_id")
            check_key_tuple = (check_task_id, check_provider, check_model_name, check_prompt_key_val, check_rep_id)
            logger.debug(f"  Checking against list item {i} key: {check_key_tuple}")
            # --- DEBUG: Print identifiers being checked --- End ---

            # Perform the comparison using the correct prompt key value
            if (check_task_id == target_task_id and
                check_provider == target_provider and
                check_model_name == target_model_name and
                check_prompt_key_val == target_prompt_key and
                check_rep_id == target_replicate_id):
                gen_result = res
                found_index = i
                logger.debug(f"  Found match at index {i}") # DEBUG
                break

        if not gen_result:
             # This should ideally not happen if stl_path_to_gen_info is built correctly
             logger.error(f"CRITICAL: Consistency error! Could not find original generation result matching gen_info: {gen_info}. Skipping assembly for STL: {stl_path}")
             continue

        # Mark this generation result as processed
        list_identifier = "zoo" if is_zoo_run else "scad"
        processed_gen_indices.add( (list_identifier, found_index) )

        # --- Extract LLM Duration if applicable ---
        llm_duration = gen_result.get("llm_duration_seconds") if not is_zoo_run else None

        # --- Retrieve Render and Check Info ---
        render_info = None
        if not is_zoo_run and scad_path_from_gen_info:
            render_info = render_results_map.get(scad_path_from_gen_info)

        check_info = check_results_map.get(stl_path)

        # --- Initialize the final entry (same structure as before) ---
        final_entry = {
            "task_id": task_id, "replicate_id": replicate_id, "model_name": model_name,
            "provider": provider, "prompt_key_used": prompt_key,
            "task_description": task_data.get("description"),
            "reference_stl_path": make_relative(task_data.get("reference_stl")), # Make reference relative too
            "prompt_used": task_data.get("description") if is_zoo_run else gen_result.get("prompt_used"),
            "llm_config": {
                 "provider": provider, "name": model_name,
                 "temperature": model_config_data.get("temperature") if not is_zoo_run else None,
                 "max_tokens": model_config_data.get("max_tokens") if not is_zoo_run else None,
                 "cli_args": model_config_data.get("cli_args") if is_zoo_run else None,
             },
            "timestamp_utc": gen_result.get("timestamp"),
            "output_scad_path": None, "output_stl_path": None, "output_summary_json_path": None,
            "render_status": None, "render_duration_seconds": None, "render_error_message": None,
            "checks": {
                "check_render_successful": None, "check_is_watertight": None,
                "check_is_single_component": None, "check_bounding_box_accurate": None,
                "check_volume_passed": None, "check_hausdorff_passed": None,
                "check_chamfer_passed": None
            },
            "geometric_similarity_distance": None, "icp_fitness_score": None,
            "hausdorff_95p_distance": None, "hausdorff_99p_distance": None,
            "reference_volume_mm3": None, "generated_volume_mm3": None,
            "reference_bbox_mm": None, "generated_bbox_aligned_mm": None,
            "generation_error": gen_result.get("error") if not gen_result.get("success") else None,
            "check_error": None,
            # --- Add time fields ---
            "llm_duration_seconds": None,
            "generation_duration_seconds": None,
            "render_duration_seconds": None, # Already exists but ensure it's here
            "prompt_tokens": None, # Add cost-related fields
            "completion_tokens": None,
            "estimated_cost": None
        }

        # --- Populate Paths (Relative) using helper ---
        final_entry["output_stl_path"] = make_relative(stl_path) # Path checked exists

        # --- Populate Provider-Specific Fields ---
        if is_zoo_run:
            generation_duration = gen_result.get("generation_duration_seconds")
            final_entry["output_scad_path"] = None
            final_entry["render_status"] = "N/A"
            final_entry["render_duration_seconds"] = None
            final_entry["render_error_message"] = None
            final_entry["output_summary_json_path"] = None
            final_entry["checks"]["check_render_successful"] = None
            final_entry["llm_duration_seconds"] = None
            final_entry["generation_duration_seconds"] = generation_duration # Assign the retrieved duration
            # --- Calculate Zoo Cost ---
            estimated_cost = None
            if generation_duration is not None:
                cost_per_minute = model_config_data.get("cost_per_minute")
                free_tier_seconds = model_config_data.get("free_tier_seconds", 0)
                try:
                    if cost_per_minute is not None:
                        cost_per_minute_f = float(cost_per_minute)
                        free_tier_seconds_f = float(free_tier_seconds)
                        generation_duration_f = float(generation_duration)
                        billable_seconds = max(0, generation_duration_f - free_tier_seconds_f)
                        estimated_cost = (billable_seconds / 60.0) * cost_per_minute_f
                    else:
                         logger.warning(f"Missing 'cost_per_minute' in config for zoo_cli model '{model_name}'. Cannot calculate cost.")
                except (ValueError, TypeError) as e:
                     logger.warning(f"Invalid cost parameter for zoo_cli model '{model_name}': {e}. Cannot calculate cost.")
            final_entry["estimated_cost"] = estimated_cost
            final_entry["prompt_tokens"] = None # Zoo doesn't use tokens
            final_entry["completion_tokens"] = None # Zoo doesn't use tokens
        else: # Not Zoo
            final_entry["generation_duration_seconds"] = None
            final_entry["llm_duration_seconds"] = llm_duration # Set the extracted LLM duration
            final_entry["prompt_tokens"] = gen_result.get("prompt_tokens")
            final_entry["completion_tokens"] = gen_result.get("completion_tokens")
            # --- Calculate LLM Cost ---
            estimated_cost = None
            prompt_tokens = final_entry["prompt_tokens"]
            completion_tokens = final_entry["completion_tokens"]
            cost_input_M = model_config_data.get("cost_per_million_input_tokens")
            cost_output_M = model_config_data.get("cost_per_million_output_tokens")
            if prompt_tokens is not None and completion_tokens is not None and cost_input_M is not None and cost_output_M is not None:
                try:
                    cost_input_f = float(cost_input_M) / 1_000_000.0
                    cost_output_f = float(cost_output_M) / 1_000_000.0
                    prompt_tokens_f = float(prompt_tokens)
                    completion_tokens_f = float(completion_tokens)
                    estimated_cost = (prompt_tokens_f * cost_input_f) + (completion_tokens_f * cost_output_f)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid cost or token parameter for LLM model '{model_name}': {e}. Cannot calculate cost.")
            elif cost_input_M is None or cost_output_M is None:
                logger.warning(f"Missing cost parameters ('cost_per_million_input_tokens' or 'cost_per_million_output_tokens') in config for LLM model '{model_name}'. Cannot calculate cost.")
            final_entry["estimated_cost"] = estimated_cost

            final_entry["output_scad_path"] = make_relative(scad_path_from_gen_info)
            if render_info:
                render_success = (render_info.get("status") == "Success")
                final_entry["render_status"] = render_info.get("status")
                render_duration = render_info.get("duration")
                final_entry["render_duration_seconds"] = render_duration # Assign render duration
                final_entry["render_error_message"] = render_info.get("error")
                final_entry["checks"]["check_render_successful"] = render_success
                if render_success:
                    rendered_stl_path = render_info.get("stl_path")
                    if rendered_stl_path != stl_path:
                         logger.warning(f"Mismatch render STL path '{rendered_stl_path}' vs check STL path '{stl_path}'") # Fixed quote
                    final_entry["output_stl_path"] = make_relative(stl_path) # Use checked path
                    final_entry["output_summary_json_path"] = make_relative(render_info.get("summary_path"))
                else:
                    final_entry["output_stl_path"] = None
                    final_entry["output_summary_json_path"] = None
            else: # Render info missing (but SCAD existed)
                logger.warning(f"Missing render info for SCAD: {scad_path_from_gen_info}")
                final_entry["render_status"] = "Error"
                final_entry["render_duration_seconds"] = None # Ensure set to None if missing
                final_entry["render_error_message"] = "Render metadata missing (possible orchestration error)"
                final_entry["checks"]["check_render_successful"] = False
                final_entry["output_stl_path"] = None
                final_entry["output_summary_json_path"] = None

        # --- Populate from Check Info ---
        if check_info:
            checks_sub_dict = check_info.get("checks", {})
            for check_key in final_entry["checks"]:
                 if check_key == "check_render_successful": continue # Don't overwrite render status
                 check_value = checks_sub_dict.get(check_key)
                 final_entry["checks"][check_key] = bool(check_value) if check_value is not None else None

            final_entry["geometric_similarity_distance"] = check_info.get("geometric_similarity_distance")
            final_entry["icp_fitness_score"] = check_info.get("icp_fitness_score")
            final_entry["hausdorff_95p_distance"] = check_info.get("hausdorff_95p_distance")
            final_entry["hausdorff_99p_distance"] = check_info.get("hausdorff_99p_distance")
            final_entry["reference_volume_mm3"] = check_info.get("reference_volume_mm3")
            final_entry["generated_volume_mm3"] = check_info.get("generated_volume_mm3")
            final_entry["reference_bbox_mm"] = check_info.get("reference_bbox_mm")
            final_entry["generated_bbox_aligned_mm"] = check_info.get("generated_bbox_aligned_mm")

            check_run_error = check_info.get("error") # Overall error during check run
            check_internal_errors = check_info.get("check_errors", []) # Specific errors within checks
            all_errors = []
            if check_run_error: all_errors.append(f"Check Run Error: {check_run_error}")
            if check_internal_errors: all_errors.extend([f"Check Detail Error: {e}" for e in check_internal_errors])
            if all_errors:
                final_entry["check_error"] = "; ".join(all_errors)
                # Log as warning since the check ran but had issues
                logger.warning(f"Check issues for {os.path.basename(stl_path or 'UNKNOWN')}: {final_entry['check_error']}")
            # If check_info exists but no 'error' or 'check_errors', assume checks passed without issue

        # Handle cases where checks *should* have run but didn't
        # Condition: Gen was successful AND (it was Zoo OR render was successful) AND no check_info exists
        elif gen_result.get("success") and (is_zoo_run or final_entry["render_status"] == "Success") and not check_info:
             logger.error(f"Generation/Render OK but no check info for {stl_path}. Checks did not run or failed to record.")
             final_entry["check_error"] = "Check results missing (setup/orchestration error?)"
             # Set all check booleans to None since they didn't run
             for check_key in final_entry["checks"]:
                 if check_key == "check_render_successful": continue
                 final_entry["checks"][check_key] = None

        final_results_list.append(final_entry)

    # --- Handle Generation Failures (Entries that never made it into stl_path_to_gen_info) ---
    all_original_results = [("scad", i, res) for i, res in enumerate(generation_results)] + \
                           [("zoo", i, res) for i, res in enumerate(direct_stl_generation_results)]

    for list_id, index, gen_result in all_original_results:
        if (list_id, index) not in processed_gen_indices:
            # This result was not processed via stl_path_to_gen_info, indicating a failure
            # Double-check it actually failed
            if gen_result.get("success"):
                 # This case is unexpected - should have been processed if successful
                 logger.error(f"Logic error: Successful '{list_id}' generation result not processed: Index={index}, Data={gen_result}")
                 continue # Skip adding a redundant failure entry

            # Populate minimal failure entry
            task_id = gen_result.get("task_id")
            model_config_used = gen_result.get("model_config_used", {}) # Get the actual config used
            model_name = model_config_used.get("name") # Get name from the used config
            provider = model_config_used.get("provider") # Get provider from the used config
            prompt_key = gen_result.get("prompt_key_used") # Get the prompt key used
            replicate_id = gen_result.get("replicate_id")
            is_zoo_run = (provider == 'zoo_cli')

            # Check if essential info is missing before proceeding
            if None in [task_id, model_name, provider, prompt_key, replicate_id]:
                 logger.error(f"Cannot assemble failure entry due to missing key info in generation result: {gen_result}")
                 continue # Skip this entry if core identifiers are missing

            # Attempt to get task description and reference (might be missing if task loading failed, unlikely here)
            # We need task_data for this, which isn't directly in gen_result. This requires a lookup.
            # For simplicity in failure reporting, use fallbacks. A more robust way would involve
            # passing the original tasks list or map into the assembler.
            task_description_fallback = f"Unknown (Task ID: {task_id}, Generation Failed)"
            reference_stl_fallback = f"Unknown (Task ID: {task_id})"

            logger.debug(f"Assembling explicit failure entry for {list_id} index {index}")
            final_entry = {
                "task_id": task_id, "replicate_id": replicate_id, "model_name": model_name,
                "provider": provider, "prompt_key_used": prompt_key,
                "task_description": task_description_fallback, # Using fallback
                "reference_stl_path": reference_stl_fallback, # Using fallback
                "prompt_used": gen_result.get("prompt_used"), # Use prompt if available from gen_result
                "llm_config": { # Populate from model_config_used
                     "provider": provider, "name": model_name,
                     "temperature": model_config_used.get("temperature") if not is_zoo_run else None,
                     "max_tokens": model_config_used.get("max_tokens") if not is_zoo_run else None,
                     "cli_args": model_config_used.get("cli_args") if is_zoo_run else None,
                 },
                "timestamp_utc": gen_result.get("timestamp"),
                "output_scad_path": None, "output_stl_path": None, "output_summary_json_path": None,
                "render_status": "N/A" if is_zoo_run else "Not Run",
                "render_duration_seconds": None, "render_error_message": None,
                "checks": { key: None for key in [ # All checks are None
                    "check_render_successful", "check_is_watertight", "check_is_single_component",
                    "check_bounding_box_accurate", "check_volume_passed", "check_hausdorff_passed",
                    "check_chamfer_passed"
                ]},
                "geometric_similarity_distance": None, "icp_fitness_score": None,
                "hausdorff_95p_distance": None, "hausdorff_99p_distance": None,
                "reference_volume_mm3": None, "generated_volume_mm3": None,
                "reference_bbox_mm": None, "generated_bbox_aligned_mm": None,
                "generation_error": gen_result.get("error", "Unknown generation failure"),
                "check_error": "Checks not run due to generation failure.",
                # --- Time/Cost fields for failures ---
                # Try to get duration even if failed, cost might be partial or zero
                "llm_duration_seconds": gen_result.get("llm_duration_seconds") if not is_zoo_run else None,
                "generation_duration_seconds": gen_result.get("generation_duration_seconds") if is_zoo_run else None,
                "render_duration_seconds": None,
                "prompt_tokens": gen_result.get("prompt_tokens"), # Tokens might exist even on failure
                "completion_tokens": gen_result.get("completion_tokens"),
                "estimated_cost": None # Cost not calculated for failures currently (could be added)
            }
            final_results_list.append(final_entry)

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
