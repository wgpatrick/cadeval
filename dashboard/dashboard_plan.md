# CadEval Results Visualization Dashboard Plan

## 1. Objective

To create a static HTML-based dashboard that visualizes the results of a CadEval run. The dashboard should provide:
*   A clear, per-model overview of success/failure rates across all evaluated tasks at different stages of the pipeline (SCAD generation, Rendering, Geometry Checks).
*   Aggregate statistics summarizing the overall performance of each model across all tasks.
*   Comparative charts showing how models stack up against each other based on these aggregate statistics.

## 2. Core Components

The solution will consist of two main parts:

*   **Data Processing Script:** A Python script (`scripts/process_results.py`) that takes a `results_*.json` file as input and generates a structured JSON file (`dashboard_data.json`) containing the necessary information and calculated statistics for the dashboard. This script operates independently of the main `run_evaluation.py` pipeline.
*   **Static Dashboard Page:** An HTML file (`dashboard.html`) with associated CSS for styling and JavaScript for loading the data, building tables (using AG-Grid), and rendering charts (using Chart.js).

## 3. Data Processing Script (`scripts/process_results.py`)

*   **Input:** Path to a specific `results_RUNID.json` file.
*   **Logic:**
    1.  **Load Data:** Read the input JSON results file. Extract Run ID from the input path if possible.
    2.  **Process Individual Results:** Iterate through each entry (task-model combination) in the results list. For each entry, determine and store clear success/failure flags for:
        *   `scad_generation_success`: Based on `output_scad_path` being present and `generation_error` being null.
        *   `render_success`: Based on `render_status == "Success"`.
        *   `geometry_check_orchestration_success`: Based on `render_success` being true and `check_error` being null (meaning the checks ran without a high-level error).
        *   `individual_geometry_checks_passed`: A boolean indicating if *all* individual checks within the `checks` dictionary (e.g., `check_is_watertight`, `check_bounding_box_accurate`) were explicitly `true`. This is only relevant if `geometry_check_orchestration_success` is true. Defaults to `false` if checks didn't run or had an orchestration error.
        *   `overall_pipeline_success`: `scad_generation_success` AND `render_success` AND `geometry_check_orchestration_success` AND `individual_geometry_checks_passed`.
        *   Store relevant metrics like `geometric_similarity_distance` and individual check statuses. Handle cases where checks didn't run (e.g., render failed) by setting check statuses to `N/A`.
    3.  **Calculate Meta-Statistics:** Group the processed results by `model_name`. For each model, calculate aggregate statistics across all its tasks:
        *   `total_tasks`: Total tasks evaluated for this model.
        *   `scad_gen_success_count`: Number of successful SCAD generations.
        *   `render_success_count`: Number of successful renders.
        *   `geo_check_run_success_count`: Number of times geometry checks ran without orchestration error.
        *   `all_geo_checks_passed_count`: Number of times all individual geometry checks passed.
        *   `overall_pipeline_success_count`: Number of times the full pipeline succeeded.
        *   `scad_gen_success_rate`: (`scad_gen_success_count` / `total_tasks`) * 100
        *   `render_success_rate`: (`render_success_count` / `scad_gen_success_count`) * 100 if `scad_gen_success_count` > 0 else 0
        *   `geo_check_run_success_rate`: (`geo_check_run_success_count` / `render_success_count`) * 100 if `render_success_count` > 0 else 0
        *   `all_geo_checks_passed_rate`: (`all_geo_checks_passed_count` / `geo_check_run_success_count`) * 100 if `geo_check_run_success_count` > 0 else 0
        *   `overall_pipeline_success_rate`: (`overall_pipeline_success_count` / `total_tasks`) * 100
        *   `average_similarity_distance`: Average `geometric_similarity_distance` for tasks that passed the full pipeline. Calculate count of valid distances.
    4.  **Structure Output Data:** Organize the processed individual results (grouped by model) and the calculated meta-statistics into a dictionary. Include the extracted run ID.
*   **Output:** A JSON file (`dashboard_data.json`) containing the structured data.

## 4. Visualization Dashboard (`dashboard.html`)

*   **Technology:**
    *   HTML: Structure the page content.
    *   CSS: Minimal custom styling for layout and readability.
    *   JavaScript:
        *   Fetch and parse `dashboard_data.json`.
        *   Use AG-Grid to dynamically generate detailed tables per model.
        *   Use Chart.js to create bar charts comparing model meta-statistics.
*   **Content & Features:**
    *   **Header:** Display the Run ID.
    *   **Summary Charts:** Bar charts comparing key meta-statistics across models (e.g., Overall Pipeline Success Rate, Render Success Rate, All Geo Checks Passed Rate).
    *   **Detailed Model Tables (AG-Grid):**
        *   One grid per model, possibly within tabs or expandable sections.
        *   Columns: Task ID, SCAD Generated (✅/❌), Rendered (✅/❌/N/A), Geo Checks Ran (✅/❌/N/A), All Geo Checks Passed (✅/❌/N/A), Individual Check Statuses (e.g., Watertight: ✅/❌/N/A), Similarity Distance (value or N/A). Use icons/colors for clarity.
    *   **Data Loading:** Load `dashboard_data.json` on page load. Optionally add a file input to load data from other run files.

## 5. Workflow

1.  Run CadEval evaluation (`python scripts/run_evaluation.py ...`).
2.  Run the processing script (`python scripts/process_results.py --input results/RUN_ID/results_RUN_ID.json --output dashboard_data.json`).
3.  Open `dashboard.html` in a web browser.

## 6. Future Enhancements (Optional)

*   Links to view generated SCAD/STL files.
*   Filtering/sorting options in AG-Grid tables.
*   Visual diff links/images.
*   Ability to compare multiple runs.
*   More sophisticated charts (scatter plots, etc.).