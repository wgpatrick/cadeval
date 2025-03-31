# Single-Part Text-to-CAD Evaluation Plan

This document outlines a **streamlined approach** for evaluating how well an LLM (via Cursor) can generate valid mechanical parts in OpenSCAD from **textual descriptions**. The plan focuses on single-part scenarios only. Future expansions can include part modifications and image-to-sketch tasks later.

---

## 1. Goal & Scope

- **Goal**: Assess how reliably and accurately the LLM can produce OpenSCAD files for **single mechanical parts** from textual prompts.  
- **Scope**:  
  - Single-part geometry only (no assemblies).  
  - Strictly text-based instructions (no sketches or images yet).  
  - Mechanical features such as holes, extrusions, rectangular/cylindrical profiles, etc.

---

## 2. Outline of Steps

1. **Create an Initial Evaluation Set (~10 Tasks)**  
2. **Automate the Code Generation (Cursor) & OpenSCAD Rendering**  
3. **Run Basic Geometry Checks**  
4. **Collect & Compare Results**

---

## 3. Constructing the Evaluation Set

1. **Number of Tasks**  
   - Start with about **10 tasks** defined by the user, potentially leveraging standard benchmarks like the Purdue Engineering Shape Benchmark (ESB) for realistic mechanical parts of varying complexity.
   - Keep the initial set small for rapid iteration.

2. **Task Format & Reference Models**
   - Each task is stored in a structured file (YAML/JSON).
   - **Reference Model:** For tasks based on benchmarks like ESB, the provided benchmark STL file will serve directly as the `reference_stl`. For custom tasks, the reference STL must be carefully created and validated.
   - **YAML Requirements Structure (Simplified):** The YAML file will contain:
     - `task_id`: A unique identifier (e.g., "esb_bracket_01").
     - `description`: The textual prompt for the LLM.
     - `reference_stl`: Path to the ground truth STL file.
     - `requirements`: Focused on essential, easily verifiable properties:
       - `bounding_box`: Target dimensions [L, W, H].
       - *(Optional)* `topology_requirements`: e.g., `expected_component_count: 1`.
       - *(Avoid complex feature lists like hole counts/positions initially; rely on Check 5 for detailed geometric fidelity).*

### Example (YAML - Revised):

```yaml
task_id: "rect_plate_4holes_simple"
description: "Create a rectangular plate 100 mm long, 50 mm wide, and 5 mm thick. Place four 10 mm diameter through-holes, centered 10mm from each edge at the corners."
reference_stl: "./reference/rect_plate_4holes_simple.stl" # Assumes this file exists and is correct
requirements:
  bounding_box: [100, 50, 5]
  topology_requirements:
    expected_component_count: 1 # This implies no floating parts
```

**Field Definitions:**

*   `task_id` (string, required): A unique identifier for the task (e.g., `snake_case_description`). Used for naming output files.
*   `description` (string, required): The natural language prompt provided to the LLM to generate the OpenSCAD model.
*   `reference_stl` (string, required): The relative path (from the project root) to the ground truth `.stl` file located in the `reference/` directory. This file is used for comparison during the geometry checks.
*   `requirements` (object, required): A dictionary containing specific criteria for evaluating the generated model.
    *   `bounding_box` (list of 3 numbers, required): The target dimensions `[Length, Width, Height]` in millimeters (mm) that the final model's bounding box should match, within a defined tolerance (see `config.yaml`). The order should be consistent.
    *   `topology_requirements` (object, optional): Specifies requirements related to the model's structure.
        *   `expected_component_count` (integer, optional): The number of distinct, unconnected solid bodies expected in the final model. For single-part designs, this should typically be `1`. If omitted, this specific check might be skipped or default to assuming 1.

---

## 4. Automating the Code Generation & Rendering

### 4.1 Generating OpenSCAD (Direct LLM API Calls)

**Approach:**

- A Python script iterates over each task defined in the `tasks/` directory.
- For each task, the script iterates through a list of target Large Language Models (LLMs) specified in the `config.yaml` under the `llm.models` key. Each entry in this list defines the model's `name`, `provider`, and specific parameters like `temperature` and `max_tokens`.
- The script formats the task `description` into the appropriate API request format for the current LLM provider.
- It uses official Python client libraries (e.g., `openai`, `anthropic`, `google-generativeai`) to send the request to the respective LLM API endpoint.
- The LLMs targeted for evaluation (as configured in `config.yaml`) typically include:
    - OpenAI: `gpt-4o-mini` (or similar small/efficient model)
    - Anthropic: `claude-3-5-sonnet-20240620`
    - Google: `gemini-1.5-pro-latest`
- The script saves the returned OpenSCAD code into a structured output directory, likely including the model name in the filename (e.g., `./generated_outputs/{task_id}_{model_name}.scad`).

**Authentication:**
- API keys for OpenAI, Anthropic, and Google Cloud will be read from environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). The script will expect these to be set in its execution environment.

**API Error Handling:**
- The script should implement basic error handling for API calls (e.g., network issues, rate limits, invalid responses) and log these failures appropriately. Retry logic may be considered for transient errors.

### 4.2 Rendering SCAD to STL (Headless Mode) - Detailed Plan (Revised)

**Purpose:** This step converts the generated `.scad` files into `.stl` mesh files using OpenSCAD's command-line interface and captures rendering metadata. These outputs are crucial for subsequent geometry checking and analysis.

**Inputs:**
1.  A list of paths to the generated `.scad` files (e.g., `./generated_outputs/{task_id}_{model_name}.scad`).
2.  The corresponding base output path for each task/model combination (e.g., `./generated_outputs/{task_id}_{model_name}`).

**Core Process:**
1.  **Iteration:** Loop through each input `.scad` file path.
2.  **Define Output Paths:** For each input, define the target paths for both the `.stl` file and the `.json` summary file (e.g., `output_base + ".stl"` and `output_base + ".json"`).
3.  **Command Construction:** Construct the OpenSCAD command-line arguments. Aim for a command like:
    ```bash
    openscad -q \
             --export-format asciistl \
             --backend Manifold \
             --summary all \
             --summary-file <output_json_path> \
             -o <output_stl_path> \
             <input_scad_path>
    ```
    *   `-q`: Quiet mode (suppresses info messages, shows errors).
    *   `--export-format asciistl`: Explicitly requests ASCII STL format for consistency (can be made configurable if binary is preferred).
    *   `--backend Manifold`: Use the faster backend (consider making optional or checking version).
    *   `--summary all`: Request all available summary information.
    *   `--summary-file <path>`: Specify the output file for the JSON summary.
    *   `-o <path>`: Specify the output STL file.
    *   `<input>`: The input SCAD file.
4.  **Execution:** Execute the command using `subprocess.run` or similar in Python.
5.  **Timeout Handling:** Implement and enforce a timeout (e.g., 60-120 seconds). Terminate the process if it exceeds the timeout.
6.  **Process Monitoring:** Capture `stdout`, `stderr`, and the process `returncode`. A non-zero `returncode` or non-empty `stderr` (even with `-q`) usually indicates an error.

**Outputs:**
1.  `.stl` files for successfully rendered models, saved to the specified output directory.
2.  `.json` summary files for successfully rendered models (if using a compatible OpenSCAD version), containing metadata like render time, geometry details, and bounding box.
3.  Status information for each rendering attempt (success, failure type, duration, path to STL, path to JSON).

**Dependencies:**
1.  **OpenSCAD Installation:** A working installation is required.
2.  **Version Requirement:** Target OpenSCAD **2021.01+**. Script should check version.
3.  **Path Configuration:** The path to the OpenSCAD executable will be read from a configuration file (e.g., `config.yaml` or `config.ini`). The script will fail if the config file or path is invalid.

**Error Handling & Reporting:**
Expand the previous list:
1.  OpenSCAD Not Found.
2.  SCAD File Not Found.
3.  **OpenSCAD Version Too Old:** If the version check fails the minimum requirement for summary files. -> *Report: "OpenSCAD version too old, summary file requires X.Y+"* (Potentially proceed without summary generation).
4.  OpenSCAD Compile Error (non-zero return code or stderr output). -> *Report: "OpenSCAD failed..." (Include stderr).*
5.  Timeout. -> *Report: "Rendering timed out."*
6.  **Summary File Not Created:** If OpenSCAD succeeded (return code 0) but the expected JSON file is missing. -> *Report: "Summary file generation failed."*
7.  Permissions Issues.

**Configuration:**
- A configuration file (e.g., `config.yaml`) will specify defaults/settings:
    - `openscad_executable_path`: Path to binary.
    - `render_timeout_seconds`: e.g., `120`.
    - `output_directory`: e.g., `./generated_outputs/`.
    - `minimum_openscad_version`: e.g., `"2021.01"`.
    - `export_format`: e.g., `asciistl`.
    - `openscad_backend`: e.g., `Manifold`.
    - `summary_options`: e.g., `all`.

**Logging:**
Update logging to include:
-   OpenSCAD version detected.
-   Full command executed.
-   Status (including summary file success/failure).
-   Path to generated summary JSON (if successful).

**Integration with Step 5:** Explicitly note that the generated JSON summary file should be passed as an input to the geometry checking step (Step 5), as it likely contains pre-calculated bounding box information and other useful geometry stats, potentially satisfying some checks directly.

---

## 5. Automated Geometry Checks (Revised based on Research)

**Purpose:** To automatically assess the geometric fidelity and topological soundness of the generated STL file against the reference STL and task requirements, producing binary pass/fail results for each check performed.

**Inputs:**
1.  Generated STL Path (`./generated_outputs/{task_id}_{model_name}.stl`).
2.  Reference STL Path (`./reference/{task_id}.stl`).
3.  Task Requirements (from YAML).
4.  OpenSCAD Summary JSON Path (Optional, from step 4.2).
5.  Rendering Status (from Step 4.2).

**Core Checks (Implemented in `geometry_check.py` using `Trimesh`, `Open3D`):**

1.  **Check 1: Render Success**
    *   **Input:** Rendering status from Step 4.2.
    *   **Logic:** Did the OpenSCAD rendering complete successfully (return code 0, no critical errors in stderr, no timeout)?
    *   **Output:** `render_successful: true/false`

2.  **Check 2: Topological Integrity - Watertight**
    *   **Input:** Generated STL Path.
    *   **Logic:** Load mesh (`Trimesh`). Use `mesh.is_watertight`. *Prerequisite: Render must be successful.*
    *   **Output:** `is_watertight: true/false/null`

3.  **Check 3: Topological Integrity - Single Component**
    *   **Input:** Generated STL Path, Task Requirements (`topology_requirements`).
    *   **Logic:** Load mesh (`Trimesh`). Use `mesh.body_count`. Check if count is 1 (if `topology_requirements` is true). *Prerequisite: Render must be successful.*
    *   **Output:** `is_single_component: true/false/null`

4.  **Check 4: Bounding Box Accuracy**
    *   **Input:** Task Requirements (`bounding_box`), OpenSCAD Summary JSON (preferred), Generated STL.
    *   **Logic:** Get bounding box extents [L, W, H] (from JSON or `Trimesh`). Compare each dimension against requirements within tolerance (e.g., start with `±0.5mm` absolute).
    *   **Output:** `bounding_box_accurate: true/false/null`

5.  **Check 5: Geometric Similarity (Mesh Comparison)**
    *   **Input:** Generated STL Path, Reference STL Path.
    *   **Logic:**
        *   Load meshes.
        *   **Alignment:** Perform **Iterative Closest Point (ICP)** alignment (e.g., using `Open3D` with its **default parameters**) to register generated mesh to reference. Record the final **ICP fitness score** (e.g., root mean square error of correspondences). If alignment fitness is poor (above a threshold, e.g., > 1.0), the subsequent distance calculation might be less meaningful.
        *   **Distance Calculation:** Calculate **Chamfer Distance** between the *aligned* meshes.
    *   **Prerequisite:** Render success, meshes load. ICP must run, though distance is calculated even if fitness is poor (allowing analysis).
    *   **Output:** Record the calculated `geometric_similarity_distance` (float) and the `icp_fitness_score` (float). Both will be null if prerequisites fail.

**Execution Logic:**
- All checks (1-5) will be attempted for each generated model, provided their prerequisites are met.

**Outputs of `geometry_check.py`:**
*   A dictionary/JSON object per evaluated model, conforming to the schema defined in Section 6, including the calculated similarity distance and ICP fitness score.

**Refinement Notes:**
*   **Thresholds & Parameters:** Initial tolerance values (bounding box: ±0.5mm) are starting points. **ICP parameters will use library defaults initially.** Geometric similarity will be assessed post-run by comparing the recorded `geometric_similarity_distance` against multiple thresholds (e.g., 0.1mm, 1mm, 10mm) during analysis.
*   **Alignment:** Successful ICP alignment (indicated by a low `icp_fitness_score`) is critical for interpreting the `geometric_similarity_distance`.
*   **Libraries:** Recommend `Trimesh` (topology), `Open3D` (ICP, Chamfer).
*   **Focus:** Checks prioritize geometric/topological correctness. Check 5 provides a quantitative measure of shape fidelity.
*   **Pre-processing:** Assume models generated at correct scale.

---

## 6. Collecting & Comparing Results

### Logging & Results Schema

- For each task and model combination evaluated, the results will be stored in a primary results file (e.g., `eval_results_run_XYZ.json`). This file will be a list of JSON objects.
- **Schema for each result entry:** This schema includes details about the specific run configuration used for this entry.
  ```json
  {
    "task_id": "string",
    "model_name": "string",                 // e.g., "gpt-4o-mini"
    "task_description": "string",           // The original description from the YAML
    "reference_stl_path": "string",         // Path to the reference STL
    "prompt_used": "string",                // The exact final prompt sent to the LLM API
    "llm_config": {                         // Optional: Capture key LLM settings if varied
        "temperature": "float/null",
        "max_tokens": "integer/null"
        // Add other relevant settings if needed
    },
    "timestamp_utc": "string",              // ISO 8601 format timestamp of this specific evaluation
    "output_scad_path": "string",
    "output_stl_path": "string",
    "output_summary_json_path": "string/null",
    "render_status": "string",              // "SUCCESS", "COMPILE_ERROR", "TIMEOUT", etc.
    "render_duration_seconds": "float/null",
    "render_error_message": "string/null",
    "checks": {
      "check_render_successful": "boolean",
      "check_is_watertight": "boolean/null",
      "check_is_single_component": "boolean/null",
      "check_bounding_box_accurate": "boolean/null"
      // Geometric similarity results are now top-level values below
    },
    "icp_fitness_score": "float/null",      // Lower is better alignment
    "geometric_similarity_distance": "float/null", // e.g., Chamfer Distance, lower is more similar
    "check_error_message": "string/null"    // Errors during geometry_check.py phase
  }
  ```
- This detailed schema ensures each result record is self-contained regarding the conditions under which it was generated.

### Versioning

- **Configuration Tracking:** Key configuration parameters (LLM used, exact prompt, LLM settings like temperature if varied) are recorded directly within each result entry in the output JSON (see schema above).
- **Run Identification:** While individual results contain configuration details, major evaluation runs (e.g., after significant script changes, prompt template updates, or testing a new set of models) should still be saved to uniquely named results files (e.g., `results/eval_results_v1_base_prompt.json`, `results/eval_results_v2_temp_0.5.json`) for organizational clarity and easier high-level comparison between distinct experimental setups.

### Iteration

- Update tasks or the scoring as new failure modes appear.

---

## 7. Future Work

### Incremental CAD Modifications

- Provide an initial SCAD file plus a textual command to modify geometry.
- Test how well the LLM adjusts an existing design.

### Image/Sketch-to-CAD

- Provide a simple line drawing or annotated photo.
- Evaluate how accurately the LLM interprets dimensions from an image.

### More Complex Parts & Assemblies

- Introduce multiple features or parametric assemblies.
- Check alignment, constraints, etc.

### Advanced Metrics

- Use surface-to-surface comparison with a reference STL (via CloudCompare or trimesh), for a more detailed similarity score.

---

## 8. Example Project Structure

```
my_cad_eval/
├── tasks/
│   ├── rect_plate_4holes.yaml
│   ├── cylinder_spacer.yaml
│   └── ...
├── reference/
│   ├── rect_plate_4holes.stl
│   ├── cylinder_spacer.stl
│   └── ...
├── scripts/
│   ├── config_loader.py          # Handles loading config.yaml
│   ├── test_config_loader.py     # Unit tests for config loader
│   ├── logger_setup.py           # Configures project logging
│   ├── test_logger_setup.py      # Unit tests for logger setup
│   ├── run_evaluation.py         # Target main orchestration script (Phase 6)
│   ├── geometry_check.py         # Target geometry checking script (Phase 5)
│   └── ...                       # Other future scripts (e.g., LLM clients, renderer)
├── generated_outputs/            # Stores LLM-generated .scad and rendered .stl/.json files
│   ├── rect_plate_4holes_gpt-4o-mini.scad
│   ├── rect_plate_4holes_gpt-4o-mini.stl
│   └── ...
├── results/                      # Stores final evaluation JSON results
│   └── eval_results_run_XYZ.json
├── logs/                         # Stores log files (ignored by git)
│   └── cadeval.log
├── config.yaml                   # Main configuration file
├── requirements.txt              # Python dependencies
├── TODO.md                       # Project tasks
└── README.md                     # This file
```

`run_evaluation.py` (future script) handles:
- Reading YAML tasks.
- Generating SCAD via LLM APIs.
- Invoking OpenSCAD headless to produce STLs.
- Running geometry checks (`geometry_check.py`).
- Saving a final JSON with results. 