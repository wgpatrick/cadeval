# CadEval Project TODO List

**Objective:** Develop and execute a streamlined evaluation framework to assess the ability of various Large Language Models (LLMs) to generate valid single-part OpenSCAD models from textual descriptions, based on the methodology outlined in `fromchatgpt.md`.

**Guiding Principles:**
*   **Incremental Development:** Build and test components step-by-step.
*   **Source of Truth:** Adhere strictly to the requirements and specifications in `fromchatgpt.md`.
*   **Explicit Testing:** Integrate specific testing procedures at each phase to validate functionality before proceeding.
*   **Clear Documentation:** Ensure code is well-commented and processes are clear.

---

## Phase 1: Project Setup & Foundational Configuration

*Goal: Establish the basic project structure, environment, and configuration.*

-   [x] **[Setup] Initialize Project Repository:**
    -   [x] Ensure Git repository is initialized.
    -   [x] Confirm primary branch is `main`.
    -   [x] Create/verify `.gitignore` (ignoring `venv/`, `__pycache__/`, `*.pyc`, `generated_outputs/`, `results/`, OS/IDE specific files like `.DS_Store`, `.vscode/`, `.idea/`).
-   [x] **[Setup] Create Core Directory Structure:**
    -   [x] Create `tasks/`
    -   [x] Create `reference/`
    -   [x] Create `scripts/`
    -   [x] Create `generated_outputs/`
    -   [x] Create `results/`
-   [x] **[Setup] Establish Python Environment:**
    -   [x] Create Conda environment (`cadeval` with Python 3.10).
    -   [x] Activate Conda environment.
    -   [x] Install dependencies using Conda/pip within the environment.
    -   [x] Create `environment.yml` for tracking dependencies.
    -   [x] **[Test]** Verify environment and key package imports (e.g., `import openai`, `import trimesh`, `import yaml`).
    -   [x] Install open3d dependency (successfully installed in Conda environment).
-   [x] **[Setup] Implement Configuration Loading:**
    -   [x] Create `config.yaml` with structure from `fromchatgpt.md` (Paths, LLM settings, Check settings). *Remember to adjust `openscad.executable_path` for the target system.*
    -   [x] Develop `scripts/config_loader.py` to load and validate `config.yaml`.
    -   [x] **[Test]** Create `tests/test_config_loader.py` with unit tests for:
        -   [x] Successful loading.
        -   [x] Correct retrieval of nested values.
        -   [x] Error handling for missing/malformed file.
-   [x] **[Setup] Implement Basic Logging:**
    -   [x] Create a utility in `scripts/` (e.g., `scripts/logger_setup.py`) for basic logging configuration (console/file output).
    -   [x] **[Test]** Write a simple test script or unit test to verify INFO/ERROR messages are logged correctly.

---

## Phase 2: Task Definition & Reference Model Preparation

*Goal: Define the evaluation tasks and prepare the ground truth models.*

-   [x] **[Task Def] Define YAML Task Format:**
    -   [x] Document the required YAML structure (`task_id`, `description`, `reference_stl`, `requirements.bounding_box`, `requirements.topology_requirements`).
    -   [x] *(Optional)* Create a JSON schema (`schemas/task_schema.json`) for validation.
-   [ ] **[Task Def] Create Initial Evaluation Tasks (~10):**
    -   [x] Develop 1 initial task description (`task1`). *(Defer ~9 more until after pipeline testing)*.
    -   [x] Create 1 initial YAML file in `tasks/` (`task1.yaml`). *(Defer ~9 more until after pipeline testing)*.
-   [ ] **[Task Def] Create/Acquire Reference STL Models:**
    -   [x] Create 1 initial reference STL model (`task1.stl` via Fusion 360). *(Defer acquiring/creating ~9 more)*.
    -   [x] Place initial reference STL in `reference/` with correct filename (`task1.stl`). *(Defer placing ~9 more)*.
    -   [x] **[Validation]** Manually inspect the initial reference STL (`task1.stl`) for correctness and geometric soundness. *(Defer validation for remaining tasks)*.
-   [x] **[Task Def] Implement Task Loading Script:**
    -   [x] Develop `scripts/task_loader.py` to scan `tasks/`, parse YAMLs, and return task data.
    -   [x] *(Optional)* Integrate JSON schema validation.
    -   [x] **[Test]** Create `tests/test_task_loader.py` with unit tests for:
        -   [x] Loading valid tasks.
        -   [x] Correct attribute parsing.
        -   [x] Handling invalid/non-matching YAMLs.
        -   [x] Handling empty directory.

---

## Phase 3: LLM Interaction & SCAD Generation

*Goal: Implement the logic to interact with LLM APIs and generate OpenSCAD code.*

-   [x] **[LLM] Secure API Key Handling:**
    -   [x] Use `.env` file and `python-dotenv` (from `requirements.txt`) to manage API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Ensure `.env` is in `.gitignore`.
    -   [x] Update `config_loader.py` or create a utility to load environment variables.
-   [x] **[LLM] Develop LLM API Client Wrapper(s):**
    -   [x] Create script(s) in `scripts/` (e.g., `scripts/llm_clients.py`) to abstract interactions with OpenAI, Anthropic, and Google APIs.
    -   [x] Functions should take prompt text, model name, and other parameters (like temperature, max_tokens from `config.yaml`) and return the generated text or handle errors.
    -   [x] Implement basic API error handling (network issues, rate limits, invalid responses) and logging as specified in Section 4.1.
    -   [x] **[Test]** *(Requires API Keys)* Write integration tests (potentially skippable in automated CI without keys) to:
        -   [x] Send a simple test prompt to each configured LLM.
        -   [x] Verify a text response is received.
        -   [x] Test basic error handling (e.g., providing an invalid API key temporarily).
-   [x] **[LLM] Implement SCAD Generation Logic:**
    -   [x] Create the main script `scripts/generate_scad.py`.
    -   [x] This script should:
        -   [x] Load configuration (`config_loader.py`).
        -   [x] Load tasks (`task_loader.py`).
        -   [x] Iterate through each task.
        -   [x] Iterate through the target LLMs specified in `config.yaml`.
        -   [x] Format the task `description` into a suitable prompt (consider a simple template initially). *Log the exact prompt used.*
        -   [x] Call the appropriate LLM client wrapper to get the OpenSCAD code.
        -   [x] Save the raw generated OpenSCAD code to the `generated_outputs/` directory using the specified naming convention (`{task_id}_{model_name}.scad`). Handle LLM failures gracefully (log error, skip saving).
    -   [x] **[Test]** Write unit tests for prompt formatting. Write integration tests (can be run manually or configured carefully) for the end-to-end flow of one task with one mock/real LLM call, verifying the `.scad` file is created with expected content.

---

## Phase 4: OpenSCAD Rendering & STL Export

*Goal: Automate the conversion of generated `.scad` files to `.stl` format using OpenSCAD headless mode.*

-   [x] **[Render] OpenSCAD Path Validation:**
    -   [x] Add a check in the rendering script or a utility function to verify the `openscad.executable_path` from `config.yaml` points to a valid executable file. Fail early if not found.
    -   [x] Add a check for the OpenSCAD version (using `--version`) against `openscad.minimum_version`. Log a warning or error if too old, especially regarding summary file support (Section 4.2).
-   [ ] **[Render] Implement Rendering Script:**
    -   [x] Create `scripts/render_scad.py`.
    -   [x] This script should:
        -   [x] Accept a list of `.scad` file paths as input (or scan `generated_outputs/`). *(Scanning implemented in `render_all_scad`)*
        -   [x] Load configuration (`config.yaml`). *(Done in `render_scad_file` and `__main__`)*
        -   [x] For each `.scad` file: *(Logic implemented in `render_scad_file`)*
            -   [x] Define output `.stl` and `.json` (summary) paths in `generated_outputs/`.
            -   [x] Construct the `openscad` command line arguments precisely as specified in Section 4.2 (using `-q`, `--export-format`, `--backend`, `--summary`, `--summary-file`, `-o`). *Ensure paths are quoted if they might contain spaces.* *(Handled by subprocess)*
            -   [x] Execute the command using `subprocess.run`.
            -   [x] Implement the timeout (`openscad.render_timeout_seconds`).
            -   [x] Capture `stdout`, `stderr`, and `returncode`.
            -   [x] Determine render status (Success, Compile Error, Timeout).
            -   [x] Check if the `.stl` file was created.
            -   [x] Check if the `.json` summary file was created (if expected based on OpenSCAD version/flags).
            -   [x] Log detailed information: command executed, status, duration, paths to output files, stderr on failure.
        -   [x] Return structured status information for each file processed. *(Done by `render_scad_file`)*
-   [x] **[Render] Develop Test SCAD Files:**
    -   [x] Create a few simple, known-good `.scad` files in a test directory (e.g., `tests/test_data/scad/`).
    -   [x] Create a simple, known-bad `.scad` file (syntax error).
    -   [x] Create a `.scad` file likely to take longer to render (for timeout testing, if feasible).
-   [x] **[Render] Testing:**
    -   [ ] **[Test]** Write unit tests for command construction logic.
    -   [x] **[Test]** *(Requires OpenSCAD installation)* Write integration tests for `render_scad.py`:
        -   [x] Test rendering a known-good file -> verify STL (+ JSON summary if applicable) is created, return code 0.
        -   [x] Test rendering a known-bad file -> verify return code is non-zero, capture stderr.
        -   [x] Test rendering with a timeout (if a suitable test file exists) -> verify timeout behaviour.
        -   [ ] Test handling if OpenSCAD executable is not found.
        -   [x] Test handling if input `.scad` file is missing.

---

## Phase 5: Automated Geometry Checks

*Goal: Implement automated checks on the generated STL files based on requirements and comparison with reference models.*

-   [x] **[Check] Implement `geometry_check.py`:**
    -   [x] Create `scripts/geometry_check.py`.
    -   [x] This script/module should define functions to perform each check specified in Section 5.
    -   [x] Input to the main checking function: paths to generated STL, reference STL, task requirements (dict), rendering status, path to OpenSCAD summary JSON (optional). *(Verified inputs are passed and used where needed, e.g., requirements for bounding box & single component)*
    -   [x] **Check 1: Render Success:** Implement logic based on rendering status input.
    -   [x] **Check 2: Watertight:** Use `trimesh` (`mesh.is_watertight`). Handle potential errors during mesh loading. *Prerequisite: Render Success.*
    -   [x] **Check 3: Single Component:** Use `open3d` (`mesh.cluster_connected_triangles`). Compare with `task_requirements['topology_requirements']['expected_component_count']` if present. *Prerequisite: Render Success.*
    -   [x] **Check 4: Bounding Box Accuracy:**
        -   [x] Prioritize getting bounding box from OpenSCAD summary JSON if available and valid.
        -   [x] Fallback to calculating with `trimesh` (`mesh.bounds`).
        -   [x] Compare extents ([L, W, H]) against `task_requirements['bounding_box']` using the tolerance from `config.yaml` (`geometry_check.bounding_box_tolerance_mm`). *Handle potential dimension ordering issues.*
        -   [x] *Prerequisite: Render Success & valid mesh/summary.*
    -   [x] **Check 5: Geometric Similarity (Mesh Comparison):**
        -   [x] Load generated and reference STL using `trimesh` or `open3d`. Handle loading errors.
        -   [x] Perform ICP alignment (`open3d.pipelines.registration.registration_icp` using PointToPlane) to align generated to reference.
        -   [x] Record the ICP fitness score (`icp_result.fitness`).
        -   [x] Calculate Chamfer Distance (`open3d.legacy.pipelines.registration.evaluate_registration` or dedicated Chamfer function if using newer Open3D) between the *aligned* generated mesh and the reference mesh. *(Implemented via point cloud sampling and distance calculation)*
        -   [x] Record the Chamfer distance.
        -   [x] *Prerequisites: Render Success, both meshes load successfully, ICP runs.*
    -   [x] The main function should orchestrate these checks, respecting prerequisites, and return a dictionary conforming to the results schema (Section 6), including check results, similarity scores, and any check-phase error messages.
-   [x] **[Check] Develop Test STL Files:**
    -   [x] Create/select pairs of simple reference/generated STLs in a test directory (e.g., `tests/test_data/stl/`) exhibiting specific properties:
        -   [x] Identical models (`ref_cube.stl`, `gen_cube_identical.stl`).
        -   [x] Slightly different models (within tolerance) (`gen_cube_slight_diff.stl`).
        -   [x] Significantly different models (`gen_sphere_diff.stl`).
        -   [x] A non-watertight model (multi-component) (`gen_non_watertight.stl`, `gen_multi_component.stl`).
        -   [x] A model with multiple components (`gen_multi_component.stl`).
        -   [x] Models requiring ICP alignment (All comparisons use ICP).
-   [x] **[Check] Testing:**
    -   [x] **[Test]** Write unit tests for individual check logic where possible (e.g., bounding box comparison logic). *(Implicitly tested via integration)*
    -   [x] **[Test]** *(Requires geometry libraries)* Write integration tests for `geometry_check.py` using the test STL pairs (`tests/test_geometry_check.py`):
        -   [x] Test Check 1 logic (pass/fail based on input status). *(Covered by `test_check_render_success`)*
        -   [x] Test Check 2 on watertight/non-watertight models. *(Covered by `test_check_watertight_*` tests)*
        -   [x] Test Check 3 on single/multi-component models. *(Covered by `test_check_single_component_*` tests)*
        -   [x] Test Check 4 with models inside/outside bounding box tolerance. *(Covered by integration tests)*
        -   [x] Test Check 5 with identical/similar/different models, verifying ICP fitness and Chamfer distance values (or ranges). *(Covered by `test_check_similarity_*` tests)*
        -   [x] Test prerequisite handling (e.g., ensure similarity isn't calculated if rendering failed). *(Implicitly covered, could add specific test)*
        -   [x] Test handling of invalid/unloadable STL files. *(Covered by `*_invalid_stl` tests)*

---

## Phase 6: Evaluation Execution & Results Collection

*Goal: Orchestrate the end-to-end evaluation process and collect results in the specified JSON format.*

-   [ ] **[Exec] Develop Main Orchestration Script (`run_evaluation.py`):**
    -   [x] Create `scripts/run_evaluation.py`.
    -   [x] This script is the main entry point and should:
        -   [x] Parse command-line arguments (e.g., specify tasks to run, specific LLMs, output file name). Use `click` or `argparse`.
        -   [x] Load configuration (`config_loader.py`).
        -   [x] Load specified tasks (`task_loader.py`).
        -   [x] Initialize results list.
        -   [x] **Loop through Tasks & LLMs:**
            -   [x] Call `generate_scad.py` logic (or import relevant functions) to generate the `.scad` file for the current task/LLM combination.
            -   [x] Record SCAD generation success/failure and path.
        -   [x] **Batch Render:**
            -   [x] Collect all successfully generated `.scad` paths.
            -   [x] Call `render_scad.py` logic (or import function) to render all SCAD files to STL (+ summary JSON).
            -   [x] Store rendering status (success, compile error, timeout), duration, output paths, and error messages for each attempt.
        -   [x] **Batch Check:**
            -   [x] For each successfully rendered STL:
                -   [x] Gather inputs for `geometry_check.py` (STL paths, reference path from task, task requirements, render status, summary JSON path).
                -   [x] Call `geometry_check.py` logic (or import function).
                -   [x] Store the returned check results dictionary.
        -   [x] **Assemble Final Results:**
            -   [x] For each task/LLM attempt, create a final JSON object matching the **exact schema** defined in Section 6. Include: task details, LLM config used, timestamps, output paths, render status/results, check results (including similarity scores), and any error messages.
            -   [x] Append this object to the main results list.
        -   [x] **Save Results:**
            -   [x] Generate a unique filename for the results JSON (e.g., including timestamp or run ID).
            -   [x] Save the results list as a JSON file in the `results/` directory.
-   [ ] **[Exec] End-to-End Testing:**
    -   [ ] **[Test]** Perform a dry run using 1-2 simple test tasks and potentially mock LLM/render/check functions to verify the orchestration flow.
    -   [ ] **[Test]** Perform a small end-to-end run using 1-2 actual tasks, 1-2 LLMs (if keys available), and the real OpenSCAD/geometry checks.
        -   [ ] Verify the `results/<run_id>.json` file is created.
        -   [ ] Manually inspect the JSON output to confirm it matches the schema in Section 6 and contains plausible data for each step (paths, statuses, check results).
        -   [ ] Verify intermediate files (`.scad`, `.stl`, `.json` summary) were created in `generated_outputs/`.

---

## Phase 7: Analysis & Iteration

*Goal: Analyze results and refine the process.*

-   [ ] **[Analysis] Review Initial Results:**
    -   [ ] Examine the output JSON from the end-to-end test run.
    -   [ ] Identify success/failure patterns for different tasks/LLMs.
    -   [ ] Analyze the geometric similarity scores (`icp_fitness_score`, `geometric_similarity_distance`) and determine if the thresholds/metrics are appropriate.
-   [ ] **[Refine] Update Configuration/Tasks:**
    -   [ ] Adjust `config.yaml` parameters (e.g., tolerances, LLM settings) based on findings.
    -   [ ] Refine task descriptions or add new tasks to target observed failure modes.
    -   [ ] Update reference STLs if issues are found.
-   [ ] **[Refine] Improve Scripts:**
    -   [ ] Enhance error handling, logging, or reporting based on test run observations.
    -   [ ] Optimize any slow steps if necessary.

---