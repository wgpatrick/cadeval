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
    -   [x] Input to the main checking function: paths to generated STL, reference STL, task requirements (dict), rendering status, path to OpenSCAD summary JSON (optional).
    -   [x] **Check 1: Render Success:** Implement logic based on rendering status input.
    -   [x] **Check 2: Watertight:** Use `trimesh` (`mesh.is_watertight`). Handle potential errors during mesh loading. *Prerequisite: Render Success.*
    -   [x] **Check 3: Single Component:** Use `open3d` (`mesh.cluster_connected_triangles`). Compare with `task_requirements['topology_requirements']['expected_component_count']` if present. *Prerequisite: Render Success.*
    -   [x] **Check 4: Bounding Box Accuracy:**
        -   [x] Implement logic comparing the AABB of the **reference STL** against the AABB of the **ICP-aligned generated STL** using `trimesh`. *(Replaced original logic based on testing)*.
        -   [x] Use tolerance from `config.yaml` (`geometry_check.bounding_box_tolerance_mm`).
        -   [x] *Prerequisites: Render Success, Similarity Check Success (provides ICP transform).*
    -   [x] **Check 5: Geometric Similarity (Mesh Comparison):**
        -   [x] Load generated and reference STL using `trimesh` or `open3d`. Handle loading errors.
        -   [x] Perform ICP alignment (`open3d.pipelines.registration.registration_icp`) to align generated to reference.
        -   [x] Record the ICP fitness score (`icp_result.fitness`).
        -   [x] Calculate Chamfer Distance between the *aligned* generated mesh and the reference mesh.
        -   [x] Record the Chamfer distance.
        -   [x] *Prerequisites: Render Success, both meshes load successfully.*
    -   [x] The main function orchestrates these checks, respecting prerequisites, and returns a dictionary conforming to the results schema (Section 6), including check results, similarity scores, and any check-phase error messages. *(Logic updated to run Similarity before Aligned BBox)*.
-   [x] **[Check] Develop Test STL Files:**
    -   [x] Create/select pairs of simple reference/generated STLs in a test directory (e.g., `tests/test_data/stl/`) exhibiting specific properties.
    -   [x] *(Files used implicitly during manual testing)*.
-   [x] **[Check] Testing:**
    -   [x] **[Test]** Write unit tests for individual check logic where possible. *(Implicitly tested via integration)*
    -   [x] **[Test]** *(Requires geometry libraries)* Write integration tests for `geometry_check.py` using the test STL pairs. *(Manual integration testing performed)*.
    -   [x] **RESOLVED:** Initial bounding box check (Check 4) failed unexpectedly. Replaced logic with aligned comparison (Ref STL vs Aligned Gen STL), which now works correctly with appropriate tolerance.

---

## Phase 6: Evaluation Execution & Results Collection

*Goal: Orchestrate the end-to-end evaluation process and collect results in the specified JSON format.*

-   [x] **[Exec] Develop Main Orchestration Script (`run_evaluation.py`):**
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
        -   [x] **Enhance Logging:** Implement detailed INFO-level logging for user feedback, showing progress through tasks, models, generation, rendering, and checking steps. Ensure logs correctly output to console based on arguments. Log final results file path.
-   [ ] **[Exec] End-to-End Testing:**
    -   [x] **[Test]** Perform a dry run using 1-2 simple test tasks and potentially mock LLM/render/check functions to verify the orchestration flow.
    -   [x] **[Test]** Perform a small end-to-end run using 1-2 actual tasks, 1-2 LLMs (if keys available), and the real OpenSCAD/geometry checks. *(Run performed, see results_20250402_102950.json)*
        -   [x] Verify the `results/<run_id>.json` file is created.
        -   [x] Manually inspect the JSON output to confirm it matches the schema in Section 6 and contains plausible data for each step (paths, statuses, check results). *(Checked results_20250402_110920.json - Schema OK, data plausible, icp_fitness_score fixed.)*
        -   [x] Verify intermediate files (`.scad`, `.stl`, `.json` summary) were created in `generated_outputs/`. *(Paths look correct in JSON, assumed created)*

---

## Phase 7: Analysis & Iteration

*Goal: Analyze results and refine the process.*

-   [ ] **[Analysis] Review Initial Results:**
    -   [x] Examine the output JSON from the end-to-end test run (`task1` specifically). *(Completed for bounding box fix verification)*.
    -   [ ] Identify success/failure patterns for different tasks/LLMs across the *full initial task set*. *(Pending broader run)*.
    -   [ ] Analyze the geometric similarity scores (`icp_fitness_score`, `geometric_similarity_distance`) and determine if the thresholds/metrics are appropriate across the *full initial task set*. *(Pending broader run)*.
-   [ ] **[Refine] Update Configuration/Tasks:** *(Pending broader analysis)*.
    -   [ ] Adjust `config.yaml` parameters (e.g., tolerances, LLM settings) based on findings from the full task set.
    -   [ ] Refine task descriptions or add new tasks to target observed failure modes.
    -   [ ] Update reference STLs if issues are found.
-   [ ] **[Refine] Improve Scripts:**
    -   [x] Enhance error handling, logging, or reporting based on test run observations. *(Completed for `geometry_check.py` related to bbox fix)*.
    -   [ ] Optimize any slow steps if necessary based on broader run observations.

---