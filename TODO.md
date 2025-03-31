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

-   [ ] **[Setup] Initialize Project Repository:**
    -   [ ] Ensure Git repository is initialized.
    -   [ ] Confirm primary branch is `main`.
    -   [ ] Create/verify `.gitignore` (ignoring `venv/`, `__pycache__/`, `*.pyc`, `generated_outputs/`, `results/`, OS/IDE specific files like `.DS_Store`, `.vscode/`, `.idea/`).
-   [ ] **[Setup] Create Core Directory Structure:**
    -   [ ] Create `tasks/`
    -   [ ] Create `reference/`
    -   [ ] Create `scripts/`
    -   [ ] Create `generated_outputs/`
    -   [ ] Create `results/`
-   [ ] **[Setup] Establish Python Environment:**
    -   [ ] Create virtual environment (`venv/`).
    -   [ ] Activate virtual environment.
    -   [ ] Create `requirements.txt` (based on dependencies listed in `fromchatgpt.md`, e.g., `openai`, `anthropic`, `google-genai`, `trimesh`, `open3d`, `numpy`, `pyyaml`, etc.).
    -   [ ] Install dependencies: `pip install -r requirements.txt`.
    -   [ ] **[Test]** Verify environment and key package imports (e.g., `import openai`, `import trimesh`, `import yaml`).
-   [ ] **[Setup] Implement Configuration Loading:**
    -   [ ] Create `config.yaml` with structure from `fromchatgpt.md` (Paths, LLM settings, Check settings). *Remember to adjust `openscad.executable_path` for the target system.*
    -   [ ] Develop `scripts/config_loader.py` to load and validate `config.yaml`.
    -   [ ] **[Test]** Create `tests/test_config_loader.py` with unit tests for:
        -   [ ] Successful loading.
        -   [ ] Correct retrieval of nested values.
        -   [ ] Error handling for missing/malformed file.
-   [ ] **[Setup] Implement Basic Logging:**
    -   [ ] Create a utility in `scripts/` (e.g., `scripts/logger_setup.py`) for basic logging configuration (console/file output).
    -   [ ] **[Test]** Write a simple test script or unit test to verify INFO/ERROR messages are logged correctly.

---

## Phase 2: Task Definition & Reference Model Preparation

*Goal: Define the evaluation tasks and prepare the ground truth models.*

-   [ ] **[Task Def] Define YAML Task Format:**
    -   [ ] Document the required YAML structure (`task_id`, `description`, `reference_stl`, `requirements.bounding_box`, `requirements.topology_requirements`).
    -   [ ] *(Optional)* Create a JSON schema (`schemas/task_schema.json`) for validation.
-   [ ] **[Task Def] Create Initial Evaluation Tasks (~10):**
    -   [ ] Develop ~10 diverse task descriptions.
    -   [ ] Create corresponding YAML files in `tasks/` (e.g., `tasks/rect_plate_4holes_simple.yaml`).
-   [ ] **[Task Def] Create/Acquire Reference STL Models:**
    -   [ ] For each task YAML, create or obtain the matching reference STL file.
    -   [ ] Place reference STLs in `reference/` with correct filenames.
    -   [ ] **[Validation]** Manually inspect each reference STL for correctness and geometric soundness.
-   [ ] **[Task Def] Implement Task Loading Script:**
    -   [ ] Develop `scripts/task_loader.py` to scan `tasks/`, parse YAMLs, and return task data.
    -   [ ] *(Optional)* Integrate JSON schema validation.
    -   [ ] **[Test]** Create `tests/test_task_loader.py` with unit tests for:
        -   [ ] Loading valid tasks.
        -   [ ] Correct attribute parsing.
        -   [ ] Handling invalid/non-matching YAMLs.
        -   [ ] Handling empty directory.

---

## Phase 3: LLM Interaction & SCAD Generation

*Goal: Implement the logic to interact with LLM APIs and generate OpenSCAD code.*

-   [ ] **[LLM] Secure API Key Handling:**
    -   [ ] Use `.env` file and `python-dotenv` (from `requirements.txt`) to manage API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Ensure `.env` is in `.gitignore`.
    -   [ ] Update `config_loader.py` or create a utility to load environment variables.
-   [ ] **[LLM] Develop LLM API Client Wrapper(s):**
    -   [ ] Create script(s) in `scripts/` (e.g., `scripts/llm_clients.py`) to abstract interactions with OpenAI, Anthropic, and Google APIs.
    -   [ ] Functions should take prompt text, model name, and other parameters (like temperature, max_tokens from `config.yaml`) and return the generated text or handle errors.
    -   [ ] Implement basic API error handling (network issues, rate limits, invalid responses) and logging as specified in Section 4.1.
    -   [ ] **[Test]** *(Requires API Keys)* Write integration tests (potentially skippable in automated CI without keys) to:
        -   [ ] Send a simple test prompt to each configured LLM.
        -   [ ] Verify a text response is received.
        -   [ ] Test basic error handling (e.g., providing an invalid API key temporarily).
-   [ ] **[LLM] Implement SCAD Generation Logic:**
    -   [ ] Create the main script `scripts/generate_scad.py`.
    -   [ ] This script should:
        -   [ ] Load configuration (`config_loader.py`).
        -   [ ] Load tasks (`task_loader.py`).
        -   [ ] Iterate through each task.
        -   [ ] Iterate through the target LLMs specified in `config.yaml`.
        -   [ ] Format the task `description` into a suitable prompt (consider a simple template initially). *Log the exact prompt used.*
        -   [ ] Call the appropriate LLM client wrapper to get the OpenSCAD code.
        -   [ ] Save the raw generated OpenSCAD code to the `generated_outputs/` directory using the specified naming convention (`{task_id}_{model_name}.scad`). Handle LLM failures gracefully (log error, skip saving).
    -   [ ] **[Test]** Write unit tests for prompt formatting. Write integration tests (can be run manually or configured carefully) for the end-to-end flow of one task with one mock/real LLM call, verifying the `.scad` file is created with expected content.

---

## Phase 4: OpenSCAD Rendering & STL Export

*Goal: Automate the conversion of generated `.scad` files to `.stl` format using OpenSCAD headless mode.*

-   [ ] **[Render] OpenSCAD Path Validation:**
    -   [ ] Add a check in the rendering script or a utility function to verify the `openscad.executable_path` from `config.yaml` points to a valid executable file. Fail early if not found.
    -   [ ] Add a check for the OpenSCAD version (using `--version`) against `openscad.minimum_version`. Log a warning or error if too old, especially regarding summary file support (Section 4.2).
-   [ ] **[Render] Implement Rendering Script:**
    -   [ ] Create `scripts/render_scad.py`.
    -   [ ] This script should:
        -   [ ] Accept a list of `.scad` file paths as input (or scan `generated_outputs/`).
        -   [ ] Load configuration (`config.yaml`).
        -   [ ] For each `.scad` file:
            -   [ ] Define output `.stl` and `.json` (summary) paths in `generated_outputs/`.
            -   [ ] Construct the `openscad` command line arguments precisely as specified in Section 4.2 (using `-q`, `--export-format`, `--backend`, `--summary`, `--summary-file`, `-o`). *Ensure paths are quoted if they might contain spaces.*
            -   [ ] Execute the command using `subprocess.run`.
            -   [ ] Implement the timeout (`openscad.render_timeout_seconds`).
            -   [ ] Capture `stdout`, `stderr`, and `returncode`.
            -   [ ] Determine render status (Success, Compile Error, Timeout).
            -   [ ] Check if the `.stl` file was created.
            -   [ ] Check if the `.json` summary file was created (if expected based on OpenSCAD version/flags).
            -   [ ] Log detailed information: command executed, status, duration, paths to output files, stderr on failure.
        -   [ ] Return structured status information for each file processed.
-   [ ] **[Render] Develop Test SCAD Files:**
    -   [ ] Create a few simple, known-good `.scad` files in a test directory (e.g., `tests/test_data/scad/`).
    -   [ ] Create a simple, known-bad `.scad` file (syntax error).
    -   [ ] Create a `.scad` file likely to take longer to render (for timeout testing, if feasible).
-   [ ] **[Render] Testing:**
    -   [ ] **[Test]** Write unit tests for command construction logic.
    -   [ ] **[Test]** *(Requires OpenSCAD installation)* Write integration tests for `render_scad.py`:
        -   [ ] Test rendering a known-good file -> verify STL (+ JSON summary if applicable) is created, return code 0.
        -   [ ] Test rendering a known-bad file -> verify return code is non-zero, capture stderr.
        -   [ ] Test rendering with a timeout (if a suitable test file exists) -> verify timeout behaviour.
        -   [ ] Test handling if OpenSCAD executable is not found.
        -   [ ] Test handling if input `.scad` file is missing.

---

## Phase 5: Automated Geometry Checks

*Goal: Implement automated checks on the generated STL files based on requirements and comparison with reference models.*

-   [ ] **[Check] Implement `geometry_check.py`:**
    -   [ ] Create `scripts/geometry_check.py`.
    -   [ ] This script/module should define functions to perform each check specified in Section 5.
    -   [ ] Input to the main checking function: paths to generated STL, reference STL, task requirements (dict), rendering status, path to OpenSCAD summary JSON (optional).
    -   [ ] **Check 1: Render Success:** Implement logic based on rendering status input.
    -   [ ] **Check 2: Watertight:** Use `trimesh` (`mesh.is_watertight`). Handle potential errors during mesh loading. *Prerequisite: Render Success.*
    -   [ ] **Check 3: Single Component:** Use `trimesh` (`mesh.body_count`). Compare with `task_requirements['topology_requirements']['expected_component_count']` if present. *Prerequisite: Render Success.*
    -   [ ] **Check 4: Bounding Box Accuracy:**
        -   [ ] Prioritize getting bounding box from OpenSCAD summary JSON if available and valid.
        -   [ ] Fallback to calculating with `trimesh` (`mesh.bounds`).
        -   [ ] Compare extents ([L, W, H]) against `task_requirements['bounding_box']` using the tolerance from `config.yaml` (`geometry_check.bounding_box_tolerance_mm`). *Handle potential dimension ordering issues.*
        -   *Prerequisite: Render Success & valid mesh/summary.*
    -   [ ] **Check 5: Geometric Similarity (Mesh Comparison):**
        -   [ ] Load generated and reference STL using `trimesh` or `open3d`. Handle loading errors.
        -   [ ] Perform ICP alignment (`open3d.pipelines.registration.registration_icp` using default parameters) to align generated to reference.
        -   [ ] Record the ICP fitness score (`icp_result.fitness`).
        -   [ ] Calculate Chamfer Distance (`open3d.legacy.pipelines.registration.evaluate_registration` or dedicated Chamfer function if using newer Open3D) between the *aligned* generated mesh and the reference mesh.
        -   [ ] Record the Chamfer distance.
        -   *Prerequisites: Render Success, both meshes load successfully, ICP runs.*
    -   [ ] The main function should orchestrate these checks, respecting prerequisites, and return a dictionary conforming to the results schema (Section 6), including check results, similarity scores, and any check-phase error messages.
-   [ ] **[Check] Develop Test STL Files:**
    -   [ ] Create/select pairs of simple reference/generated STLs in a test directory (e.g., `tests/test_data/stl/`) exhibiting specific properties:
        -   [ ] Identical models.
        -   [ ] Slightly different models (within tolerance).
        -   [ ] Significantly different models.
        -   [ ] A non-watertight model.
        -   [ ] A model with multiple components.
        -   [ ] Models requiring ICP alignment.
-   [ ] **[Check] Testing:**
    -   [ ] **[Test]** Write unit tests for individual check logic where possible (e.g., bounding box comparison logic).
    -   [ ] **[Test]** *(Requires geometry libraries)* Write integration tests for `geometry_check.py` using the test STL pairs:
        -   [ ] Test Check 1 logic (pass/fail based on input status).
        -   [ ] Test Check 2 on watertight/non-watertight models.
        -   [ ] Test Check 3 on single/multi-component models.
        -   [ ] Test Check 4 with models inside/outside bounding box tolerance.
        -   [ ] Test Check 5 with identical/similar/different models, verifying ICP fitness and Chamfer distance values (or ranges).
        -   [ ] Test prerequisite handling (e.g., ensure similarity isn't calculated if rendering failed).
        -   [ ] Test handling of invalid/unloadable STL files.

---

## Phase 6: Evaluation Execution & Results Collection

*Goal: Orchestrate the end-to-end evaluation process and collect results in the specified JSON format.*

-   [ ] **[Exec] Develop Main Orchestration Script (`run_evaluation.py`):**
    -   [ ] Create `scripts/run_evaluation.py`.
    -   [ ] This script is the main entry point and should:
        -   [ ] Parse command-line arguments (e.g., specify tasks to run, specific LLMs, output file name). Use `click` or `argparse`.
        -   [ ] Load configuration (`config_loader.py`).
        -   [ ] Load specified tasks (`task_loader.py`).
        -   [ ] Initialize results list.
        -   [ ] **Loop through Tasks & LLMs:**
            -   [ ] Call `generate_scad.py` logic (or import relevant functions) to generate the `.scad` file for the current task/LLM combination.
            -   [ ] Record SCAD generation success/failure and path.
        -   [ ] **Batch Render:**
            -   [ ] Collect all successfully generated `.scad` paths.
            -   [ ] Call `render_scad.py` logic (or import function) to render all SCAD files to STL (+ summary JSON).
            -   [ ] Store rendering status (success, compile error, timeout), duration, output paths, and error messages for each attempt.
        -   [ ] **Batch Check:**
            -   [ ] For each successfully rendered STL:
                -   [ ] Gather inputs for `geometry_check.py` (STL paths, reference path from task, task requirements, render status, summary JSON path).
                -   [ ] Call `geometry_check.py` logic (or import function).
                -   [ ] Store the returned check results dictionary.
        -   [ ] **Assemble Final Results:**
            -   [ ] For each task/LLM attempt, create a final JSON object matching the **exact schema** defined in Section 6. Include: task details, LLM config used, timestamps, output paths, render status/results, check results (including similarity scores), and any error messages.
            -   [ ] Append this object to the main results list.
        -   [ ] **Save Results:**
            -   [ ] Generate a unique filename for the results JSON (e.g., including timestamp or run ID).
            -   [ ] Save the results list as a JSON file in the `results/` directory.
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