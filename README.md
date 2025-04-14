# CadEval: Text-to-CAD Evaluation Framework

This repository contains a framework for evaluating the capability of Large Language Models (LLMs) to generate 3D models (specifically OpenSCAD code or direct STL files) from textual descriptions.

---

## Goal & Scope

- **Goal**: Assess the reliability and geometric accuracy of various LLMs in producing single, simple mechanical parts based on text prompts, including analysis based on task complexity.
- **Scope**:
  - Focus on single-part generation.
  - Input is primarily text descriptions defined in task files.
  - Evaluation involves automated generation, rendering (where applicable), geometric checks against reference models, and visualization of results via a dashboard.
  - Supports standard LLMs (via API like OpenAI, Anthropic, Google) and specialized models like Zoo ML's Text-to-CAD (via CLI).

---

## Workflow Overview

The evaluation process follows these steps:

1.  **Configuration (`config.yaml`)**: Defines models to test, API keys (sourced from `.env`), prompt templates, directories, geometry check thresholds, and evaluation parameters (e.g., number of replicates per task).
2.  **Task Definition (`tasks/*.yaml`)**: Each task YAML file specifies:
    *   `task_id`: Unique identifier.
    *   `description`: The text prompt for the LLM.
    *   `reference_stl`: Path to the corresponding ground truth STL model in `reference/`.
    *   `manual_operations` (Optional): An integer indicating the complexity of the task, used for complexity analysis.
    *   `requirements` (Optional): Can include `bounding_box` or `topology_requirements`.
3.  **Execution (`scripts/run_evaluation.py`)**: This is the main script that orchestrates the evaluation:
    *   Loads configuration (`config.yaml`) and selected tasks (`tasks/`).
    *   Iterates through specified models (from `config.yaml`), prompts (from `config.yaml`), and the requested number of replicates.
    *   **For standard LLMs:** Calls `scripts/generate_scad.py` to interact with LLM APIs, saving generated `.scad` files to `results/{run_id}/scad/`.
    *   **For `zoo_cli`:** Calls the `zoo ml text-to-cad export` command directly, saving the output `.stl` to `results/{run_id}/stl/`.
    *   **Rendering:** For successfully generated `.scad` files, calls OpenSCAD via `scripts/render_scad.py` to render them into `.stl` files in `results/{run_id}/stl/`.
    *   **Geometry Checks:** For all successfully generated/rendered `.stl` files, calls `scripts/geometry_check.py` to perform checks against the reference STL.
    *   Aggregates raw results (generation status, render status, check results, metrics) for each attempt into `results/{run_id}/results_{run_id}.json`. Also creates a log file `results/{run_id}/run_{run_id}.log`.
4.  **Post-Processing (`scripts/process_results.py`)**:
    *   Reads the raw `results_{run_id}.json` file from a specified run.
    *   Calculates summary statistics (`meta_statistics` grouped by model/prompt, `task_statistics` grouped by task).
    *   Calculates complexity analysis based on `manual_operations` from task YAMLs.
    *   Formats the processed data and statistics into `dashboard/dashboard_data.json`.
5.  **Dashboard (`dashboard/`)**:
    *   A web-based dashboard (`dashboard.html`, `dashboard.js`, `dashboard.css`) reads `dashboard_data.json`.
    *   Displays interactive charts (using Chart.js) and tables visualizing model performance across various metrics.

---

## Project Structure

```
CadEval/
├── config.yaml             # Main configuration file
├── environment.yml           # Conda environment definition (or requirements.txt)
├── requirements.txt        # pip requirements file (optional if using conda)
├── requirements-dev.txt    # Optional dev dependencies
├── .env                    # API Keys and environment variables (add to .gitignore!)
├── .gitignore              # Specifies intentionally untracked files
├── README.md               # This file
├── tasks/                  # Task definition YAML files
│   └── *.yaml
├── reference/              # Reference/ground truth STL models
│   └── *.stl
├── scripts/                # Core Python scripts
│   ├── run_evaluation.py     # Main orchestration script
│   ├── process_results.py    # Post-processing for dashboard
│   ├── geometry_check.py     # Performs geometric comparisons
│   ├── render_scad.py        # Handles OpenSCAD rendering
│   ├── generate_scad.py      # Handles LLM API calls for SCAD generation
│   ├── task_loader.py        # Loads task definitions
│   ├── config_loader.py      # Loads config.yaml
│   └── logger_setup.py       # Configures logging
├── results/                # Output directory for evaluation runs
│   └── {run_id}/             # Each run gets its own directory
│       ├── results_{run_id}.json # Raw detailed results for the run
│       ├── run_{run_id}.log      # Log file for the run
│       ├── scad/                 # Generated .scad files (for non-Zoo models)
│       │   └── *.scad
│       └── stl/                  # Generated/rendered .stl files
│           └── *.stl
├── dashboard/              # Web dashboard files
│   ├── dashboard.html
│   ├── dashboard.js
│   ├── dashboard.css
│   └── dashboard_data.json   # Processed data consumed by the dashboard
└── # Other files/dirs (.git, .pytest_cache, tests/, etc.)
```

---

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd CadEval
    ```

2.  **Create Environment using Conda:**
    ```bash
    conda env create -f environment.yml
    conda activate cadeval
    ```

3.  **Install OpenSCAD:** Download and install OpenSCAD (version 2021.01 or later recommended) from [openscad.org](https://openscad.org/). Ensure it's added to your system's PATH or update the `openscad.executable_path` in `config.yaml` accordingly.

4.  **Install ZooML CLI (Optional):** If evaluating the `zoo-ml-text-to-cad` model, follow the installation instructions for the Zoo command-line tools.

5.  **Configure API Keys:**
    *   Create a file named `.env` in the project root directory (if it doesn't exist).
    *   Add your API keys like this:
        ```dotenv
        OPENAI_API_KEY=your_openai_key
        ANTHROPIC_API_KEY=your_anthropic_key
        GOOGLE_API_KEY=your_google_key
        # Add other keys if needed
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file to avoid committing secrets.

6.  **Review `config.yaml`:**
    *   Verify the `openscad.executable_path`.
    *   Add/remove/modify LLM models under `llm.models`. Ensure providers match the API clients used in `scripts/generate_scad.py`.
    *   Adjust prompt templates under `prompts` if desired.
    *   Review geometry check thresholds under `geometry_check`.
    *   Set the desired `evaluation.num_replicates`.

---

## Usage

1.  **Activate Environment:**
    ```bash
    conda activate cadeval
    ```

2.  **Run Evaluation:** Execute the main orchestration script from the project root directory.
    *   **Run all tasks, models, and the 'default' prompt defined in `config.yaml`:**
        ```bash
        python scripts/run_evaluation.py
        ```
    *   **Specify tasks, models, prompts, or run ID:**
        ```bash
        # Run only specific tasks
        python scripts/run_evaluation.py --tasks task_id_1 task_id_2

        # Run only specific models (names must match config.yaml)
        python scripts/run_evaluation.py --models gpt-4o-mini claude-3-5-sonnet-20240620

        # Run using specific prompt template keys (from config.yaml)
        python scripts/run_evaluation.py --prompts concise default

        # Assign a custom run ID
        python scripts/run_evaluation.py --run-id my_test_run_01

        # Combine options
        python scripts/run_evaluation.py --tasks task_id_1 --models o1-2024-12-17 --prompts default --replicates 1 --run-id specific_test
        ```
    *   Output for the run will be created in `results/{run_id}/`.

3.  **Process Results for Dashboard:** After an evaluation run completes, process its raw results JSON.
    ```bash
    # Replace {run_id} with the actual ID of the run you want to process
    # Example: python scripts/process_results.py --results-path results/20231027_153000/results_20231027_153000.json
    python scripts/process_results.py --results-path results/{run_id}/results_{run_id}.json
    ```
    *   This generates/updates `dashboard/dashboard_data.json`.

4.  **View Dashboard:**
    *   Open the `dashboard/dashboard.html` file in your web browser.
    *   The dashboard loads data from `dashboard_data.json` and displays interactive charts and summary tables. Refresh the browser page after running `process_results.py` to see updated data.

5.  **Run Unit Tests (Optional):**
    *   Ensure you have any development dependencies installed (if applicable, e.g., `pip install -r requirements-dev.txt` if it existed and contained `pytest`). You might need to install pytest directly: `pip install pytest`.
    *   Run pytest from the project root directory:
        ```bash
        pytest
        ```

---

## Geometry Checks Performed

The `scripts/geometry_check.py` script performs the following evaluations on successfully generated/rendered STL files:

*   **Watertight:** Checks if the mesh is manifold using Open3D.
*   **Single Component:** Verifies the mesh consists of a single connected component (or the number specified in task requirements).
*   **Bounding Box Accuracy:** Compares the dimensions of the generated model's aligned bounding box (using ICP alignment) to the reference model's bounding box within a tolerance (`geometry_check.bounding_box_tolerance_mm`).
*   **Volume Accuracy:** Compares the volume of the generated model to the reference model's volume within a percentage threshold (`geometry_check.volume_threshold_percent`).
*   **Geometric Similarity (Chamfer Distance):** Calculates the Chamfer distance between generated and reference point clouds after ICP alignment (`geometry_check.chamfer_threshold_mm`). Lower is better.
*   **Hausdorff Distance (95th & 99th Percentile):** Calculates percentile Hausdorff distances after alignment (`geometry_check.hausdorff_threshold_mm` applies to the 95p).
*   **ICP Fitness:** Reports the alignment quality score from the Iterative Closest Point algorithm.

---

## Evaluation Improvements / Future Work

Potential areas for future development and improvement of this evaluation framework include:

*   **Expanded Task Suite:** Increase the number and diversity of tasks to cover a wider range of geometric features, complexities, and potential edge cases.
*   **Prompt Engineering & Optimization:** Investigate the impact of different system prompts, few-shot examples, or providing specific OpenSCAD documentation snippets to improve model performance and reliability.
*   **Multimodal Input:** Extend the framework to evaluate model performance based on visual inputs, such as sketches, technical drawings, or existing images.
*   **Incremental Modification Tasks:**
    *   **Adding Operations:** Evaluate the ability of LLMs to add new features (holes, extrusions, etc.) to existing valid OpenSCAD code or STL models based on text instructions.
    *   **Editing/Fixing:** Assess how well models can correct errors or modify specific features in existing SCAD code based on feedback or new requirements.
*   **STL-to-STL Evaluation:** Evaluate the direct ability of models (like Zoo ML) to generate an STL file that closely matches a reference STL, potentially using different metrics or comparison techniques.

---

*(Refer to git history for older versions of this README.)*

