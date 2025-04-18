import json
import os
import datetime
import sys

# --- Configuration ---
old_run_id = "20250417_170208"
new_sonnet_run_id = "20250418_093709"
target_model_name = "claude-3-7-sonnet-20250219"
# Use absolute base path
workspace_root = "/Users/willpatrick/Documents/CadEval"
base_results_dir = os.path.join(workspace_root, "results")
# --- End Configuration ---

# Construct absolute file paths
old_results_path = os.path.join(base_results_dir, old_run_id, f"results_{old_run_id}.json")
new_sonnet_results_path = os.path.join(base_results_dir, new_sonnet_run_id, f"results_{new_sonnet_run_id}.json")

# Generate absolute output path
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_run_id = f"merged_run_{timestamp}"
output_dir = os.path.join(base_results_dir, output_run_id)
output_filename = f"results_{output_run_id}.json"
output_path = os.path.join(output_dir, output_filename)

# --- Load Data ---
old_data = []
new_sonnet_data = []

print(f"Attempting to load old results from: {old_results_path}")
# ... (loading logic remains the same) ...
try:
    with open(old_results_path, 'r') as f:
        old_data = json.load(f)
    print(f"-> Loaded {len(old_data)} results.")
except FileNotFoundError:
    print(f"Error: File not found: {old_results_path}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Could not parse JSON from {old_results_path}: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading {old_results_path}: {e}")
    sys.exit(1)


print(f"Attempting to load new Sonnet results from: {new_sonnet_results_path}")
try:
    with open(new_sonnet_results_path, 'r') as f:
        new_sonnet_data = json.load(f)
    print(f"-> Loaded {len(new_sonnet_data)} results.")
except FileNotFoundError:
    print(f"Error: File not found: {new_sonnet_results_path}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Could not parse JSON from {new_sonnet_results_path}: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading {new_sonnet_results_path}: {e}")
    sys.exit(1)


# --- Filter Data ---
print(f"Filtering results...")
# ... (filtering logic remains the same) ...
filtered_old_data = [entry for entry in old_data if entry.get("model_name") != target_model_name]
count_removed = len(old_data) - len(filtered_old_data)
print(f"-> Kept {len(filtered_old_data)} entries from old run (removed {count_removed} '{target_model_name}' entries).")

count_kept_new = len(new_sonnet_data)
all_new_are_target = all(entry.get("model_name") == target_model_name for entry in new_sonnet_data)
if not all_new_are_target:
     print(f"Warning: The new results file {new_sonnet_results_path} contains models other than '{target_model_name}'. Only '{target_model_name}' entries will be added.")
     new_sonnet_data = [entry for entry in new_sonnet_data if entry.get("model_name") == target_model_name]
     count_kept_new = len(new_sonnet_data)

print(f"-> Will add {count_kept_new} '{target_model_name}' entries from new run.")

if count_removed != count_kept_new:
    print(f"Note: The number of removed Sonnet entries ({count_removed}) does not match the number added from the new run ({count_kept_new}). This is expected if the number of tasks/replicates differed or if some failed in the original run.")


# --- Combine Data ---
combined_data = filtered_old_data + new_sonnet_data
print(f"Total combined results: {len(combined_data)} entries.")

# --- Create Output Directory ---
print(f"Attempting to create output directory: {output_dir}") # ADDED DEBUG
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"-> Output directory ensured/created: {output_dir}") # ADDED DEBUG
except OSError as e:
    print(f"Error creating output directory {output_dir}: {e}")
    sys.exit(1)
except Exception as e: # Catch other potential exceptions
    print(f"Unexpected error during directory creation for {output_dir}: {e}")
    sys.exit(1)


# --- Write Combined Data ---
print(f"Attempting to write combined results to: {output_path}") # ADDED DEBUG
try:
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
    print(f"-> Success! Combined results saved to: {output_path}") # ADDED DEBUG
    # Optional: Verify file exists after writing
    if os.path.exists(output_path):
        print("-> Verified file exists after writing.")
    else:
        print("-> WARNING: File does not seem to exist immediately after writing.")

except IOError as e:
    print(f"Error writing combined results to {output_path}: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred writing {output_path}: {e}")
    sys.exit(1)
