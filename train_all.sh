#!/bin/bash

# Ensure the script exits if any command fails
set -e

# --- Configuration ---
# Option 1: Explicit list of config files
# CONFIG_FILES=(
#     "configs/my_hyperparam_search/dino_r50_no_proto_pretrain/run001_lr0.001_hcl_node.py"
#     "configs/my_hyperparam_search/dino_r50_no_proto_pretrain/run002_lr0.0001_hcl_depth.py"
#     "configs/my_hyperparam_search/dino_r50_with_proto_pretrain_frozen/run003_lr0.001_hcl_node.py"
#     # ... add all your config files
# )

# Option 2: Find all python files in a specific directory (uncomment to use)
CONFIG_DIRS=(
    "configs/test"
    # Add more directories here
)

# Initialize CONFIG_FILES array
CONFIG_FILES=()

# Loop through each directory in CONFIG_DIRS
for DIR in "${CONFIG_DIRS[@]}"; do
    # Find .py files directly under the current directory (not recursive)
    # and add them to the CONFIG_FILES array
    # Use a temporary array to hold files from the current directory to handle spaces in filenames correctly
    CURRENT_FILES=()
    while IFS= read -r -d $'\0' file; do
        CURRENT_FILES+=("$file")
    done < <(find "$DIR" -maxdepth 1 -type f -name "*.py" -print0)
    
    # Sort files from the current directory and add to the main list
    if [ ${#CURRENT_FILES[@]} -gt 0 ]; then
        # Sort and append. Using process substitution and mapfile for robust handling.
        mapfile -t SORTED_CURRENT_FILES < <(printf "%s\n" "${CURRENT_FILES[@]}" | sort)
        CONFIG_FILES+=("${SORTED_CURRENT_FILES[@]}")
    fi
done

# The CONFIG_FILES array is now populated. The rest of the script uses it.

# Training script
# Use dist_train.sh for multi-GPU or if it's your standard way
TRAIN_SCRIPT="./tools/dist_train.sh"
# Or for single GPU:
# TRAIN_SCRIPT="python tools/train.py"

GPUS=4 # Number of GPUs per job for dist_train.sh, or ignored by single GPU script
# MASTER_PORT_START=29500 # For dist_train.sh, if running multiple distributed jobs concurrently

# --- Execution ---
echo "Starting batch of experiments..."

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file $CONFIG_FILE not found. Skipping."
        continue
    fi

    echo "----------------------------------------------------------------------"
    echo "Running experiment with config: $CONFIG_FILE"
    echo "Timestamp: $(date)"
    echo "----------------------------------------------------------------------"

    # Create a unique work_dir based on the config file path
    # e.g., configs/my_hyperparam_search/dino_r50_no_proto_pretrain/run001_lr0.001_hcl_node.py
    # becomes work_dirs/my_hyperparam_search_dino_r50_no_proto_pretrain_run001_lr0.001_hcl_node
    WORK_DIR_NAME=$(echo "$CONFIG_FILE" | sed 's|^configs/||g' | sed 's|/|_|g' | sed 's|.py$||g')
    WORK_DIR="work_dirs/$WORK_DIR_NAME"

    echo "Output directory: $WORK_DIR"

    # Construct the command
    # For dist_train.sh
    CMD="bash $TRAIN_SCRIPT $CONFIG_FILE $GPUS --work-dir $WORK_DIR"
    # For single GPU train.py
    # CMD="$TRAIN_SCRIPT $CONFIG_FILE --work-dir $WORK_DIR"

    # Optional: Add other common arguments like --amp, --auto-scale-lr
    # CMD+=" --amp"
    # CMD+=" --cfg-options randomness.seed=42" # Example of overriding a config option

    echo "Executing: $CMD"
    
    # Execute the command
    if eval $CMD; then
        echo "Successfully completed experiment: $CONFIG_FILE"
    else
        echo "Error during experiment: $CONFIG_FILE. Check logs in $WORK_DIR"
        # Optionally, decide whether to continue or exit the script on error
        # exit 1 # Uncomment to stop on first error
    fi
    echo "----------------------------------------------------------------------"
    # MASTER_PORT_START=$((MASTER_PORT_START + 1)) # Increment port for next distributed job if needed
done

echo "All specified experiments have been processed."