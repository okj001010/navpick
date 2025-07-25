#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [TASK_NAME] --run_name [RUN_NAME] [other arguments]"
    exit 1
}

TASK_NAME=""
RUN_NAME=""
LOG="--logger wandb"
HEADLESS="--headless"

# Parse first argument as TASK_NAME (must not start with --)
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    TASK_NAME="$1"
    shift
else
    echo "Error: TASK_NAME is required as the first argument."
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                RUN_NAME="$2"
                shift 2
            else
                echo "Error: --run_name requires a non-empty argument."
                usage
            fi
            ;;
        --no-log)
            LOG=""
            shift
            ;;
        --no-headless)
            HEADLESS=""
            shift
            ;;
        *)
            OTHER_ARGS+="$1 "
            shift
            ;;
    esac
done

if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is a mandatory argument."
    usage
fi

LOG_PROJECT_NAME="$TASK_NAME"
TASK_TRAIN="${TASK_NAME}-train"

echo "Training $TASK_NAME with $RUN_NAME"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK_TRAIN" \
    --log_project_name "$LOG_PROJECT_NAME" \
    --max_iterations 50000 \
    $LOG \
    $HEADLESS \
    --run_name "$RUN_NAME" $OTHER_ARGS
