#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [TASK_NAME] --load_run [RUN_NAME] [other arguments]"
    exit 1
}

TASK_NAME=""
LOAD_RUN=""
NUM_ENVS=""
HEADLESS_FLAG=""
REAL_TIME_FLAG=""

# Parse first argument as TASK_NAME (must not start with --)
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    TASK_NAME="$1"
    shift
else
    echo "Error: TASK_NAME is required as the first argument."
    usage
fi

# Parse other arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless)
            HEADLESS_FLAG="--headless"
            shift
            ;;
        --real_time)
            REAL_TIME_FLAG="--real_time"
            shift
            ;;
        --load_run)
            if [[ -n "$2" ]]; then
                LOAD_RUN="--load_run $2"
                shift 2
            else
                echo "Error: --load_run requires an argument."
                usage
            fi
            ;;
        --num_envs)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                NUM_ENVS="--num_envs $2"
                shift 2
            else
                echo "Error: --num_envs requires a valid integer."
                exit 1
            fi
            ;;
        *)
            OTHER_ARGS+="$1 "
            shift
            ;;
    esac
done

# Check mandatory
if [[ -z "$LOAD_RUN" ]]; then
    echo "Error: --load_run is a mandatory argument."
    usage
fi

EXPERIMENT_NAME="g1_${TASK_NAME}"
TASK_PLAY="${TASK_NAME}-play"

echo "Testing $TASK_NAME with $LOAD_RUN"

# Execute
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task "$TASK_PLAY" \
    --experiment_name "$EXPERIMENT_NAME" \
    --video --video_length 500 \
    $HEADLESS_FLAG \
    $REAL_TIME_FLAG \
    $LOAD_RUN \
    $NUM_ENVS \
    $OTHER_ARGS
