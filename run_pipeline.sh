#!/bin/bash
set -e  # Exit on error

# Determine base directory (current working directory)
BaseDir=$(pwd)

# Define paths relative to the base directory (use forward slashes for Unix)
INPUT_LOG="$BaseDir/hadoop/combined_logs.txt"
PATH_DRAIN="$BaseDir/states/drain3_state_hdfs.bin"
PATH_TOKENIZER="$BaseDir/states/tokenizer_state_hdfs.txt"
OUTPUT_LOG_TEMPORARY="$BaseDir/temp/log_reformatted.log"
FINAL_OUTPUT="$BaseDir/data/hdfs.log"
REGEX='^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<message>.+?) (?P<host>mesos-\d+)$'

# Create temp directory if it doesn't exist
mkdir -p "$BaseDir/temp"

# Cleanup function to remove temporary log file if it exists
cleanup_temp_file() {
    if [[ -f "$OUTPUT_LOG_TEMPORARY" ]]; then
        echo "Cleaning up temporary file..."
        rm -f "$OUTPUT_LOG_TEMPORARY"
    fi
}

# Run a step with error handling
run_step() {
    local message="$1"
    shift
    local action=("$@")

    echo "---- $message ----"
    if ! "${action[@]}"; then
        echo "ERROR: $message failed"
        cleanup_temp_file
        exit 1
    fi
}

# Trap to always clean up temporary file on script exit
trap cleanup_temp_file EXIT

# Execute pipeline steps
run_step "Running initial process_log_file..." python -m utils.log_reformatter "$INPUT_LOG" "$OUTPUT_LOG_TEMPORARY" --custom_pattern "$REGEX"

run_step "Running generate_drain_and_tokenizer..." python -m utils.generate_drain_and_tokenizer "$OUTPUT_LOG_TEMPORARY" --drain_output "$PATH_DRAIN" --tokenizer_output "$PATH_TOKENIZER"

run_step "Updating log file with event IDs..." python -m utils.process_log_file "$OUTPUT_LOG_TEMPORARY" "$FINAL_OUTPUT" --drain_state "$PATH_DRAIN"

echo "Pipeline completed successfully! Final output is at: $FINAL_OUTPUT"
