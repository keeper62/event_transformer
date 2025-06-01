# Determine base directory (current working directory)
$BaseDir = Get-Location

# Define paths relative to the base directory
$INPUT_LOG = Join-Path $BaseDir "hadoop\combined_logs.txt"
$PATH_DRAIN = Join-Path $BaseDir "states\drain3_state_hdfs.bin"
$PATH_TOKENIZER = Join-Path $BaseDir "states\tokenizer_state_hdfs.txt"
$OUTPUT_LOG_TEMPORARY = Join-Path $BaseDir "temp\log_reformatted.log"
$FINAL_OUTPUT = Join-Path $BaseDir "data\hdfs.log"
$REGEX = "^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<message>.+?) (?P<hostname>mesos-\d+)$"

# Create temp directory if it doesn't exist
$tempDir = Join-Path $BaseDir "temp"
if (-not (Test-Path -Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir | Out-Null
}

# Define cleanup action
function Cleanup-TempFile {
    if (Test-Path -Path $OUTPUT_LOG_TEMPORARY) {
        Write-Host "Cleaning up temporary file..."
        Remove-Item -Path $OUTPUT_LOG_TEMPORARY -Force
    }
}

# Function to run a step with error handling
function Run-Step {
    param(
        [string]$Message,
        [scriptblock]$Action
    )
    Write-Host "---- $Message ----"
    try {
        & $Action
        if ($LASTEXITCODE -ne 0) {
            throw "$Message failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Host "ERROR: $_"
        Cleanup-TempFile
        exit 1
    }
}

try {
    # Step 1: process_log_file (first script)
    Run-Step -Message "Running initial process_log_file..." -Action {
        python -m utils.log_reformatter $INPUT_LOG $OUTPUT_LOG_TEMPORARY --custom_pattern $REGEX
    }

    # Step 2: generate_drain_and_tokenizer
    Run-Step -Message "Running generate_drain_and_tokenizer..." -Action {
        python -m utils.generate_drain_and_tokenizer $OUTPUT_LOG_TEMPORARY --drain_output $PATH_DRAIN --tokenizer_output $PATH_TOKENIZER
    }

    # Step 3: process_log_file (with drain state)
    Run-Step -Message "Updating log file with event IDs..." -Action {
        python -m utils.process_log_file $OUTPUT_LOG_TEMPORARY $FINAL_OUTPUT --drain_state $PATH_DRAIN
    }

    Write-Host "Pipeline completed successfully! Final output is at: $FINAL_OUTPUT"
}
finally {
    # Always clean up the temporary file
    Cleanup-TempFile
}
