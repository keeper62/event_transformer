param(
    [string]$INPUT_LOG = "hadoop\combined_logs.txt",
    [string]$PATH_DRAIN = "states\drain3_state_hdfs.bin",
    [string]$PATH_TOKENIZER = "states\tokenizer_state_hdfs.txt",
    [string]$FINAL_OUTPUT = "data\hdfs.log",
    [string]$REGEX = "^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<message>.+?) (?P<hostname>mesos-\d+)$"
)

# Determine base directory (current working directory)
$BaseDir = Get-Location

# Resolve full paths
$INPUT_LOG = Join-Path $BaseDir $INPUT_LOG
$PATH_DRAIN = Join-Path $BaseDir $PATH_DRAIN
$PATH_TOKENIZER = Join-Path $BaseDir $PATH_TOKENIZER
$FINAL_OUTPUT = Join-Path $BaseDir $FINAL_OUTPUT
$OUTPUT_LOG_TEMPORARY = Join-Path $BaseDir "temp\log_reformatted.log"

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
    Run-Step -Message "Running initial process_log_file..." -Action {
        python -m utils.log_reformatter $INPUT_LOG $OUTPUT_LOG_TEMPORARY --custom_pattern "$REGEX"
    }

    Run-Step -Message "Running generate_drain_and_tokenizer..." -Action {
        python -m utils.generate_drain_and_tokenizer $OUTPUT_LOG_TEMPORARY --drain_output $PATH_DRAIN --tokenizer_output $PATH_TOKENIZER
    }

    Run-Step -Message "Updating log file with event IDs..." -Action {
        python -m utils.process_log_file $OUTPUT_LOG_TEMPORARY $FINAL_OUTPUT --drain_state $PATH_DRAIN
    }

    Write-Host "Pipeline completed successfully! Final output is at: $FINAL_OUTPUT"
}
finally {
    Cleanup-TempFile
}
