from models import LogTokenizer
from pathlib import Path
from datetime import datetime

def read_data(path):
    base_path = Path(path)
    console_logs = []
    message_logs = []
    # Process console logs
    for file_path in (base_path / "console").glob("*"):
        print(file_path)
        with file_path.open(errors='replace' ) as f:
            for line in f:
                parsed = parse_console_log(line.strip())
                if parsed is None:
                    continue
                console_logs.append(parsed)
    # Process message logs
    for file_path in (base_path / "message").glob("*"):
        print(file_path)
        with file_path.open(errors='replace') as f:
            for line in f:
                parsed = parse_message_log(line.strip())
                message_logs.append(parsed)
                
    return combine_and_sort_logs(console_logs, message_logs)

def parse_console_log(line: str) -> dict:
    parts = line.strip().split(' ', 2)  # Split into 3 parts: timestamp, hostname, rest
    if len(parts) < 3:
        return None
    timestamp_str, hostname, message = parts
    # Parse timestamp into Unix time
    try:
        dt = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return None
    unix_timestamp = dt.timestamp()
    return hostname, unix_timestamp, message

def parse_message_log(line: str) -> dict:
    parts = line.strip().split(' ', 6)  # Priority, Version, Timestamp, Host, Service, Rest
    if len(parts) < 6:
        raise ValueError(f"Invalid message log format: {line}")
    timestamp_str = parts[1]
    hostname = parts[2]
    message = parts[6]
    # Parse timestamp into Unix time
    dt = datetime.fromisoformat(timestamp_str)
    unix_timestamp = dt.timestamp()
    return hostname, unix_timestamp, message
    
def combine_and_sort_logs(console_logs, message_logs):
    all_logs = console_logs + message_logs
    # Sort by timestamp (index 1)
    all_logs_sorted = sorted(all_logs, key=lambda x: x[1])
    # Drop hostname, only keep (timestamp, message)
    stripped_logs = [message for _, timestamp, message in all_logs_sorted]
    return stripped_logs

if __name__ == "__main__":
    path = "data/xc40"
    
    tokenizer = LogTokenizer("tokenizer_state_xc40.txt")
    
    raw = read_data(path)

    tokenizer.train(raw)
    tokenizer.save("tokenizer_state_xc40.txt")