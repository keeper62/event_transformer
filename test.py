from pathlib import Path
from datetime import datetime

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

    # parts[2] = timestamp
    # parts[3] = hostname
    # parts[5] = rest containing "SESSION - MESSAGE"

    timestamp_str = parts[1]
    hostname = parts[2]
    message = parts[6]

    # Parse timestamp into Unix time
    dt = datetime.fromisoformat(timestamp_str)
    unix_timestamp = dt.timestamp()

    return hostname, unix_timestamp, message

def load_logs(base_dir: str, hostname: str = None) -> tuple:
    base_path = Path(base_dir)
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
                if hostname and parsed[0] != hostname:
                    continue
                console_logs.append(parsed)
    
    # Process message logs
    for file_path in (base_path / "message").glob("*"):
        print(file_path)
        with file_path.open(errors='replace') as f:
            for line in f:
                parsed = parse_message_log(line.strip())
                if hostname and parsed[0] != hostname:
                    continue
                message_logs.append(parsed)

    return combine_and_sort_logs(console_logs, message_logs)

def combine_and_sort_logs(console_logs, message_logs):
    all_logs = console_logs + message_logs
    # Sort by timestamp (index 1)
    all_logs_sorted = sorted(all_logs, key=lambda x: x[1])
    # Drop hostname, only keep (timestamp, message)
    stripped_logs = [(timestamp, message) for _, timestamp, message in all_logs_sorted]
    return stripped_logs

if __name__ == "__main__":
    logs = load_logs("data/xc40", "c0-0c0s0n1")
    print("Loaded logs:", logs[0:10])