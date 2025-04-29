from .abstract_dataset import AbstractBGLDataset
from pathlib import Path
from datetime import datetime

class Dataset(AbstractBGLDataset):
    def _read_data(self, path):
        hostname = "c0-0c0s0n1"
        base_path = Path(path)
        console_logs = []
        message_logs = []

        # Process console logs
        for file_path in (base_path / "console").glob("*"):
            with file_path.open(errors='replace' ) as f:
                for line in f:
                    parsed = self.parse_console_log(line.strip())
                    if parsed is None:
                        continue
                    if hostname and parsed[0] != hostname:
                        continue
                    console_logs.append(parsed)

        # Process message logs
        for file_path in (base_path / "message").glob("*"):
            with file_path.open(errors='replace') as f:
                for line in f:
                    parsed = self.parse_message_log(line.strip())
                    if hostname and parsed[0] != hostname:
                        continue
                    message_logs.append(parsed)

        return self.combine_and_sort_logs(console_logs, message_logs)
    
    def parse_console_log(self, line: str) -> dict:
        parts = line.strip().split(' ', 2)  # Split into 3 parts: timestamp, hostname, rest
        if len(parts) < 3:
            return None

        _, hostname, message = parts

        return hostname, message

    def parse_message_log(self, line: str) -> dict:
        parts = line.strip().split(' ', 6)  # Priority, Version, Timestamp, Host, Service, Rest

        if len(parts) < 6:
            raise ValueError(f"Invalid message log format: {line}")

        hostname = parts[2]
        message = parts[6]

        return hostname, message
    
    def combine_and_sort_logs(self, console_logs, message_logs):
        all_logs = console_logs + message_logs
        # Sort by timestamp (index 1)
        all_logs_sorted = sorted(all_logs, key=lambda x: x[1])
        # Drop hostname, only keep (timestamp, message)
        stripped_logs = [message for _, message in all_logs_sorted]
        return stripped_logs