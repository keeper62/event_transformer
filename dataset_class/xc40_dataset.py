from .abstract_dataset import AbstractMultiHostDataset
from collections import defaultdict

class Dataset(AbstractMultiHostDataset):
    def _read_data(self, path):
        """
        Reads pre-sorted log file with format:
        <event_id> <iso_timestamp> <hostname> <message>
        
        Returns: List[List[Tuple[hostname, message]] grouped by event_id
        """
        grouped_logs = defaultdict(list)
        
        with open(path, 'r', errors='replace') as f:
            for line in f:
                try:
                    # Split into 4 parts: event_id, timestamp, hostname, message
                    parts = line.strip().split(' ', 3)
                    if len(parts) < 4:
                        continue
                        
                    event_id, timestamp, hostname, message = parts
                    grouped_logs[hostname].append((timestamp, int(event_id), message))
                except Exception as e:
                    print(f"Skipping malformed line: {line.strip()} | Error: {e}")
                    continue
        sorted_grouped_list = [
            [(event_id, message) for _, event_id, message in sorted(grouped_logs[hostname], key=lambda x: x[0])]
            for hostname in sorted(grouped_logs)
        ]
        
        # Return as list of messages per event_id, maintaining original order
        return list(sorted_grouped_list)[:5]