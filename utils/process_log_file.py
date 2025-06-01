import re
import argparse
from tqdm import tqdm
from pathlib import Path
from models import LogTemplateMiner

def update_log_file_with_event_ids(input_path: str, 
                                   output_path: str,
                                   drain_state_path: str,
                                   fallback_hostname: str = "unknown_host",
                                   max_lines: int = None,
                                   show_progress: bool = True):
    """
    Process log file, retrieve event IDs from Drain3, and update log lines with event_id, timestamp, hostname, message.
    Consecutive lines with the same event_id are skipped.
    """
    pattern = r'^(?P<timestamp>\d+(?:\.\d+)?) (?P<hostname>\S+) (?P<message>.+)$'
    
    # Initialize components
    template_miner = LogTemplateMiner(state_path=drain_state_path)
    
    # Read input lines
    with open(input_path, "r", encoding='utf-8', errors='ignore') as f_in:
        lines = f_in.readlines()[:max_lines] if max_lines else f_in
        
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)
        
        print(f"Total lines to process: {total_lines}")
        
        if show_progress:
            lines = tqdm(lines, desc="Updating logs with event IDs")
        
        # Write updated lines to output file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f_out:
            prev_event_id = None
            for line in lines:
                try:
                    match = re.match(pattern, line)
                    if match:
                        groups = match.groupdict()
                        timestamp = groups.get('timestamp', '').strip()
                        hostname = groups.get('hostname', '').strip() or fallback_hostname
                        message = groups.get('message', '').strip()
                        
                        result = template_miner.get_event_id(message)
                        event_id = result if result else 'unknown_event'
                        
                        if event_id == prev_event_id:
                            continue  # Skip consecutive duplicate event_id
                        
                        f_out.write(f"{event_id} {timestamp} {hostname} {message}\n")
                        prev_event_id = event_id
                except Exception as e:
                    print(f"Error processing line: {line.strip()}. Error: {e}")

    if show_progress:
        print(f"\nSuccessfully updated log file with event IDs at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update log file with event IDs from Drain3')
    parser.add_argument('input_path', type=str, help='Path to input log file')
    parser.add_argument('output_path', type=str, help='Path to output log file')
    parser.add_argument('--drain_state', type=str, default="states/drain3_state.bin",
                        help='Path to Drain3 state file')
    parser.add_argument('--fallback_hostname', type=str, default="unknown_host",
                        help='Fallback hostname if missing')
    parser.add_argument('--max_lines', type=int, default=None,
                        help='Maximum number of lines to process')
    parser.add_argument('--no_progress', action='store_false', dest='show_progress',
                        help='Disable progress bars')
    
    args = parser.parse_args()
    
    update_log_file_with_event_ids(
        input_path=args.input_path,
        output_path=args.output_path,
        drain_state_path=args.drain_state,
        fallback_hostname=args.fallback_hostname,
        max_lines=args.max_lines,
        show_progress=args.show_progress
    )

