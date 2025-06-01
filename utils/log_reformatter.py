import re
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List, Union, Pattern
from datetime import datetime

def reformat_log_file(input_path: str, 
                     output_path: str,
                     custom_patterns: Optional[List[Union[str, Pattern]]] = None,
                     fallback_timestamp: str = "unknown_timestamp",
                     fallback_hostname: str = "unknown_hostname",
                     max_lines: Optional[int] = None,
                     timestamp_format: str = "%Y-%m-%d %H:%M:%S,%f",
                     show_progress: bool = True) -> None:
    """
    Reformat log file into structured format with either:
    - timestamp hostname message
    - timestamp message
    
    Args:
        input_path: Path to input log file
        output_path: Path to save reformatted log file
        format_type: Output format - either "timestamp_hostname_message" or "timestamp_message"
        custom_patterns: List of custom regex patterns (strings or compiled patterns) to try
        fallback_timestamp: Text to use when timestamp can't be extracted
        fallback_hostname: Text to use when hostname can't be extracted
        max_lines: Maximum number of lines to process (None for all)
        show_progress: Whether to show progress bar

    Examples:
        Basic usage:
            python log_reformatter.py input.log output.log

        With multiple custom patterns:
            python log_reformatter.py input.log output.log \\
                --custom_pattern '^\\[(?P<timestamp>.*?)\\]\\s+(?P<hostname>\\w+)' \\
                --custom_pattern '^LOG:(?P<timestamp>\\d{8})-(?P<hostname>\\w{4})'

        Custom fallback values:
            python log_reformatter.py input.log output.log \\
                --fallback_timestamp "NO_TIMESTAMP" \\
                --fallback_hostname "NO_HOST"
    """
    # Default patterns (can be extended or overridden by custom_patterns)
    default_patterns = [
        # Syslog format: Dec 31 23:59:59 hostname message
        r'^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<hostname>\S+)\s+(?P<message>.*)$',
        # ISO timestamp format: 2023-12-31T23:59:59.000Z hostname message
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+(?P<hostname>\S+)\s+(?P<message>.*)$',
        # Windows Event Log format: 2023-12-31 23:59:59,INFO,hostname,message
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d+)?)\s*,\s*\S+\s*,\s*(?P<hostname>\S+)\s*,\s*(?P<message>.*)$',
        # Just timestamp and message
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<message>.*)$',
    ]
    
    # Combine default and custom patterns
    patterns = []
    
    # Add custom patterns first (so they take precedence)
    if custom_patterns:
        for pattern in custom_patterns:
            if isinstance(pattern, str):
                patterns.append(re.compile(pattern))
            else:  # Assume it's already a compiled pattern
                patterns.append(pattern)
    
    # Add default patterns
    for pattern in default_patterns:
        patterns.append(re.compile(pattern))
    
    # Prepare output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r", encoding='utf-8', errors='ignore') as f_in, \
         open(output_path, "w", encoding='utf-8') as f_out:
        
        # Get total lines for progress bar if needed
        if show_progress and max_lines is None:
            total_lines = sum(1 for _ in f_in)
            f_in.seek(0)
        else:
            total_lines = max_lines if max_lines else 0
        
        lines = f_in.readlines()[:max_lines] if max_lines else f_in
        
        if show_progress:
            lines = tqdm(lines, desc="Reformatting logs", total=total_lines)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    groups = match.groupdict()
                    timestamp = groups.get('timestamp', '').strip()
                    hostname = groups.get('hostname', '').strip()
                    message = groups.get('message', '').strip()
                    
                    if not timestamp:
                        timestamp = fallback_timestamp
                    else:
                        dt = datetime.strptime(timestamp, timestamp_format)
                        timestamp = dt.timestamp()
                        
                    if not hostname:
                        hostname = fallback_hostname
                    
                    f_out.write(f"{timestamp} {hostname} {message}\n")

    if show_progress:
        print(f"\nSuccessfully reformatted logs to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reformat log files into structured format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""Examples:
        Basic usage:
            python log_reformatter.py input.log output.log
        
        With multiple custom patterns:
            python log_reformatter.py input.log output.log 
            --custom_pattern '^\\[(?P<timestamp>.*?)\\]\s+(?P<hostname>\\w+)' 
            --custom_pattern '^LOG:(?P<timestamp>\\d{8})-(?P<hostname>\\w{4})'

        Custom fallback values:
            python log_reformatter.py input.log output.log 
            --fallback_timestamp "NO_TIMESTAMP" 
            --fallback_hostname "NO_HOST"
        """)
    
    parser.add_argument('input_path', type=str, help='Path to input log file')
    parser.add_argument('output_path', type=str, help='Path to save reformatted log file')
    parser.add_argument('--custom_pattern', type=str, action='append', dest='custom_patterns',
                       help='Custom regex pattern to add (can specify multiple)')
    parser.add_argument('--timestamp_format', type=str, default="%Y-%m-%d %H:%M:%S,%f",
                        help="Timestamp format according to ISO 8601")
    parser.add_argument('--fallback_timestamp', type=str, default="unknown_timestamp",
                       help='Text to use when timestamp cannot be extracted')
    parser.add_argument('--fallback_hostname', type=str, default="unknown_hostname",
                       help='Text to use when hostname cannot be extracted')
    parser.add_argument('--max_lines', type=int, default=None,
                       help='Maximum number of lines to process')
    parser.add_argument('--no_progress', action='store_false', dest='show_progress',
                       help='Disable progress bars')
    
    args = parser.parse_args()
    
    reformat_log_file(
        input_path=args.input_path,
        output_path=args.output_path,
        custom_patterns=args.custom_patterns,
        fallback_timestamp=args.fallback_timestamp,
        fallback_hostname=args.fallback_hostname,
        max_lines=args.max_lines,
        timestamp_format=args.timestamp_format,
        show_progress=args.show_progress
    )
    
# If it returns an empty file, the log file might need to be reprocessed 
# so each individual event is a single line.