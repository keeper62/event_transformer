import argparse
from pathlib import Path
from models.token import LogTokenizer, LogTemplateMiner
from tqdm import tqdm
import re

def process_log_file(input_path: str, 
                    drain_state_path: str = "states/drain3_state.bin",
                    tokenizer_state_path: str = "states/tokenizer_state.txt",
                    max_lines: int = None,
                    show_progress: bool = True):
    """
    Process log file to generate Drain3 state and tokenizer vocabulary
    
    Args:
        input_path: Path to input log file
        drain_state_path: Output path for Drain3 state
        tokenizer_state_path: Output path for tokenizer state
        max_lines: Maximum number of lines to process (None for all)
        show_progress: Whether to show progress bar
    """
    pattern = r'^(?P<timestamp>\d+(?:\.\d+)?) (?P<hostname>\S+) (?P<message>.+)$'
    
    # Initialize components
    template_miner = LogTemplateMiner(drain_state_path)
    tokenizer = LogTokenizer(tokenizer_length=2323)
    
    # First pass: Extract templates with Drain3 and collect tokenizer data
    with open(input_path, "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[:max_lines] if max_lines else f
        
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Rewind to start of file
        
        print(f"Total lines to process: {total_lines}")
        
        if show_progress:
            lines = tqdm(lines, desc="Processing logs with Drain3 and Tokenizer")
        
        for line in lines:
            try:
                match = re.match(pattern, line)
                if match:
                    log_message = match.group('message')
                    template_miner.add_log_message(log_message)
                    tokenizer.train(log_message.split())  # Accumulate word counts
            except Exception as e:
                print(f"Error processing line: {line.strip()}. Error: {e} "
                    "Make sure the log file is properly formatted using log_formatter.py!")
    
    # Save outputs
    Path(drain_state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tokenizer_state_path).parent.mkdir(parents=True, exist_ok=True)
    
    template_miner.save_state()
    tokenizer.save(tokenizer_state_path)

    if show_progress:
        print(f"\nSuccessfully processed {lines.n} log templates")
        print(f"Drain3 state saved to: {drain_state_path}")
        print(f"Tokenizer state saved to: {tokenizer_state_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process log files to generate Drain3 state and tokenizer vocabulary')
    parser.add_argument('input_path', type=str, help='Path to input log file')
    parser.add_argument('--drain_output', type=str, default="states/drain3_state.bin", 
                       help='Output path for Drain3 state file')
    parser.add_argument('--tokenizer_output', type=str, default="states/tokenizer_state.txt",
                       help='Output path for tokenizer state file')
    parser.add_argument('--max_lines', type=int, default=None,
                       help='Maximum number of lines to process')
    parser.add_argument('--no_progress', action='store_false', dest='show_progress',
                       help='Disable progress bars')
    
    args = parser.parse_args()
    
    process_log_file(
        input_path=args.input_path,
        drain_state_path=args.drain_output,
        tokenizer_state_path=args.tokenizer_output,
        max_lines=args.max_lines,
        show_progress=args.show_progress
    )