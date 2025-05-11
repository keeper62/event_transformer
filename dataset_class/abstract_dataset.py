from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
import logging

import os

def setup_logger(name: str | None = None) -> logging.Logger:
    """Setup logger that works with PyTorch Lightning."""
    logger = logging.getLogger(name or __name__)
    
    # Clear existing configuration
    logger.handlers.clear()
    logger.propagate = False  # Critical for PL compatibility
    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:  # Main process only
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, template_miner=None, tokenizer=None, test_mode=False):
        self.path = path
        self.template_miner = template_miner
        self.tokenizer = tokenizer
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        self.test_mode = test_mode
        
        self.logger = setup_logger(self.__class__.__name__)
        
        # Read and process data
        self.data = self._read_data(path)  # list of grouped (event_id, message) lists
        self.grouped_data = self._process_groups()
        self.sample_index = self._generate_adaptive_windows()
        
        # Cleanup
        del self.data

    def __len__(self) -> int:
        """Returns the number of samples in the dataset"""
        if not hasattr(self, 'sample_index'):
            return 0
        return len(self.sample_index)

    def _process_groups(self):
        """Process raw data groups into tensors with shape validation"""
        processed_groups = []
        
        for group_idx, group in enumerate(self.data):
            try:
                tokens, sequences = zip(*[
                    (int(event_id), self.tokenizer(message)) 
                    for event_id, message in group
                ])
                
                # Convert to tensors
                token_tensor = torch.tensor(tokens, dtype=torch.long)
                seq_tensor = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in sequences])
                
                # Validate shapes
                if len(tokens) != len(sequences):
                    raise ValueError(f"Group {group_idx}: tokens length {len(tokens)} != sequences length {len(sequences)}")
                
                processed_groups.append((token_tensor, seq_tensor))
                
            except Exception as e:
                self.logger.error(f"Error processing group {group_idx}: {str(e)}")
                raise
                
        return processed_groups

    def _generate_adaptive_windows(self):
        """Generate windows with shape validation"""
        sample_indices = []
        total_window_size = self.context_length + self.prediction_steps
        self.logger.debug(f"Generating windows with context_length={self.context_length}, "
              f"prediction_steps={self.prediction_steps}, "
              f"total_window_size={total_window_size}")

        for group_idx, (tokens, _) in enumerate(self.grouped_data):
            seq_len = len(tokens)
            
            if seq_len <= total_window_size:
                if seq_len < total_window_size:
                    self.logger.warning(f"Warning: Group {group_idx} length {seq_len} < required window size {total_window_size}")
                sample_indices.append((group_idx, 0))
                continue
                
            pos = 0
            while pos + total_window_size <= seq_len:
                # Validate window bounds
                if pos + total_window_size > seq_len:
                    self.logger.warning(f"Invalid window bounds: pos={pos}, seq_len={seq_len}, window_size={total_window_size}")
                    break
                    
                sample_indices.append((group_idx, pos))
                pos += max(1, self.context_length // 4)  # Simplified stride for debugging

        self.logger.debug(f"Generated {len(sample_indices)} windows total")
        return sample_indices

    def __getitem__(self, idx: int):
        group_idx, start_idx = self.sample_index[idx]
        tokens, sequences = self.grouped_data[group_idx]

        # Slice windows with bounds checking
        input_end = start_idx + self.context_length
        
        if input_end + 1 > len(tokens):
            raise ValueError(
                f"Window out of bounds: start_idx={start_idx}, "
                f"input_end={input_end}, output_end={input_end + 1}, "
                f"token_length={len(tokens)}"
            )
        
        input_window = tokens[start_idx:input_end]
        output_window = tokens[start_idx + 1:input_end + 1]
        input_sequences = sequences[start_idx:input_end]
        
        self.logger.debug(f"""Data device types - Input window: {input_window.device}, Output window: {output_window.device}, Input sequence: 
                          {input_sequences.device}""")
        
        return input_window, output_window, input_sequences

    @abstractmethod
    def _read_data(self, path):
        pass
