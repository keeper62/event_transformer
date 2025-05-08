from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, template_miner=None, tokenizer=None, test_mode=False):
        self.path = path
        self.template_miner = template_miner
        self.tokenizer = tokenizer
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        self.test_mode = test_mode
        
        # Read and process data
        self.data = self._read_data(path)  # list of grouped (event_id, message) lists
        self.grouped_data = self._process_groups()
        self.sample_index = self._generate_adaptive_windows()
        
        # Cleanup
        del self.data

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
                logger.debug(f"Error processing group {group_idx}: {str(e)}")
                raise
                
        return processed_groups

    def _generate_adaptive_windows(self):
        """Generate windows with shape validation"""
        sample_indices = []
        total_window_size = self.context_length + self.prediction_steps
        logger.debug(f"Generating windows with context_length={self.context_length}, "
              f"prediction_steps={self.prediction_steps}, "
              f"total_window_size={total_window_size}")

        for group_idx, (tokens, _) in enumerate(self.grouped_data):
            seq_len = len(tokens)
            
            if seq_len <= total_window_size:
                if seq_len < total_window_size:
                    logger.debug(f"Warning: Group {group_idx} length {seq_len} < required window size {total_window_size}")
                sample_indices.append((group_idx, 0))
                continue
                
            pos = 0
            while pos + total_window_size <= seq_len:
                # Validate window bounds
                if pos + total_window_size > seq_len:
                    logger.debug(f"Invalid window bounds: pos={pos}, seq_len={seq_len}, window_size={total_window_size}")
                    break
                    
                sample_indices.append((group_idx, pos))
                pos += max(1, self.context_length // 4)  # Simplified stride for debugging

        print(f"Generated {len(sample_indices)} windows total")
        return sample_indices

    def __getitem__(self, idx: int):
        group_idx, start_idx = self.sample_index[idx]
        tokens, sequences = self.grouped_data[group_idx]
        
        # Debug before slicing
        logger.debug(f"\nSample {idx} from group {group_idx}:")
        logger.debug(f"Original tokens length: {len(tokens)}")
        logger.debug(f"Start index: {start_idx}, context_length: {self.context_length}, prediction_steps: {self.prediction_steps}")

        # Slice windows with bounds checking
        input_end = start_idx + self.context_length
        output_end = input_end + self.prediction_steps
        
        if output_end > len(tokens):
            raise ValueError(
                f"Window out of bounds: start_idx={start_idx}, "
                f"input_end={input_end}, output_end={output_end}, "
                f"token_length={len(tokens)}"
            )
        
        input_window = tokens[start_idx:input_end]
        output_window = tokens[input_end:output_end]
        input_sequences = sequences[start_idx:input_end]
        
        # Debug output shapes
        logger.debug(f"Shapes - input_window: {input_window.shape}, "
              f"output_window: {output_window.shape}, "
              f"input_sequences: {input_sequences.shape}")
        
        return input_window, output_window, input_sequences

    @abstractmethod
    def _read_data(self, path):
        pass
