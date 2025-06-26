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
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class AbstractMultiHostDataset(Dataset, ABC):
    def __init__(self, path, context_length, uncommon_ids = None, template_miner=None, tokenizer=None, test_mode=False):
        self.path = path
        self.template_miner = template_miner
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.test_mode = test_mode
        self.uncommon_ids = uncommon_ids
        
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
        sample_indices = []
        total_window_size = self.context_length + 1
        stride = max(1, self.context_length // 4)
        self.logger.debug(f"Generating windows with context_length={self.context_length}")

        for group_idx, (tokens, _) in enumerate(self.grouped_data):
            seq_len = len(tokens)
            
            if seq_len <= total_window_size:
                if seq_len < total_window_size:
                    self.logger.warning(f"Warning: Group {group_idx} length {seq_len} < required window size {total_window_size}")
                if self.uncommon_ids is None or any(t.item() in self.uncommon_ids for t in tokens):
                    sample_indices.append((group_idx, 0))
                continue
            
            pos = 0
            if self.uncommon_ids is None:
                # Normal sliding windows
                while pos + total_window_size <= seq_len:
                    sample_indices.append((group_idx, pos))
                    pos += stride
            else:
                # Filtered window generation — only include windows with at least one uncommon id
                pos = 0
                while pos + total_window_size <= seq_len:
                    window_token = tokens[pos + total_window_size - 1]
                    if window_token in self.uncommon_ids:
                        sample_indices.append((group_idx, pos))
                    pos += 1

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

class AbstractSingleGroupDataset(Dataset, ABC):
    def __init__(self, path, context_length, uncommon_ids = None, template_miner=None, tokenizer=None, test_mode=False):
        self.path = path
        self.template_miner = template_miner
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.test_mode = test_mode
        self.uncommon_ids = uncommon_ids
        
        self.logger = setup_logger(self.__class__.__name__)
        
        # Read and process data
        self.data = self._read_data(path)  # list of (event_id, message) tuples
        self.tokens, self.sequences = self._process_data()
        self.sample_indices = self._generate_adaptive_windows()
        
        # Cleanup
        del self.data

    def __len__(self) -> int:
        """Returns the number of samples in the dataset"""
        return len(self.sample_indices)
    
    def _process_data(self):
        """Process raw data into tensors with shape validation"""
        try:
            # Process in chunks to avoid memory issues
            batch_size = 10000  # Adjust based on your GPU memory
            token_batches = []
            seq_batches = []
            
            for i in range(0, len(self.data), batch_size):
                batch = self.data[i:i+batch_size]
                tokens, sequences = zip(*[
                    (int(event_id), self.tokenizer(message)) 
                    for event_id, message in batch
                ])
                
                # Convert batch to tensors and move to device
                token_batches.append(torch.tensor(tokens, dtype=torch.long))
                seq_batches.append(torch.stack([
                    torch.tensor(seq, dtype=torch.long) 
                    for seq in sequences
                ]))
            
            # Concatenate batches
            token_tensor = torch.cat(token_batches)
            seq_tensor = torch.cat(seq_batches)
            
            if len(token_tensor) != len(seq_tensor):
                raise ValueError(f"Tokens length {len(token_tensor)} != sequences length {len(seq_tensor)}")
            
            return token_tensor, seq_tensor
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def _generate_adaptive_windows(self):
        total_window_size = self.context_length + 1
        seq_len = len(self.tokens)
        sample_indices = []
        stride = max(1, self.context_length // 4)

        if seq_len <= total_window_size:
            if seq_len < total_window_size:
                self.logger.warning(f"Data length {seq_len} < required window size {total_window_size}")
            # If uncommon_ids is None or tokens include any uncommon_id, add index 0
            if self.uncommon_ids is None or any(t.item() in self.uncommon_ids for t in self.tokens):
                sample_indices.append(0)
            return sample_indices

        if self.uncommon_ids is None:
            # Normal window generation (no filtering)
            pos = 0
            while pos + total_window_size <= seq_len:
                sample_indices.append(pos)
                pos += stride
            self.logger.debug(f"Generated {len(sample_indices)} windows normally")
        else:
            # Filtered window generation — only include windows with at least one uncommon id
            pos = 0
            while pos + total_window_size <= seq_len:
                window_token = self.tokens[pos + total_window_size - 1]
                if window_token in self.uncommon_ids:
                    sample_indices.append(pos)
                pos += 1
            self.logger.info(f"Filtered down to {len(sample_indices)} windows based on uncommon_ids")

        return sample_indices

    def __getitem__(self, idx: int):
        start_idx = self.sample_indices[idx]

        # Slice windows with bounds checking
        input_end = start_idx + self.context_length
        
        if input_end + 1 > len(self.tokens):
            raise ValueError(
                f"Window out of bounds: start_idx={start_idx}, "
                f"input_end={input_end}, output_end={input_end + 1}, "
                f"token_length={len(self.tokens)}"
            )
        
        input_window = self.tokens[start_idx:input_end]
        output_window = self.tokens[start_idx + 1:input_end + 1]
        input_sequences = self.sequences[start_idx:input_end]
        
        self.logger.debug(f"""Data device types - Input window: {input_window.device}, Output window: {output_window.device}, Input sequence: 
                          {input_sequences.device}""")
        
        return input_window, output_window, input_sequences

    @abstractmethod
    def _read_data(self, path):
        pass