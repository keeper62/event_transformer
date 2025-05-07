from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, template_miner=None, tokenizer=None, test_mode=False):
        self.path = path
        self.template_miner = template_miner
        self.tokenizer = tokenizer
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        self.test_mode = test_mode
        
        self.data = self._read_data(path)  # -> list of grouped (event_id, message) lists

        # Process each group into tokens + sequences
        self.grouped_data = []
        for group in self.data:
            processed = [
                (int(i), self.tokenizer(d)) for i, d in group
            ]
            tokens, sequences = zip(*processed)
            sequences = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in sequences])
            tokens = torch.tensor(tokens, dtype=torch.long)
            self.grouped_data.append((tokens, sequences))

        # Generate adaptive window indices for each group
        self.sample_index = self._generate_adaptive_windows()
                
        del self.data  # Free memory after processing

    def _generate_adaptive_windows(self):
        """Generate windows with adaptive spreading based on content"""
        sample_indices = []
        
        for group_idx, (tokens, _) in enumerate(self.grouped_data):
            seq_len = len(tokens)
            total_window_size = self.context_length + self.prediction_steps
            
            if seq_len <= total_window_size:
                # For short sequences, just take the whole sequence
                sample_indices.append((group_idx, 0))
                continue
                
            pos = 0
            while pos + total_window_size <= seq_len:
                # Add current window
                sample_indices.append((group_idx, pos))
                
                # Calculate next position based on content uniqueness
                current_window = tokens[pos:pos+self.context_length]
                min_gap = 1  # Minimum step between windows
                max_gap = self.context_length // 2  # Maximum lookahead
                next_pos = pos + min_gap
                
                # Look ahead for similar content
                for lookahead in range(min_gap, max_gap + 1):
                    if pos + lookahead + total_window_size > seq_len:
                        break
                        
                    next_window = tokens[pos+lookahead:pos+lookahead+self.context_length]
                    
                    # Calculate similarity (using unique elements for efficiency)
                    current_set = set(current_window.tolist())
                    next_set = set(next_window.tolist())
                    similarity = len(current_set & next_set) / self.context_length
                    
                    if similarity < 0.6:  # Threshold for "different" windows
                        next_pos = pos + lookahead
                        break
                
                pos = next_pos
                
        return sample_indices

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        group_idx, start_idx = self.sample_index[idx]
        tokens, sequences = self.grouped_data[group_idx]
        
        input_window = tokens[start_idx : start_idx + self.context_length]
        output_window = tokens[start_idx + self.context_length : start_idx + self.context_length + self.prediction_steps]
        input_sequences = sequences[start_idx : start_idx + self.context_length]

        return input_window, output_window, input_sequences

    @abstractmethod
    def _read_data(self, path):
        pass
