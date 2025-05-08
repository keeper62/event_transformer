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
            sequences = torch.stack([torch.tensor(seq, dtype=torch.torch.int16) for seq in sequences])
            tokens = torch.tensor(tokens, dtype=torch.torch.int16)
            self.grouped_data.append((tokens, sequences))

        # Generate adaptive window indices for each group
        self.sample_index = self._generate_adaptive_windows()
                
        del self.data  # Free memory after processing

    def _generate_adaptive_windows(self):
        """Generate windows with adaptive spreading and stride support"""
        sample_indices = []
        
        # Configurable parameters
        base_stride = max(1, self.context_length // 4)  # Default stride (25% of context length)
        min_gap = base_stride  # Minimum step between windows
        max_gap = self.context_length // 2  # Maximum lookahead
        similarity_threshold = 0.6  # Content similarity threshold
        
        for group_idx, (tokens, _) in enumerate(self.grouped_data):
            seq_len = len(tokens)
            total_window_size = self.context_length + self.prediction_steps
            
            if seq_len <= total_window_size:
                sample_indices.append((group_idx, 0))
                continue
                
            pos = 0
            while pos + total_window_size <= seq_len:
                # Add current window
                sample_indices.append((group_idx, pos))
                
                # Start with base stride
                next_pos = pos + base_stride
                current_window = tokens[pos:pos+self.context_length]
                
                # Only do content-aware adjustment if we have room to look ahead
                if next_pos + total_window_size <= seq_len:
                    # Look ahead for similar content (within stride bounds)
                    for lookahead in range(min_gap, min(max_gap, seq_len - pos - total_window_size) + 1):
                        next_window = tokens[pos+lookahead:pos+lookahead+self.context_length]
                        
                        # Fast similarity check using unique elements
                        current_unique = set(current_window.tolist())
                        next_unique = set(next_window.tolist())
                        similarity = len(current_unique & next_unique) / self.context_length
                        
                        if similarity < similarity_threshold:
                            next_pos = pos + lookahead
                            break
                
                # Ensure we make progress and don't get stuck
                pos = max(pos + 1, next_pos)  # Always move forward by at least 1
                
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
