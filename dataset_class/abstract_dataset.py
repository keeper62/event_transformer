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
        
        # Read and process data
        self.data = self._read_data(path)  # list of grouped (event_id, message) lists
        self.grouped_data = self._process_groups()
        self.sample_index = self._generate_adaptive_windows()
        
        # Cleanup
        del self.data

    def _process_groups(self):
        """Process raw data groups into device-aware tensors."""
        processed_groups = []
        
        for group in self.data:
            # Process tokens and sequences
            tokens, sequences = zip(*[
                (int(event_id), self.tokenizer(message)) 
                for event_id, message in group
            ])
            
            # Convert to tensors and move to device
            token_tensor = torch.tensor(
                tokens, 
                dtype=torch.long,  # More efficient than int16 for modern GPUs
            )
            
            seq_tensor = torch.stack([
                torch.tensor(seq, dtype=torch.long)
                for seq in sequences
            ])
            
            processed_groups.append((token_tensor, seq_tensor))
            
        return processed_groups

    def _generate_adaptive_windows(self):
        """Generate windows with adaptive spreading and stride support."""
        sample_indices = []
        
        # Configurable parameters
        base_stride = max(1, self.context_length // 4)
        min_gap = base_stride
        max_gap = self.context_length // 2
        similarity_threshold = 0.6
        total_window_size = self.context_length + self.prediction_steps
        
        for group_idx, (tokens, _) in enumerate(self.grouped_data):
            seq_len = len(tokens)
            
            if seq_len <= total_window_size:
                sample_indices.append((group_idx, 0))
                continue
                
            pos = 0
            while pos + total_window_size <= seq_len:
                sample_indices.append((group_idx, pos))
                current_window = tokens[pos:pos+self.context_length]
                
                # Start with base stride
                next_pos = pos + base_stride
                
                # Content-aware adjustment if possible
                if next_pos + total_window_size <= seq_len:
                    current_unique = torch.unique(current_window)
                    current_set = set(current_unique.cpu().numpy())
                    
                    for lookahead in range(min_gap, min(max_gap, seq_len - pos - total_window_size) + 1):
                        next_window = tokens[pos+lookahead:pos+lookahead+self.context_length]
                        next_unique = torch.unique(next_window)
                        next_set = set(next_unique.cpu().numpy())
                        
                        # Calculate Jaccard similarity
                        intersection = len(current_set & next_set)
                        similarity = intersection / self.context_length
                        
                        if similarity < similarity_threshold:
                            next_pos = pos + lookahead
                            break
                
                pos = max(pos + 1, next_pos)
                
        return sample_indices

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        group_idx, start_idx = self.sample_index[idx]
        tokens, sequences = self.grouped_data[group_idx]
        
        # Slice windows
        input_end = start_idx + self.context_length
        output_end = input_end + self.prediction_steps
        
        input_window = tokens[start_idx:input_end]
        output_window = tokens[input_end:output_end]
        input_sequences = sequences[start_idx:input_end]
        
        return input_window, output_window, input_sequences

    @abstractmethod
    def _read_data(self, path):
        pass
