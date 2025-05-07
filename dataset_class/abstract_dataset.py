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
        
        # Store raw data in memory-efficient format
        self.raw_data = self._read_data(path)
        
        # Process into memory-friendly structures
        self._process_data()
        
        # Generate adaptive windows
        self.sample_index = self._generate_adaptive_windows()

    def _process_data(self):
        """Process data into memory-efficient format"""
        self.group_metadata = []
        self.all_tokens = []
        self.all_sequences = []
        
        current_token_idx = 0
        current_seq_idx = 0
        
        for group in self.raw_data:
            # Process tokens and sequences
            tokens = []
            sequences = []
            for event_id, message in group:
                tokens.append(int(event_id))
                sequences.append(self.tokenizer(message))
            
            # Store metadata
            group_len = len(tokens)
            self.group_metadata.append({
                'token_start': current_token_idx,
                'seq_start': current_seq_idx,
                'length': group_len
            })
            
            # Extend flat storage
            self.all_tokens.extend(tokens)
            self.all_sequences.extend(sequences)
            
            current_token_idx += group_len
            current_seq_idx += group_len
        
        # Convert to tensors (still on CPU)
        self.all_tokens = torch.tensor(self.all_tokens, dtype=torch.long)
        self.all_sequences = torch.stack([torch.tensor(s, dtype=torch.long) for s in self.all_sequences])
        
        del self.raw_data  # Free memory

    def _generate_adaptive_windows(self):
        """Generate windows with stride and content-awareness"""
        sample_indices = []
        base_stride = max(1, self.context_length // 4)
        
        for group_idx, meta in enumerate(self.group_metadata):
            group_len = meta['length']
            total_window_size = self.context_length + self.prediction_steps
            
            if group_len <= total_window_size:
                sample_indices.append((group_idx, 0))
                continue
                
            pos = 0
            while pos + total_window_size <= group_len:
                sample_indices.append((group_idx, pos))
                
                # Get current window tokens for similarity check
                token_start = meta['token_start'] + pos
                current_window = self.all_tokens[token_start:token_start+self.context_length]
                
                # Adaptive stride logic
                next_pos = pos + base_stride
                max_possible_gap = min(self.context_length // 2, group_len - pos - total_window_size)
                
                for gap in range(base_stride, max_possible_gap + 1):
                    check_pos = pos + gap
                    check_start = meta['token_start'] + check_pos
                    next_window = self.all_tokens[check_start:check_start+self.context_length]
                    
                    # Efficient similarity check
                    if len(set(current_window.tolist()) & set(next_window.tolist())) / self.context_length < 0.6:
                        next_pos = check_pos
                        break
                
                pos = max(pos + 1, next_pos)
                
        return sample_indices

    def __getitem__(self, idx):
        group_idx, start_idx = self.sample_index[idx]
        meta = self.group_metadata[group_idx]
        
        # Calculate absolute positions
        token_start = meta['token_start'] + start_idx
        seq_start = meta['seq_start'] + start_idx
        
        # Extract windows (still on CPU)
        input_window = self.all_tokens[token_start:token_start+self.context_length]
        output_window = self.all_tokens[token_start+self.context_length:token_start+self.context_length+self.prediction_steps]
        input_sequences = self.all_sequences[seq_start:seq_start+self.context_length]
        
        return input_window, output_window, input_sequences

    def collate_fn(self, batch):
        """Custom collate to move only needed data to GPU"""
        input_windows, output_windows, input_sequences = zip(*batch)
        
        # Stack and move to GPU in one operation
        input_windows = torch.stack(input_windows).to(self.device)
        output_windows = torch.stack(output_windows).to(self.device)
        input_sequences = torch.stack(input_sequences).to(self.device)
        
        return input_windows, output_windows, input_sequences

    @property
    def device(self):
        """Helper to get current device"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def _read_data(self, path):
        pass
