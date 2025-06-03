import os
import time
import yaml
import torch
import pandas as pd
from models import Transformer, LogTemplateMiner, LogTokenizer
from pathlib import Path
import math

# Configuration
SAVED_MODELS_DIR = "saved_models"  # Directory containing your .ckpt files
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128  # Can be larger for 1D data
SEQ_LENGTH = 64  # Your sequence length
NUM_WARMUP = 10  # Warmup iterations
NUM_TEST = 100  # Measurement iterations
RESULTS_FILE = 'inference_times_pytorch.csv'

def build_model_from_config(config):
    """Build PyTorch model from config file"""
    return Transformer(config)

def load_lightning_checkpoint(ckpt_path, model):
    """Load weights from Lightning checkpoint into PyTorch model"""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Handle different Lightning checkpoint formats
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Remove 'model.' prefix if present (common in Lightning checkpoints)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def measure_performance(model, input_tensor, sequences):
    """Measure inference time with proper GPU synchronization"""
    # Warmup
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model.predict(input_tensor, sequences)
    
    # Timed runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(NUM_TEST):
        with torch.no_grad():
            _ = model.predict(input_tensor, sequences)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    avg_time = elapsed / NUM_TEST
    return avg_time

def find_checkpoint_files(directory):
    """Find all relevant checkpoint files in directory"""
    return [f for f in Path(directory).glob('trained_model_*_final.ckpt')]

def load_config(config_name, directory):
    """Load YAML config file"""
    config_path = Path(directory) / f"{config_name}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def convert(input, tokenizer, template_miner):
    return torch.stack([torch.tensor(tokenizer.transform(template_miner.decode_event_id_sequence(event_id)), 
                                                      device = DEVICE) for event_id in input.tolist()])

def main():
    ckpt_files = list(Path(SAVED_MODELS_DIR).glob('trained_model_*_final.ckpt'))
    
    if not ckpt_files:
        print(f"No checkpoints found in {SAVED_MODELS_DIR}")
        return
    
    results = []
    # Create random input sequence (long dtype for embedding layers)
    dummy_input = torch.randint(1, 100, (SEQ_LENGTH, BATCH_SIZE), 
                              dtype=torch.long, device=DEVICE)
    
    for ckpt_path in ckpt_files:
        try:
            # Extract config name
            config_name = ckpt_path.stem.replace('trained_model_', '').replace('_final', '')
            config_path = Path("configs") / f"{config_name}.yaml"
            
            print(f"\nProcessing {ckpt_path.name}...")
            
            # Load config and build model
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)['base_config']
                
            template_miner = LogTemplateMiner(config['dataset']['drain_path'])
            config['model']['vocab_size'] = template_miner.get_vocab_size()
            tokenizer = LogTokenizer(config['tokenizer']['tokenizer_length'], config['tokenizer']['tokenizer_path'])
            config['tokenizer']['vocab_size'] = tokenizer.get_vocab_size()
            
            model = build_model_from_config(config)
            model = load_lightning_checkpoint(ckpt_path, model)
            
            # Verify forward pass
            with torch.no_grad():
                sequences = convert(dummy_input[0], tokenizer, template_miner) 
                model.predict(dummy_input[0], sequences)  # Test single sample
            
            # Measure performance
            sequences_input = torch.stack([convert(input, tokenizer, template_miner) for input in dummy_input])
            avg_time = measure_performance(model, dummy_input, sequences_input)
            avg_ms = avg_time * 1000
            samples_per_sec = BATCH_SIZE / avg_time
            
            results.append({
                'config': config_name,
                'checkpoint': ckpt_path.name,
                'avg_time_ms': round(avg_ms, 2),
                'throughput_samples_sec': int(samples_per_sec),
                'batch_size': BATCH_SIZE,
                'sequence_length': SEQ_LENGTH,
                'device': DEVICE
            })
            
            print(f"Results: {avg_ms:.2f}ms per batch | {samples_per_sec:.0f} samples/sec")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {ckpt_path.name}: {str(e)}")
            raise
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == '__main__':
    main()