import yaml
import torch
from models import Transformer
from data.bgl_dataset import BGLDataset

col_names = [
    "Label", "Timestamp", "Date", "Node", "Time", 
    "NodeRepeat", "Type", "Component", "Level", "Content"
]

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load dataset (raw text logs)
    dataset = BGLDataset(path='data/BGL/BGL.log', labels=col_names, max_lines=50)
    log_messages = dataset.data['Content'].to_list()
    
    # Load Transformer model
    config = load_config("configs/base_config.yaml")['base_config']
    
    model = Transformer(config)
    model.eval()  # Set to evaluation mode
    
    # Predict future event IDs
    with torch.no_grad():
        output = model(log_messages)  # Transformer output (batch_size, seq_len, vocab_size)
        predicted_event_ids = torch.argmax(output, dim=-1)  # Get most probable event ID sequence

    # Convert predicted event IDs back to human-readable templates
    predicted_templates = model.embedding_layer.token_embedding.decode_event_id_sequences(predicted_event_ids)

    print("\nPredicted Future Log Templates:")
    for i, template_seq in enumerate(predicted_templates[:5]):  # Show first 5 predictions
        print(f"Prediction {i+1}:")
        for template in template_seq:
            print(f"  - {template}")
        print("\n")

if __name__ == "__main__":
    main()
