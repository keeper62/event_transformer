from models import Transformer, LogTokenizer, load_config
from dataset_class.bgl_dataset import BGLDataset

def main():
    # Load Transformer model
    config = load_config("configs/base_config.yaml")['base_config']
    
    tokenizer = LogTokenizer()
    
    # Load dataset
    dataset = BGLDataset(path=config['dataset']['path'], columns=config['dataset']['columns'], transform=tokenizer.transform(), max_lines=10000, data_column="Content")
    dataset.construct_steps(config['model']['prediction_steps'], config['model']['context_length'])
    
    tokenizer.train_template_miner(dataset.data)
    config['model']['vocab_size'] = tokenizer.get_vocab_size()
    
    # Load Transformer model
    model = Transformer(config)
    #model.load_state_dict(torch.load("trained_model.pth", weights_only=False))
    model.to(config['device']).eval()
    
    data = dataset[0][0]
    
    predicted_events = model.predict(data)
    
    print("\nPredicted Log Events:")
    for step, predicted_events in enumerate(predicted_events):
        predicted_template = tokenizer.decode_event_id_sequence(predicted_events)
        print(f"Step {step+1}: {predicted_template} (ID: {predicted_events})")
        
if __name__ == "__main__":
    main()