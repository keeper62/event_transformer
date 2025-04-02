from models import Transformer, LogTokenizer, load_config
from dataset_class.bgl_dataset import BGLDataset
import torch

def main():
    # Load Transformer model
    config = load_config("configs/base_config.yaml")['base_config']
    
    tokenizer = LogTokenizer("drain3_state.bin")
    tokenizer.load_state()
    
    config['model']['vocab_size'] = tokenizer.get_vocab_size()
    
    # Load dataset
    dataset = BGLDataset(
        path=config['dataset']['path'], 
        prediction_steps=config['model']['prediction_steps'],
        context_length=config['model']['context_length'],
        transform=tokenizer.transform, 
    )
    
    # Load Transformer model
    model = Transformer(config)
    
    ## Load the checkpoint
    #checkpoint = torch.load("save\\first_iteration\\trained_model_with_abs_pos.ckpt", map_location=torch.device("cpu"))
    #
    ## Extract model state_dict
    #state_dict = checkpoint["state_dict"]
    #
    ## Remove the 'model.' prefix if it exists (Lightning sometimes prepends this)
    #state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    #
    #model.load_state_dict(state_dict)
    model.to(config['device']).eval()
    
    data = dataset[0][0]
    
    predicted_events = model.predict(data)
    
    print("\nPredicted Log Events:")
    for step, predicted_events in enumerate([predicted_events]):
        predicted_template = tokenizer.decode_event_id_sequence(predicted_events)
        print(f"Step {step+1}: {predicted_template} (ID: {predicted_events})")
        
if __name__ == "__main__":
    main()
    
# Give a probability prediction of how often the autoregressive has happened to give certainty?