import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

if __name__ == "__main__":
    from models.transformer import Transformer
    from data.dataset import CustomDataset
    from torch.utils.data import DataLoader
    
    # Load config
    import yaml
    with open("configs/base_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Dummy dataset for testing evaluation
    data = torch.randint(0, config['vocab_size'], (100, config['max_len']))
    labels = torch.randint(0, config['vocab_size'], (100, config['max_len']))
    dataset = CustomDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=8)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config).to(device)

    # Evaluate model
    acc = evaluate_model(model, dataloader, device)
    print(f"Model Accuracy: {acc:.4f}")
