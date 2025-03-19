import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os

def train_model(model, dataloader, optimizer, loss_fn, device, vocab_size, config_name, save_model=False, visualize=False, num_epochs=10):
    model.to(device)
    train_losses = []
    accuracies = []
    result = {}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        all_preds, all_targets = [], []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for batch in progress_bar:
            inputs, targets = batch  # Targets should contain the next n_steps ahead
            inputs, targets = inputs.to(device), targets.to(device)
            
            if targets.min() == targets.max():
                continue  # Skip batch with only one unique target
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            true_vals = targets.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(true_vals)
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = accuracy_score(all_targets, all_preds)
        train_losses.append(avg_loss)
        accuracies.append(accuracy)
        
        result[epoch] = {"targets": all_targets, "predictions": all_preds, "loss": avg_loss, "accuracy": accuracy}
        logging.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    if visualize:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs+1), train_losses, marker='o', linestyle='-', label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs+1), accuracies, marker='o', linestyle='-', label='Accuracy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        
        plt.show()
    
    if save_model:
        os.makedirs("save", exist_ok=True)
        torch.save(model.state_dict(), f"save/trained_model_{config_name}.pth")
        logging.info(f"Model saved as 'trained_model_{config_name}.pth'")
        
    return result