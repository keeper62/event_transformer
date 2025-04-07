import argparse
import torch
import pytorch_lightning as pl
from models import load_config
from training import DataModule, TransformerLightning
import os
import sys

pl.seed_everything(42, workers=True, verbose=False)
sys.path.append(os.path.abspath("."))  

from pytorch_lightning.callbacks import Callback, EarlyStopping

class EarlyEpochStopping(Callback):
    def __init__(self, min_batches=100, patience=50, tol=0.001):
        """
        Args:
            min_batches (int): Minimum batches before checking for plateaus.
            patience (int): Number of batches to wait before stopping epoch.
            tol (float): Minimum relative change to consider improvement.
        """
        self.min_batches = min_batches
        self.patience = patience
        self.tol = tol
        self.batch_losses = []
        self.batches_without_improvement = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]
        self.batch_losses.append(loss.detach().cpu().item())

        # Only check after `min_batches` have been processed
        if len(self.batch_losses) < self.min_batches:
            return

        # Check if loss has improved significantly
        recent_losses = self.batch_losses[-self.patience:]
        if len(recent_losses) < 2:
            return

        # Compute relative change
        rel_change = abs(recent_losses[-1] - recent_losses[-2]) / (recent_losses[-2] + 1e-8)

        if rel_change < self.tol:
            self.batches_without_improvement += 1
        else:
            self.batches_without_improvement = 0

        # If no improvement for `patience` batches, stop the epoch early
        if self.batches_without_improvement >= self.patience:
            print(f"Stopping epoch early (batch {batch_idx}) due to plateau.")
            trainer.should_stop = True  # Forces current epoch to end

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset at the start of each epoch
        self.batch_losses = []
        self.batches_without_improvement = 0

def train_with_config(config, config_name, num_accelerators, num_nodes, accelerator, test_mode=False):    
    data_module = DataModule(config, test_mode=test_mode)
    model = TransformerLightning(config, config_name, test_mode=test_mode)

    trainer = pl.Trainer(
        max_epochs=1 if test_mode else config['training'].get('num_epochs', 10),  # Reduce epochs in test mode
        devices=num_accelerators, 
        accelerator=accelerator, 
        num_nodes=num_nodes, 
        strategy='ddp',  # PyTorch Lightning automatically handles DDP
        logger=not test_mode,
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision if GPU is available
        callbacks=[
            EarlyEpochStopping(
            min_batches=100,  # Wait at least 100 batches before checking
            patience=15,      # Stop epoch if no improvement in 50 batches
            tol=0.001         # Minimum relative change to count as improvement
            ),
            EarlyStopping(monitor="val/loss", patience=3)  # Normal early stopping
        ]
    )

    trainer.fit(model, datamodule=data_module)
    
    # Save model only if not in test mode
    if not test_mode:
        model_path = f"save/trained_model_{config_name}.ckpt"
        trainer.save_checkpoint(model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Test mode: Training completed on a small subset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model with specified configuration.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to the config file.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of distributed nodes.")
    parser.add_argument("--accelerator", type=str, default="cpu", help="Which accelerator to use.")
    parser.add_argument("--num_accelerators", type=int, default=1, help="Number of GPUs or CPUs to use.")
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode with a small dataset.")

    args = parser.parse_args()
    config, config_name = load_config(args.config), os.path.splitext(os.path.basename(args.config))[0]

    print(f"Training with configuration: {config_name}")
    print(f"Arguments: {args}")
    train_with_config(config['base_config'], config_name, args.num_accelerators, args.num_nodes, args.accelerator, test_mode=args.test_mode)
    print("Training completed successfully.")
