import argparse
import torch
import pytorch_lightning as pl
from models import Transformer, LogTokenizer, load_config
from dataset_class.bgl_dataset import BGLDataset
from torch.utils.data import DataLoader, random_split
import os
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Ensures reproducibility

torch.set_float32_matmul_precision('medium') 

class BGLDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = LogTokenizer("drain3_state.bin")

    def setup(self, stage=None):
        dataset = BGLDataset(
            path=self.config['dataset']['path'], 
            prediction_steps=self.config['model']['prediction_steps'],
            context_length=self.config['model']['context_length'],
            transform=self.tokenizer.transform, 
        )
        self.tokenizer.load_state()
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

class TransformerLightning(pl.LightningModule):
    def __init__(self, config, config_name):
        super().__init__()
        set_seed(42)  # Ensure same initialization each time
        self.save_hyperparameters()
        self.model = Transformer(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
        self.config_name = config_name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.hparams.config['model']['vocab_size']), targets.view(-1))
        self.log('train_loss', loss, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.hparams.config['model']['vocab_size']), targets.view(-1))
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.config['training'].get('lr', 1e-4))

def train_with_config(config, config_name, num_accelerators, num_nodes, accelerator):
    set_seed(42)  # Ensure reproducibility before training
    data_module = BGLDataModule(config)
    model = TransformerLightning(config, config_name)

    trainer = pl.Trainer(
        max_epochs=config['training'].get('num_epochs', 10), 
        devices=num_accelerators, 
        accelerator=accelerator, 
        num_nodes=num_nodes, 
        strategy='ddp'  # PyTorch Lightning automatically handles DDP
    )

    trainer.fit(model, datamodule=data_module)
    
    # Save model after training
    model_path = f"save/trained_model_{config_name}.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    set_seed(42)  # Ensures all randomness is controlled globally
    
    parser = argparse.ArgumentParser(description="Train Transformer model with specified configuration.")
    parser.add_argument("--config", type=str, default="configs\\base_config.yaml", help="Path to the config file.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of distributed nodes.")
    parser.add_argument("--accelerator", type=str, default="cpu", help="Which accelerator to use.")
    parser.add_argument("--num_accelerators", type=int, default=-1, help="Number of GPUs or CPUs to use.")

    args = parser.parse_args()
    config, config_name = load_config(args.config), os.path.splitext(os.path.basename(args.config))[0]

    print(f"Training with configuration: {config_name}")
    print(f"Arguments: {args}")
    train_with_config(config['base_config'], config_name, args.num_accelerators, args.num_nodes, args.accelerator)
    print("Training completed successfully.")
