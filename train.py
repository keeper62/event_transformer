import argparse
import torch
import pytorch_lightning as pl
from models import Transformer, LogTokenizer, load_config
from dataset_class.bgl_dataset import BGLDataset
from torch.utils.data import DataLoader, random_split
import pickle
import time
import os

class BGLDataModule(pl.LightningDataModule):
    def __init__(self, config, num_workers):
        super().__init__()
        self.config = config
        self.tokenizer = LogTokenizer()
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = BGLDataset(
            path=self.config['dataset']['path'], 
            columns=self.config['dataset']['columns'], 
            transform=self.tokenizer.transform, 
            max_lines=3000,
            data_column="Content"
        )
        dataset.construct_steps(self.config['model']['prediction_steps'], self.config['model']['context_length'])
        self.tokenizer.train_template_miner(dataset.data)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['training']['batch_size'], num_workers=self.num_workers, pin_memory=True, shuffle=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['training']['batch_size'], num_workers=self.num_workers, pin_memory=True, shuffle=False, persistent_workers=True)

class TransformerLightning(pl.LightningModule):
    def __init__(self, config, config_name):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
        self.config_name = config_name
        self.predictions = []
        self.targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        device = inputs.device  # Ensure consistency

        # Ensure inputs & targets are on the correct device
        inputs, targets = inputs.to(device), targets.to(device)

        # Sanity check for target values
        assert targets.min() >= 0 and targets.max() < self.hparams.config['model']['vocab_size'], \
            f"Targets out of range: min={targets.min()}, max={targets.max()}, expected < {self.hparams.config['model']['vocab_size']}"

        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.hparams.config['model']['vocab_size']), targets.view(-1))
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        device = inputs.device  # Ensure consistency

        # Ensure inputs & targets are on the correct device
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.hparams.config['model']['vocab_size']), targets.view(-1))
        
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)

        self.predictions.append(outputs.detach().cpu())  # Move to CPU before storing
        self.targets.append(targets.detach().cpu())

        return loss
    
    def on_validation_epoch_end(self):
        if len(self.predictions) > 0:
            self.predictions = torch.cat(self.predictions, dim=0).cpu()
            self.targets = torch.cat(self.targets, dim=0).cpu()
            
            predicted_values = torch.argmax(self.predictions, dim=-1)
            
            results = {
                "predictions": predicted_values.numpy(),
                "targets": self.targets.numpy()
            }
            os.makedirs("results", exist_ok=True)
            with open(f'results/{self.config_name}_{int(time.time())}.pkl', 'wb') as f:
                pickle.dump(results, f)
        
        self.predictions = []
        self.targets = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.config['training'].get('lr', 1e-4))

def train_with_config(config, config_name, num_gpu, num_nodes, accelerator, num_workers):
    data_module = BGLDataModule(config, num_workers)
    model = TransformerLightning(config, config_name)
    trainer = pl.Trainer(max_epochs=config['training'].get('num_epochs', 10), devices=num_gpu, accelerator=accelerator, num_nodes=num_nodes)
    trainer.fit(model, datamodule=data_module)
    
    # Save model after training
    model_path = f"save/trained_model_{config_name}.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model with specified configuration.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to the config file.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of distributed nodes.")
    parser.add_argument("--accelerator", type=str, default="cpu", help="Which accelerator to use.")
    parser.add_argument("--num_cpu_cores", type=int, default=2, help="Number of CPU cores to use outside of accelerated tasks.")
    parser.add_argument("--num_accelerators", type=int, default=1, help="Number of GPUs or CPUs to use.")
    
    args = parser.parse_args()
    
    config, config_name = load_config(args.config), os.path.splitext(os.path.basename(args.config))[0]
    
    print(f"Training with configuration: {config_name}")
    train_with_config(config['base_config'], config_name, args.num_accelerators, args.num_nodes, args.accelerator, args.num_cpu_cores)
    print("Training completed successfully.")

