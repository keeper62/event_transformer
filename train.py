import argparse
import torch
import pytorch_lightning as pl
from models import Transformer, LogTokenizer, load_config
from dataset_class.bgl_dataset import BGLDataset
from torch.utils.data import DataLoader, random_split
import os
import torchmetrics

pl.seed_everything(42, workers=True, verbose=False)

torch.cuda.empty_cache()
print(torch.cuda.memory_summary()) 

class BGLDataModule(pl.LightningDataModule):
    def __init__(self, config, test_mode=False):
        super().__init__()
        self.config = config
        self.tokenizer = LogTokenizer("drain3_state.bin")
        self.test_mode = test_mode  # New flag for testing mode

    def setup(self, stage=None):
        dataset = BGLDataset(
            path=self.config['dataset']['path'], 
            prediction_steps=self.config['model']['prediction_steps'],
            context_length=self.config['model']['context_length'],
            transform=self.tokenizer.transform, 
            test_mode=self.test_mode  # Pass test mode to dataset
        )
        self.tokenizer.load_state()
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        # Limit dataset size in test mode
        if self.test_mode:
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(min(len(self.train_dataset), 100)))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(min(len(self.val_dataset), 20)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

class TransformerLightning(pl.LightningModule):
    def __init__(self, config, config_name, test_mode=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
        self.config_name = config_name
        self.test_mode = test_mode
        
        num_classes = config['model']['vocab_size']

        # Define metrics with persistent=False to avoid excessive memory usage
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, persistent=False)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, top_k=1, persistent=False)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, persistent=False)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, persistent=False)
        self.top5_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5, persistent=False)
        self.mAP = torchmetrics.AveragePrecision(task="multiclass", num_classes=num_classes, persistent=False)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, persistent=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.test_mode and batch_idx > 5:
            return None

        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        
        preds = torch.argmax(outputs, dim=1).detach()
        targets = targets.detach()
        
        # Compute metrics and detach them
        with torch.no_grad():
            acc = self.accuracy(preds, targets)
            r1 = self.recall(preds, targets)
            prec = self.precision(preds, targets)
            f1 = self.f1_score(preds, targets)
            top5_acc = self.top5_accuracy(preds, targets)
            map_score = self.mAP(preds, targets)
            perplexity = torch.exp(preds)  # Perplexity (PPL)

        # Log metrics (convert tensors to Python scalars to avoid memory issues)
        self.log('train_loss', loss.item(), prog_bar=True, logger=True)
        self.log('train_acc', acc.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('train_R1', r1.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('train_precision', prec.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('train_f1', f1.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('train_top5_acc', top5_acc.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('train_mAP', map_score.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('train_perplexity', perplexity.item(), prog_bar=True, logger=True, sync_dist=True)

        # Free memory
        del inputs, targets, outputs, acc, r1, prec, f1, top5_acc, map_score, perplexity
        torch.cuda.empty_cache()
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)

        # Compute metrics
        acc = self.accuracy(outputs, targets)
        r1 = self.recall(outputs, targets)
        prec = self.precision(outputs, targets)
        f1 = self.f1_score(outputs, targets)
        top5_acc = self.top5_accuracy(outputs, targets)
        map_score = self.mAP(outputs, targets)
        perplexity = torch.exp(loss)

        # Log metrics
        self.log('val_loss', loss.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('val_R1', r1.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', prec.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', f1.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top5_acc.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mAP', map_score.item(), prog_bar=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity.item(), prog_bar=True, logger=True, sync_dist=True)

        # Free memory
        del inputs, targets, outputs, acc, r1, prec, f1, top5_acc, map_score, perplexity
        torch.cuda.empty_cache()

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.config['training'].get('lr', 1e-4))
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()


def train_with_config(config, config_name, num_accelerators, num_nodes, accelerator, test_mode=False):
    data_module = BGLDataModule(config, test_mode=test_mode)
    model = TransformerLightning(config, config_name, test_mode=test_mode)

    trainer = pl.Trainer(
        max_epochs=1 if test_mode else config['training'].get('num_epochs', 10),  # Reduce epochs in test mode
        devices=num_accelerators, 
        accelerator=accelerator, 
        num_nodes=num_nodes, 
        strategy='ddp',  # PyTorch Lightning automatically handles DDP
        logger=not test_mode,
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision if GPU is available
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
