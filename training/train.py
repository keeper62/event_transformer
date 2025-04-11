import torchmetrics
import importlib
from models import Transformer, LogTokenizer
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, config, test_mode=False):
        super().__init__()
        self.config = config
        self.tokenizer = LogTokenizer(config['dataset']['drain_path'])
        self.test_mode = test_mode  # New flag for testing mode
        
        dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
        self.dataset_class = getattr(dataset_module, "Dataset")

    def setup(self, stage=None):
        dataset = self.dataset_class(
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
        return DataLoader(self.train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

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
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average='micro')
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average='micro')
        
        self.train_top5_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5)
        self.val_top5_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5)
        
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x, timestamps):
        return self.model(x, timestamps)

    def _process_batch(self, batch):
        """Helper to handle input/target processing for the new output shape"""
        inputs, targets, timestamps = batch
        
        # Model now outputs predictions for all positions (batch_size, seq_len, vocab_size)
        outputs = self(inputs, timestamps)  
        
        # Reshape for loss/metrics: (batch_size*seq_len, vocab_size) vs (batch_size*seq_len)
        return (
            outputs.view(-1, outputs.size(-1)),  # Flatten all predictions
            targets.view(-1)                     # Flatten all targets
        )

    def training_step(self, batch, batch_idx):
        if self.test_mode and batch_idx > 5:
            return None

        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Update metrics (operate on flattened outputs)
        self.train_accuracy(logits, targets)
        self.train_top5_acc(logits, targets)
        self.train_f1(logits, targets)

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):        
        self.log('train/acc', self.train_accuracy.compute(), prog_bar=True, sync_dist=True)
        self.log('train/top5_acc', self.train_top5_acc.compute(), prog_bar=False, sync_dist=True)
        self.log('train/f1', self.train_f1.compute(), prog_bar=True, sync_dist=True)
        self.train_accuracy.reset()
        self.train_top5_acc.reset()
        self.train_f1.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)

        self.val_accuracy(logits, targets)
        self.val_top5_acc(logits, targets)
        self.val_f1(logits, targets)

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self.log('val/acc', self.val_accuracy.compute(), prog_bar=True, sync_dist=True)
        self.log('val/top5_acc', self.val_top5_acc.compute(), prog_bar=False, sync_dist=True)
        self.log('val/f1', self.val_f1.compute(), prog_bar=True, sync_dist=True)
        self.val_accuracy.reset()
        self.val_top5_acc.reset()
        self.val_f1.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.config['training'].get('lr', 1e-4))
