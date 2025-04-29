import torchmetrics
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

from models import Transformer, LogTemplateMiner, LogTokenizer, compute_class_weights
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

class DataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], test_mode: bool = False):
        """Data module for handling dataset preparation and loading.
        
        Args:
            config: Configuration dictionary
            test_mode: If True, uses smaller dataset for testing
        """
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self._setup_complete = False
        
        # Initialize components
        self.template_miner = LogTemplateMiner(config['dataset']['drain_path'])
        self.tokenizer = LogTokenizer(config['dataset']['vocab_path'])
        self.template_miner.load_state()
        
        # Dynamic dataset class loading
        dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
        self.dataset_class = getattr(dataset_module, "Dataset")
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare datasets for current stage."""
        if self._setup_complete:
            return
            
        full_dataset = self.dataset_class(
            path=self.config['dataset']['path'],
            prediction_steps=self.config['model']['prediction_steps'],
            context_length=self.config['model']['context_length'],
            transform=self.template_miner.transform,
            tokenizer=self.tokenizer,
            test_mode=self.test_mode
        )

        # Split dataset
        if self.test_mode:
            # For testing, use a small subset
            test_size = min(100, len(full_dataset) // 10)
            train_size = len(full_dataset) - test_size
            self.train_dataset, self.test_dataset = random_split(
                full_dataset, 
                [train_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            self.val_dataset = self.test_dataset  # Use test set as val in test mode
        else:
            # Normal train/val split
            val_size = min(int(0.2 * len(full_dataset)), 1000)  # Cap validation size
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        self._setup_complete = True
        logger.info(f"Dataset setup complete: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        if not self._setup_complete:
            self.setup()
        return compute_class_weights(self.train_dataset, self.config['model']['vocab_size'])

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> DataLoader:
        """Helper to create dataloaders with consistent settings."""
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['dataset'].get('num_workers', 4),
            pin_memory=self.config['dataset'].get('pin_memory', True),
            persistent_workers=self.config['dataset'].get('persistent_workers', True),
            drop_last=shuffle  # Only drop last for training
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset:
            return self._create_dataloader(self.test_dataset, shuffle=False)
        return None


class TransformerLightning(pl.LightningModule):
    def __init__(self, config: Dict[str, Any], config_name: str, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.model = Transformer(config)
        self.config_name = config_name
        self.num_classes = config['model']['vocab_size']
        
        # Initialize metrics as attributes but don't compute them yet
        self._init_metrics()
        
        # Loss function
        loss_kwargs = {
            'label_smoothing': config['training'].get('label_smoothing', 0.1),
            'ignore_index': 0
        }
        if class_weights is not None:
            loss_kwargs['weight'] = class_weights
        self.loss_fn = torch.nn.CrossEntropyLoss(**loss_kwargs)
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.validation_step_outputs = []  # Store validation step outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _init_metrics(self):
        """Initialize all metrics without computing them."""
        # Training metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            average='micro'
        )
        
        # Validation metrics
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            average='micro'
        )
        self.val_top5_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            top_k=5
        )
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")

    def _process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        batch_size, seq_len = inputs.size()
        device = inputs.device
        
        outputs = torch.zeros(batch_size, self.model.n_steps, self.num_classes, device=device)
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, 
                              enabled=self.hparams.config['training'].get('mixed_precision', True) and 
                                     torch.cuda.is_available()):
            for step in range(self.model.n_steps):
                logits = self(inputs)
                logits_last = logits[:, -1, :]
                preds = logits_last.argmax(dim=-1)
                outputs[:, step, :] = logits_last
                inputs = torch.cat([inputs[:, 1:], preds.unsqueeze(1)], dim=1)
        
        return outputs.reshape(-1, self.num_classes), targets.reshape(-1)

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Update training metrics
        preds = logits.argmax(dim=-1)
        self.train_acc.update(preds, targets)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Compute and log training accuracy at epoch end
        if self.trainer.sanity_checking:
            return
            
        self.log("train/acc_epoch", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=-1)
        
        # Store outputs for epoch_end
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': targets,
            'logits': logits
        })
        
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs or self.trainer.sanity_checking:
            return
            
        # Aggregate all validation step outputs
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        
        # Update metrics with all validation data
        correct = (all_preds == all_targets).long()
        truth = torch.ones_like(correct)
        
        self.val_acc.update(all_preds, all_targets)
        self.val_top5_acc.update(all_logits, all_targets)
        self.val_f1.update(correct, truth)
        self.val_precision.update(correct, truth)
        self.val_recall.update(correct, truth)
        
        # Track best validation loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        # Log all metrics
        self.log_dict({
            "val/loss": avg_loss,
            "val/acc": self.val_acc.compute(),
            "val/top5_acc": self.val_top5_acc.compute(),
            "val/f1": self.val_f1.compute(),
            "val/precision": self.val_precision.compute(),
            "val/recall": self.val_recall.compute(),
            "val/best_loss": self.best_val_loss,
        }, prog_bar=True)
        
        # Reset metrics and clear outputs
        self.val_acc.reset()
        self.val_top5_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.validation_step_outputs.clear()
        
    def configure_optimizers(self):
        training_cfg = self.hparams.config['training']
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training_cfg.get('lr', 1e-4),
            weight_decay=training_cfg.get('weight_decay', 0.01)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }