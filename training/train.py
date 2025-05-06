import torchmetrics
import importlib
from typing import Optional, Dict, Any, Tuple
import logging

from models import Transformer, LogTemplateMiner, LogTokenizer, compute_class_weights
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl

from torch.functional import F

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
        self.tokenizer = LogTokenizer(config['tokenizer']['tokenizer_length'], tokenizer_path=config['tokenizer']['tokenizer_path'])
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
            template_miner=self.template_miner.transform,
            tokenizer=self.tokenizer.transform,
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
            prefetch_factor=2,  
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
        self.template_miner = LogTemplateMiner(config['dataset']['drain_path'])
        self.tokenizer = LogTokenizer(
            tokenizer_length=config['tokenizer']['tokenizer_length'], 
            tokenizer_path=config['tokenizer']['tokenizer_path']
        )
        self.template_miner.load_state()
        self.config_name = config_name
        self.num_classes = config['model']['vocab_size']
        
        # Initialize multiclass metrics
        self._init_multiclass_metrics()
        
        # Class weights handling
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Multiclass focal loss
        self.loss_fn = self._get_multiclass_focal_loss(
            alpha=self.class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            reduction='mean',
            label_smoothing=config['training'].get('label_smoothing', 0.1)
        )
        
        self.best_val_loss = float('inf')
        self.validation_step_outputs = []

    def _init_multiclass_metrics(self):
        """Initialize proper multiclass metrics."""
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
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average='macro'
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=self.num_classes,
            average='macro'
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=self.num_classes,
            average='macro'
        )

    def _get_multiclass_focal_loss(self, **kwargs):
        """Multiclass focal loss implementation."""
        class FocalLoss(torch.nn.Module):
            def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
                super().__init__()
                if alpha is not None:
                    self.register_buffer('alpha', alpha)
                else:
                    self.alpha = None
                self.gamma = gamma
                self.reduction = reduction
                self.label_smoothing = label_smoothing

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(
                    inputs, 
                    targets,
                    weight=self.alpha,
                    reduction='none',
                    label_smoothing=self.label_smoothing
                )
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** self.gamma * ce_loss

                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                return focal_loss

        if 'alpha' in kwargs and kwargs['alpha'] is not None:
            kwargs['alpha'] = kwargs['alpha'].to(self.device)
        return FocalLoss(**kwargs)

    def _process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets, sequences = batch
        device = inputs.device
        outputs = []

        with torch.amp.autocast(device_type=device.type, enabled=self.hparams.config['training'].get('mixed_precision', True)):
            current_inputs = inputs.clone()
            current_sequences = sequences.clone()

            for step in range(self.model.n_steps):
                logits = self(current_inputs, current_sequences)
                logits_last = logits[:, -1, :]
                outputs.append(logits_last)

                with torch.no_grad():
                    probs = torch.softmax(logits_last, dim=-1)
                    preds = probs.argmax(dim=-1)
                    
                    current_inputs = torch.cat([current_inputs[:, 1:], preds.unsqueeze(1)], dim=1)
                    
                    pred_templates = [self.template_miner.decode_event_id_sequence(p.item()) for p in preds]
                    tokenized = self.tokenizer.batch_transform(pred_templates)
                    current_sequences = torch.cat([
                        current_sequences[:, 1:], 
                        torch.tensor(tokenized, device=device).unsqueeze(1)
                    ], dim=1)

        outputs = torch.stack(outputs, dim=1).float()
        return outputs.reshape(-1, self.num_classes), targets.reshape(-1)

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=-1)
        
        # Update multiclass metrics
        self.val_acc(preds, targets)
        self.val_top5_acc(logits, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        
        self.log("val/loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        self.log_dict({
            "val/acc": self.val_acc.compute(),
            "val/top5_acc": self.val_top5_acc.compute(),
            "val/f1": self.val_f1.compute(),
            "val/precision": self.val_precision.compute(), 
            "val/recall": self.val_recall.compute()
        }, prog_bar=True)
        
        # Reset metrics
        self.val_acc.reset()
        self.val_top5_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        
    def configure_optimizers(self):
        training_cfg = self.hparams.config['training']

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training_cfg.get('lr', 1e-4),
            weight_decay=training_cfg.get('weight_decay', 0.01),
            fused=True  # Enable fused implementation for better performance
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_cfg.get('lr_factor', 0.1),
            patience=training_cfg.get('lr_patience', 3),
            threshold=1e-4,
            min_lr=1e-6
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