import torchmetrics
import importlib
from typing import Optional, Dict, Any, Tuple
import logging

from models import Transformer, LogTemplateMiner, LogTokenizer
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl

from torch.functional import F

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os

import logging

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)  

def setup_logger(name: str | None = None) -> logging.Logger:
    """Setup logger that works with PyTorch Lightning."""
    logger = logging.getLogger(name or __name__)
    
    # Clear existing configuration
    logger.handlers.clear()
    logger.propagate = False  # Critical for PL compatibility
    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:  # Main process only    
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class ImportantClassWrapper(torchmetrics.Metric):
    def __init__(self, metric, important_classes, prefix='val/important_'):
        super().__init__()
        self.metric = metric
        self.important_classes = important_classes
        self.prefix = prefix
        self._seen_samples = 0
        self._has_updates = False  # New flag to track updates

    def update(self, preds, target):
        mask = torch.isin(target, self.important_classes)
        if mask.any():
            self.metric.update(preds[mask], target[mask])
            self._seen_samples += mask.sum().item()
            self._has_updates = True  # Mark that we have updates

    def compute(self):
        if not self._has_updates:
            # Return properly formatted metrics with NaN values
            if isinstance(self.metric, torchmetrics.F1Score):
                return {f"{self.prefix}f1": torch.tensor(float('nan'), device=self.device)}
            return {}
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
        self._seen_samples = 0
        self._has_updates = False

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self._logger = setup_logger(self.__class__.__name__)

    def forward(self, inputs, targets):
        """
        More numerically stable implementation
        """
        # Flatten if needed
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.size(-1))
            targets = targets.reshape(-1)

        # Debug shapes
        self._logger.debug(f"FocalLoss shapes - inputs: {inputs.shape}, targets: {targets.shape}")

        # Compute log softmax directly (more stable than softmax + log)
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Get the log probability of the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1))
        log_pt = log_pt.squeeze(1)
        
        # Compute pt (probability of true class)
        pt = log_pt.exp()  # equivalent to softmax then gather
        
        # Compute focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            focal_term = alpha_t * focal_term
        
        # Final loss
        loss = -focal_term * log_pt

        self._logger.debug(f"Loss stats - min: {loss.min().item():.4f}, max: {loss.max().item():.4f}, mean: {loss.mean().item():.4f}")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class DataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], test_mode: bool = False, logger = None):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self._setup_complete = False
        
        self._logger = setup_logger(self.__class__.__name__)
        
        # Initialize components
        self.template_miner = LogTemplateMiner(config['dataset']['drain_path'])
        self.tokenizer = LogTokenizer(
            config['tokenizer']['tokenizer_length'], 
            tokenizer_path=config['tokenizer']['tokenizer_path']
        )
        self.template_miner.load_state()
        
        # Dynamic dataset class loading
        dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
        self.dataset_class = getattr(dataset_module, "Dataset")
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """One-time download/preparation steps"""
        pass  # Add any data downloading/preprocessing here if needed

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets for each stage"""
        if self._setup_complete:
            return
            
        # Initialize full dataset
        self.full_dataset = self.dataset_class(
            path=self.config['dataset']['path'],
            prediction_steps=self.config['model']['prediction_steps'],
            context_length=self.config['model']['context_length'],
            template_miner=self.template_miner.transform,
            tokenizer=self.tokenizer.transform,
            test_mode=self.test_mode
        )

        # Split dataset
        if self.test_mode:
            test_size = min(100, len(self.full_dataset) // 10)
            train_size = len(self.full_dataset) - test_size
            self.train_dataset, self.test_dataset = random_split(
                self.full_dataset, 
                [train_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            self.val_dataset = self.test_dataset
        else:
            val_size = min(int(0.2 * len(self.full_dataset)), 2500)
            train_size = len(self.full_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        self._setup_complete = True
        self._logger.info(f"Dataset setup complete: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")

    def _validate_dataset_shapes(self, dataset):
        """Validate that dataset samples have expected shapes"""
        try:
            for i in range(min(5, len(dataset))):  # Check first 5 samples
                input_window, output_window, sequences = dataset[i]
                
                self._logger.debug(f"Sample {i} shapes - "
                           f"input: {input_window.shape}, "
                           f"output: {output_window.shape}, "
                           f"sequences: {sequences.shape}")
                
                assert len(input_window) == self.config['model']['context_length'], \
                    f"Input window length {len(input_window)} != context_length {self.config['model']['context_length']}"
                
                assert len(output_window) == self.config['model']['prediction_steps'], \
                    f"Output window length {len(output_window)} != prediction_steps {self.config['model']['prediction_steps']}"
                
        except Exception as e:
            self._logger.error(f"Dataset validation failed: {str(e)}")
            raise

    def get_class_weights(self, strategy: str = "inverse", max_samples: int = 1000) -> torch.Tensor:
        """
        Compute class weights to handle imbalance. Supports large datasets via sampling.
        
        Args:
            strategy (str): Weighting strategy:
                - "inverse": 1 / class_count (default)
                - "sqrt": 1 / sqrt(class_count)
                - "balanced": (n_samples / n_classes) / class_count
            max_samples (int): Cap on samples to process (for efficiency).
        
        Returns:
            torch.Tensor: [vocab_size] tensor of class weights.
        """
        if not self._setup_complete:
            self.setup()

        vocab_size = self.config['model']['vocab_size']
        counts = torch.zeros(vocab_size, dtype=torch.long)
        
        # Sample a subset if dataset is large
        n_samples = min(len(self.train_dataset), max_samples)
        indices = torch.randperm(len(self.train_dataset))[:n_samples]
        
        # Process all samples at once if possible (more efficient)
        try:
            # Try to get all targets in one batch
            all_targets = torch.cat([self.train_dataset[idx][0].flatten() for idx in indices])
            unique, counts_batch = torch.unique(all_targets, return_counts=True)
            counts[unique] += counts_batch
        except (MemoryError, RuntimeError):
            # Fall back to per-sample processing if batch processing fails
            for idx in indices:
                _, targets, _ = self.train_dataset[idx]
                unique, counts_batch = torch.unique(targets.flatten(), return_counts=True)
                counts[unique] += counts_batch
        
        # Avoid division by zero for unseen classes
        counts = counts.float() + 1e-6  # Smoothing
        
        # Compute weights based on strategy
        if strategy == "inverse":
            weights = 1.0 / counts
        elif strategy == "sqrt":
            weights = 1.0 / torch.sqrt(counts)
        elif strategy == "balanced":
            weights = (n_samples / vocab_size) / counts
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Normalize to sum to vocab_size (optional)
        weights = weights * (vocab_size / weights.sum())
        
        return weights
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset'].get('num_workers'),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset'].get('num_workers'),
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        if not hasattr(self, 'test_dataset'):
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset'].get('num_workers'),
            pin_memory=True
        )


class TransformerLightning(pl.LightningModule):
    def __init__(self, config: Dict[str, Any], config_name: str, class_weights: torch.Tensor, 
                 important_classes: torch.Tensor | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
        # Model components
        self.model = Transformer(config)
        self.num_classes = config['model']['vocab_size']
        
        self.important_classes = important_classes.long() if important_classes is not None else torch.tensor([], dtype=torch.long)
        self.importance_boost_factor = config['training'].get('importance_boost_factor', 15.0)
        
        # Initialize all metrics
        self._init_metrics()
        
        # Class weights and loss
        self.class_weights = class_weights
        
            # Verify class weights match num_classes
        if len(self.class_weights) != self.num_classes:
            raise ValueError(f"class_weights length ({len(self.class_weights)}) "
                        f"doesn't match num_classes ({self.num_classes})")
        
        # Verify important_classes are valid
        if len(self.important_classes) > 0:
            invalid = self.important_classes[(self.important_classes < 0) | 
                                            (self.important_classes >= self.num_classes)]
            if len(invalid) > 0:
                raise ValueError(f"Invalid important_classes indices: {invalid.cpu().numpy()}")
            
        self.loss_fn = FocalLoss(
            alpha=self.class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            reduction='mean'
        )
        
    def forward(self, inputs: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, sequences) 

    def _init_metrics(self):
        """Initialize all TorchMetrics for evaluation"""
    # Common settings
        metric_args = {
            'task': 'binary'
        }
        self.train_acc = torchmetrics.Accuracy(**metric_args)
        
        # Validation metrics
        self.val_acc = torchmetrics.Accuracy(**metric_args)
        self.val_precision = torchmetrics.Precision(**metric_args)
        self.val_recall = torchmetrics.Recall(**metric_args)
        self.val_f1 = torchmetrics.F1Score(**metric_args)
        
        self.val_top5 = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
            top_k=5
        )
        
        # Important classes (keep original multi-class behavior)
        if self.important_classes is not None:
            self.val_important = ImportantClassWrapper(
                torchmetrics.F1Score(
                    task='multiclass',
                    num_classes=self.num_classes,
                    average='weighted'
                ),
                important_classes=self.important_classes,
                prefix='val/important_'
            )

    def _adjust_class_weights(self, original_weights: torch.Tensor) -> torch.Tensor:
        """Boost weights for important classes (tensor version)"""
        adjusted_weights = original_weights.clone()
        if len(self.important_classes) > 0:
            # Ensure we don't try to boost classes beyond the weight vector length
            valid_mask = self.important_classes < len(adjusted_weights)
            valid_classes = self.important_classes[valid_mask]
            adjusted_weights[valid_classes] *= self.importance_boost_factor
        return adjusted_weights

    def _process_batch(self, batch):
        inputs, targets, sequences = batch
        logits = self.model(inputs, sequences)
        
        logits = logits.half()  
        
        # Reshape logits from [N, B, C] to [N * B, C]
        logits = logits.reshape(-1, logits.shape[-1])

        # Reshape targets from [N, B] to [N * B]
        targets = targets.reshape(-1)
        
        return logits, targets    # Now targets already shifted

    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Get predictions
        preds = logits.argmax(dim=-1)
        is_correct = (preds == targets)  # Binary correctness [N*B]
        
        # Update binary metrics
        self.val_acc.update(is_correct, torch.ones_like(is_correct))  # Targets are all "1" (correct)
        self.val_precision.update(is_correct, torch.ones_like(is_correct))
        self.val_recall.update(is_correct, torch.ones_like(is_correct))
        self.val_f1.update(is_correct, torch.ones_like(is_correct))
        
        # Update top-5 (still multiclass)
        self.val_top5.update(logits, targets)
        
        # Update important class metrics (original multi-class)
        if hasattr(self, 'val_important'):
            self.val_important(preds, targets)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Log all metrics
        metrics = {
            'val/acc': self.val_acc.compute(),
            'val/top5_acc': self.val_top5.compute(),
            'val/precision': self.val_precision.compute(),
            'val/recall': self.val_recall.compute(),
            'val/f1': self.val_f1.compute(),
        }
           
        # Handle important classes
        if hasattr(self, 'val_important'):
            important_metrics = self.val_important.compute()
            if isinstance(important_metrics, dict):
                metrics.update({
                    f"val/important_{k}": v 
                    for k, v in important_metrics.items()
                })
            else:
                metrics['val/important_f1'] = important_metrics
            
            # Convert sample count to float explicitly
            metrics['val/important_samples'] = torch.tensor(
                float(self.val_important._seen_samples),  # Explicit float conversion
                device=self.device
            )
        
        self.log_dict(metrics, prog_bar=False)
        
        # Reset all metrics
        self.val_acc.reset()
        self.val_top5.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        
        if hasattr(self, 'val_important'):
            self.val_important.reset()

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Compute training accuracy (binary)
        preds = logits.argmax(dim=-1)
        is_correct = (preds == targets).float()
        self.train_acc.update(is_correct, torch.ones_like(is_correct))
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            logging.debug(f"\nMemory Usage (Batch {batch_idx}):")
            if torch.cuda.is_available():
                logging.debug(f"GPU Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                logging.debug(f"GPU Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            else:
                import psutil
                process = psutil.Process()
                logging.debug(f"CPU Memory Used: {process.memory_info().rss/1e9:.2f}GB")
                logging.debug(f"System Available: {psutil.virtual_memory().available/1e9:.2f}GB")
            
    def configure_optimizers(self):
        training_cfg = self.hparams.config['training']

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training_cfg.get('lr', 1e-4),
            weight_decay=training_cfg.get('weight_decay', 0.01),
            fused=False
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