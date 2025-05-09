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
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

from torchmetrics import Metric
import torch

class HashedF1(Metric):
    def __init__(self, num_buckets: int = 100):
        super().__init__()
        self.num_buckets = num_buckets
        
        # States for TP, FP, FN (using torch.Tensor)
        self.add_state("tp", default=torch.zeros(num_buckets, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_buckets, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_buckets, dtype=torch.long), dist_reduce_fx="sum")
        
        # Pre-compute hash parameters
        self.register_buffer("hash_prime", torch.tensor(2654435761, dtype=torch.long))  # Large prime number

    def _hash_class(self, class_idx: torch.Tensor) -> torch.Tensor:
        """GPU-compatible hashing using multiplicative hashing"""
        return (class_idx * self.hash_prime) % self.num_buckets

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Move to CPU if needed (hashing is faster on CPU)
        if preds.is_cuda:
            preds_cpu = preds.cpu()
            targets_cpu = targets.cpu()
        else:
            preds_cpu = preds
            targets_cpu = targets
        
        # Vectorized hashing
        pred_hashes = self._hash_class(preds_cpu)
        target_hashes = self._hash_class(targets_cpu)
        
        # Calculate matches
        matches = (preds_cpu == targets_cpu)
        
        # Update TP (correct predictions)
        unique_tp, counts_tp = torch.unique(pred_hashes[matches], return_counts=True)
        self.tp[unique_tp] += counts_tp.to(self.tp.device)
        
        # Update FP (wrong predictions)
        unique_fp, counts_fp = torch.unique(pred_hashes[~matches], return_counts=True)
        self.fp[unique_fp] += counts_fp.to(self.fp.device)
        
        # Update FN (missed predictions)
        unique_fn, counts_fn = torch.unique(target_hashes[~matches], return_counts=True)
        self.fn[unique_fn] += counts_fn.to(self.fn.device)

    def compute(self):
        total_tp = self.tp.sum().float()
        total_fp = self.fp.sum().float()
        total_fn = self.fn.sum().float()
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        return 2 * (precision * recall) / (precision + recall + 1e-8)

    def reset(self):
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
        self._logger = setup_logger(self.__class__.__name__)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, L, C) or (N, C) where C = num_classes
            targets: (N, L) or (N,) containing class indices
        """
        # Flatten sequence dimensions if needed
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.size(-1))  # [N*L, C]
            targets = targets.reshape(-1)  # [N*L]
        
        # Debug shapes
        self._logger.debug(f"FocalLoss shapes - inputs: {inputs.shape}, targets: {targets.shape}")
        
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        
        # Compute loss
        log_probs = torch.log(probs.clamp(min=1e-8))
        ce_loss = -targets_one_hot * log_probs
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss
        
        loss = focal_weight.unsqueeze(1) * ce_loss
        
        self._logger.debug("Loss calculated!")
        
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

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> DataLoader:
        def collate_fn(batch):
            # Add batch shape debugging
            inputs, targets, sequences = zip(*batch)
            
            self._logger.debug(f"Raw batch shapes - "
                       f"inputs: {[x.shape for x in inputs]}, "
                       f"targets: {[x.shape for x in targets]}, "
                       f"sequences: {[x.shape for x in sequences]}")
            
            inputs = torch.stack(inputs)
            targets = torch.stack(targets)
            sequences = torch.stack(sequences)
            
            self._logger.debug(f"Stacked batch shapes - "
                       f"inputs: {inputs.shape}, "
                       f"targets: {targets.shape}, "
                       f"sequences: {sequences.shape}")
            
            return inputs, targets, sequences

        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['dataset'].get('num_workers', 1),
            pin_memory=self.config['dataset'].get('pin_memory', True),
            persistent_workers=self.config['dataset'].get('persistent_workers', True),
            prefetch_factor=2,
            collate_fn=collate_fn,  # Use our custom collate with logging
            drop_last=shuffle
        )

    def get_class_weights(self, strategy: str = "inverse", max_samples: int = 10000) -> torch.Tensor:
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
        
        for idx in indices:
            _, targets, _ = self.train_dataset[idx]
            unique, counts_batch = torch.unique(targets, return_counts=True)
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
        self.save_hyperparameters(ignore=['class_weights', 'important_classes'])
        
        # Model components
        self.model = Transformer(config)
        self.template_miner = LogTemplateMiner(config['dataset']['drain_path'])
        self.tokenizer = LogTokenizer(
            tokenizer_length=config['tokenizer']['tokenizer_length'],
            tokenizer_path=config['tokenizer']['tokenizer_path']
        )
        self.template_miner.load_state()
        self.config_name = config_name
        self.num_classes = config['model']['vocab_size']  # 6220 classes
        
        # Initialize metrics
        self.val_f1 = HashedF1(num_buckets=1000)  # More buckets = better accuracy
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)
        
        # Class tracking setup
        self.important_classes = important_classes.long() if important_classes is not None else torch.tensor([], dtype=torch.long)
        self.importance_boost_factor = config['training'].get('importance_boost_factor', 15.0)
        
        self.class_weights = self._adjust_class_weights(class_weights)
        
        # Loss function
        self.loss_fn = FocalLoss(
            alpha=self.class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            reduction='mean'
        )
        
    def forward(self, inputs: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, sequences) 

    def _adjust_class_weights(self, original_weights: torch.Tensor) -> torch.Tensor:
        """Boost weights for important classes (tensor version)"""
        adjusted_weights = original_weights.clone()
        if len(self.important_classes) > 0:
            # Ensure we don't try to boost classes beyond the weight vector length
            valid_mask = self.important_classes < len(adjusted_weights)
            valid_classes = self.important_classes[valid_mask]
            adjusted_weights[valid_classes] *= self.importance_boost_factor
        return adjusted_weights

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
                    
                    # Debug the new sequence tensor
                    new_sequence = torch.tensor(tokenized, device=device).unsqueeze(1)
                    
                    current_sequences = torch.cat([
                        current_sequences[:, 1:], 
                        new_sequence
                    ], dim=1)

        outputs = torch.stack(outputs, dim=1).float()
        final_targets = targets.reshape(-1)
        
        return outputs, final_targets

    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Get predictions
        preds = logits.argmax(dim=-1).view(-1)
        targets = targets.view(-1)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        
        # Update metrics
        self.val_acc(preds, targets)
        self.val_top5(logits.view(-1, self.num_classes), targets)  # Needs unflattened logits
        self.val_f1(preds, targets)
        
        return loss

    def on_validation_epoch_end(self):
        # Log all metrics
        self.log_dict({
            'val/acc': self.val_acc.compute(),
            'val/top5_acc': self.val_top5.compute(),
            'val/f1': self.val_f1.compute()
        }, prog_bar=True)
        
        # Reset metrics
        self.val_acc.reset()
        self.val_top5.reset()
        self.val_f1.reset()

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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