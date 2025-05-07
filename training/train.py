import torchmetrics
import importlib
from typing import Optional, Dict, Any, Tuple
import logging

from models import Transformer, LogTemplateMiner, LogTokenizer
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl

from torch.functional import F

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing as mp

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', num_classes=None):
        """
        PyTorch implementation of Focal Loss.
        
        Args:
            alpha (Tensor, optional): Class weighting tensor (size [C]). If None, no weighting.
            gamma (float): Focusing parameter (gamma >= 0). Higher gamma reduces easy example impact.
            label_smoothing (float): Smoothing factor for one-hot labels (0 <= label_smoothing < 1).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Dict, Any, Optional
import importlib
import logging

logger = logging.getLogger(__name__)

class DataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], test_mode: bool = False):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self._setup_complete = False
        
        # Initialize components
        self.template_miner = LogTemplateMiner(config['dataset']['drain_path'])
        self.tokenizer = LogTokenizer(
            config['tokenizer']['tokenizer_length'],
            tokenizer_path=config['tokenizer']['tokenizer_path']
        )
        self.template_miner.load_state()
        
        # Dynamic dataset class loading with error handling
        try:
            dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
            self.dataset_class = getattr(dataset_module, "Dataset")
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load dataset class: {e}")

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
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
            test_size = min(100, len(full_dataset) // 10)
            train_size = len(full_dataset) - test_size
            self.train_dataset, self.test_dataset = random_split(
                full_dataset, 
                [train_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            self.val_dataset = self.test_dataset
        else:
            val_size = min(int(0.2 * len(full_dataset)), 2500)
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        self._setup_complete = True
        logger.info(f"Dataset setup complete: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")

    def _safe_collate(self, batch):
        """Worker-safe collate that handles both single and multi-GPU"""
        inputs, targets, seqs = zip(*batch)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return (
            torch.stack(inputs).to(device, non_blocking=True),
            torch.stack(targets).to(device, non_blocking=True),
            torch.stack(seqs).to(device, non_blocking=True)
        )

    def _create_dataloader(self, dataset, shuffle):
        """Create dataloader with proper distributed training support"""
        # Disable persistent workers in DDP mode
        use_persistent = self.config['dataset'].get('persistent_workers', False)
        if torch.distributed.is_initialized():
            use_persistent = False
            
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['dataset'].get('num_workers', min(4, mp.cpu_count())),
            pin_memory=True,
            persistent_workers=use_persistent,
            collate_fn=self._safe_collate,
            drop_last=shuffle,
            multiprocessing_context=mp.get_context('spawn') if torch.cuda.is_available() else None
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset:
            return self._create_dataloader(self.test_dataset, shuffle=False)
        return None

    def get_class_weights(self, strategy: str = "inverse", max_samples: int = 10000) -> torch.Tensor:
        """Optimized class weight calculation with memory safety"""
        if not self._setup_complete:
            self.setup()

        vocab_size = self.config['model']['vocab_size']
        counts = torch.zeros(vocab_size, dtype=torch.long)
        
        # Use batched sampling for memory efficiency
        batch_size = 1024
        n_samples = min(len(self.train_dataset), max_samples)
        indices = torch.randperm(len(self.train_dataset))[:n_samples]
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            targets = torch.cat([self.train_dataset[idx][1].unsqueeze(0) for idx in batch_indices])
            unique, counts_batch = torch.unique(targets, return_counts=True)
            counts[unique] += counts_batch

        counts = counts.float() + 1e-6
        
        if strategy == "inverse":
            weights = 1.0 / counts
        elif strategy == "sqrt":
            weights = 1.0 / torch.sqrt(counts)
        elif strategy == "balanced":
            weights = (n_samples / vocab_size) / counts
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return (weights * (vocab_size / weights.sum())).detach()

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up resources"""
        if hasattr(self, 'train_dataset'):
            del self.train_dataset
        if hasattr(self, 'val_dataset'):
            del self.val_dataset
        if hasattr(self, 'test_dataset'):
            del self.test_dataset
        self._setup_complete = False

class TransformerLightning(pl.LightningModule):
    def __init__(self, config: Dict[str, Any], config_name: str, class_weights: torch.Tensor, important_classes: torch.Tensor | None = None):
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
        self.num_classes = config['model']['vocab_size']
        
        self.important_classes = important_classes.long() if important_classes is not None else torch.tensor([], dtype=torch.long)
        self.importance_boost_factor = config['training'].get('importance_boost_factor', 15.0)
        
        self.class_weights = self._adjust_class_weights(class_weights)
        self._init_important_metrics()
        
        # Initialize metrics
        self._init_metrics()
        
        # Loss function
        self.loss_fn = FocalLoss(
            alpha=self.class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            reduction='mean',
            num_classes=self.num_classes
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.validation_step_outputs = []
        
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

    def _init_metrics(self):
        """Initialize metrics with better handling for multi-class scenario."""
        # Main metrics - using macro average which is better for imbalanced classes
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=self.num_classes,
            average='micro'
        )
        
        # Precision and Recall with macro average
        self.val_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=self.num_classes,
            average='weighted',
            ignore_index=-1  # Ignore padding if any
        )
        
        self.val_recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=self.num_classes,
            average='weighted',
            ignore_index=-1
        )
        
        # F1 scores - both macro and weighted
        self.val_f1_macro = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average='macro'
        )
        
        self.val_f1_weighted = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average='weighted'
        )
        
        # For class distribution analysis
        self.val_confmat = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=self.num_classes,
            normalize='true'
        )
        
    def _init_important_metrics(self):
        """Initialize only the important-class-specific metrics"""
        # Using separate metric objects for cleaner reset
        self.val_important_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_important_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average='none')
        self.val_important_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average='none')

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
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=-1)
        
        # Update all metrics
        self.val_acc.update(preds, targets)  # Changed from direct call to update()
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_f1_macro.update(preds, targets)
        self.val_f1_weighted.update(preds, targets)
        self.val_confmat.update(preds, targets)
        
        self.validation_step_outputs.append({
            'preds': preds,
            'targets': targets,
            'loss': loss
        })
        
        # Only update important class metrics if they exist
        if len(self.important_classes) > 0:
            important_mask = torch.isin(targets, self.important_classes)
            if important_mask.any():
                # Update overall important accuracy
                self.val_important_acc.update(preds[important_mask], targets[important_mask])
                
                # Update per-class precision/recall
                self.val_important_precision.update(preds, targets)
                self.val_important_recall.update(preds, targets)
        
        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Compute confusion matrix and get class-wise metrics
        confmat = self.val_confmat.compute()
        
        # Get metrics for the 10 most frequent and 10 least frequent classes
        class_counts = torch.bincount(all_targets)
        top_classes = torch.topk(class_counts, 10).indices
        bottom_classes = torch.topk(class_counts, 10, largest=False).indices
        
        # Compute all metrics first
        metrics = {
            'val/loss': torch.stack([x['loss'] for x in self.validation_step_outputs]).mean(),
            'val/acc': self.val_acc.compute(),
            'val/precision_macro': self.val_precision.compute(),
            'val/recall_macro': self.val_recall.compute(),
            'val/f1_macro': self.val_f1_macro.compute(),
            'val/f1_weighted': self.val_f1_weighted.compute(),
        }
        
        # Convert class counts to float for logging
        class_counts_float = class_counts.float()
        
        # Prepare class-specific metrics
        class_metrics = {}
        for i in top_classes:
            if class_counts[i] > 0:
                precision = confmat[i, i] / confmat[:, i].sum()
                recall = confmat[i, i] / confmat[i, :].sum()
                class_metrics.update({
                    f'val_top_classes/class_{i}_precision': precision,
                    f'val_top_classes/class_{i}_recall': recall,
                    f'val_top_classes/class_{i}_support': class_counts_float[i]  # Convert to float
                })
                
        for i in bottom_classes:
            if class_counts[i] > 0:
                precision = confmat[i, i] / confmat[:, i].sum()
                recall = confmat[i, i] / confmat[i, :].sum()
                class_metrics.update({
                    f'val_rare_classes/class_{i}_precision': precision,
                    f'val_rare_classes/class_{i}_recall': recall,
                    f'val_rare_classes/class_{i}_support': class_counts_float[i]  # Convert to float
                })
        
        # Log important class metrics if they exist
        if len(self.important_classes) > 0:
            # Get overall important class accuracy
            important_acc = self.val_important_acc.compute()
            metrics['val/important_acc'] = important_acc
            
            # Get per-class metrics only for important classes
            precisions = self.val_important_precision.compute()
            recalls = self.val_important_recall.compute()
            
            for i, cls in enumerate(self.important_classes):
                cls = cls.item()
                metrics.update({
                    f'val/important_{cls}_precision': precisions[cls],
                    f'val/important_{cls}_recall': recalls[cls],
                    f'val/important_{cls}_fnr': 1 - recalls[cls]
                })
        
        # Log all metrics at once
        self.log_dict(metrics, prog_bar=True)
        self.log_dict(class_metrics)
        
        # Clear validation outputs and reset metrics
        self.validation_step_outputs.clear()
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset both regular and important metrics"""
        # Reset all metrics
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1_macro.reset()
        self.val_f1_weighted.reset()
        self.val_confmat.reset()
        
        # Reset important metrics
        if len(self.important_classes) > 0:
            self.val_important_acc.reset()
            self.val_important_precision.reset()
            self.val_important_recall.reset()
            
    def configure_optimizers(self):
        training_cfg = self.hparams.config['training']

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training_cfg.get('lr', 1e-4),
            weight_decay=training_cfg.get('weight_decay', 0.01),
            fused=True
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