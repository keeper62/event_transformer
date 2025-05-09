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
                 important_classes: torch.Tensor | None = None, top_k: int = 20, bottom_k: int = 20):
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
        
        # Module-specific logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(logging.DEBUG)
        
        # File handler (only added by rank 0 to avoid duplicate logs)
        file_handler = logging.FileHandler("training.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

            
        self._logger.info("Initializing transformer model")
        
        # Class tracking setup
        self.important_classes = important_classes.long().to(self.device) if important_classes is not None else torch.tensor([], dtype=torch.long, device=self.device)
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.importance_boost_factor = config['training'].get('importance_boost_factor', 15.0)
        
        self.class_weights = self._adjust_class_weights(class_weights)
        
        # Initialize metrics
        self._init_metrics()
        self._init_tracking_structures()
        
        # Loss function
        self.loss_fn = FocalLoss(
            alpha=self.class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            reduction='mean'
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.validation_step_outputs = []
        
    def forward(self, inputs: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        self._logger.debug(f"Model input shapes - inputs: {inputs.shape}, sequences: {sequences.shape}")
        self._logger.debug(f"Model input devices - inputs: {inputs.device}, sequences: {sequences.device}")
        output = self.model(inputs, sequences)
        self._logger.debug(f"Model output shape: {output.shape}")
        return output
    
    def _init_tracking_structures(self):
        """Initialize structures for tracking class performance"""
        # For important classes
        self.important_class_samples = []
        
        # For top/bottom k classes
        self.class_counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.top_bottom_samples = []
        self._sample_size = 1000  # Samples to keep for analysis

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
        """Initialize memory-efficient metrics"""
        common_args = {
            'task': 'multiclass',
            'num_classes': self.num_classes,
            'sync_on_compute': False
        }
        
        # Core metrics
        self.val_acc = torchmetrics.Accuracy(**common_args, average='micro')
        
        # Lightweight F1 approximation
        self._f1_samples = []  # Store (correct, pred, target) tuples
        self._f1_sample_size = 1000  # Number of samples to keep
        
        # Important class tracking
        if len(self.important_classes) > 0:
            self.val_important_acc = torchmetrics.Accuracy(**common_args)
            
    def _update_f1_approximation(self, preds, targets):
        """Store samples for F1 approximation"""
        if len(self._f1_samples) < self._f1_sample_size:
            # Store (is_correct, predicted_class, true_class)
            is_correct = (preds == targets).float()
            samples = torch.stack([is_correct, preds.float(), targets.float()], dim=1)
            self._f1_samples.append(samples)
            
    def _compute_approximate_f1(self):
        """Compute approximate F1 score from samples"""
        if not self._f1_samples:
            return torch.tensor(0.0, device=self.device)
        
        samples = torch.cat(self._f1_samples)
        is_correct = samples[:, 0]
        preds = samples[:, 1].long()
        targets = samples[:, 2].long()
        
        # Compute precision and recall approximation
        true_pos = is_correct.sum()
        pred_pos = len(preds)
        actual_pos = len(targets)
        
        precision = true_pos / pred_pos
        recall = true_pos / actual_pos
        
        return 2 * (precision * recall) / (precision + recall + 1e-8)
            
    def _track_class_performance(self, preds: torch.Tensor, targets: torch.Tensor):
        """Track samples for important and top/bottom classes with proper device handling"""
        # Ensure tensors are on the same device
        device = targets.device
        
        # Update class counts (ensure counts tensor is on correct device)
        unique, counts = torch.unique(targets, return_counts=True)
        self.class_counts = self.class_counts.to(device)
        self.class_counts[unique] += counts.to(device)
        
        # Sample data for analysis
        if len(preds) > self._sample_size:
            indices = torch.randperm(len(preds), device=device)[:self._sample_size]
            preds = preds[indices]
            targets = targets[indices]
        
        # Track important classes (with device-safe isin implementation)
        if len(self.important_classes) > 0:
            # Move important_classes to correct device
            important_classes = self.important_classes.to(device)
            
            # Device-safe mask creation
            if torch.__version__ >= '2.2':
                important_mask = torch.isin(targets, important_classes)
            else:
                # Fallback for PyTorch < 2.2
                important_mask = torch.isin(targets.cpu(), important_classes.cpu()).to(device)
            
            if important_mask.any():
                self.important_class_samples.append((
                    preds[important_mask],
                    targets[important_mask]
                ))
        
        # Track top/bottom k classes (with proper device handling)
        current_counts = self.class_counts.cpu().numpy()  # numpy works on CPU
        top_classes = np.argpartition(-current_counts, self.top_k)[:self.top_k]
        bottom_classes = np.argpartition(current_counts, self.bottom_k)[:self.bottom_k]
        
        # Create track_classes tensor on the correct device
        track_classes = torch.tensor(
            np.concatenate([top_classes, bottom_classes]),
            device=device
        )
        
        # Device-safe mask creation for tracking
        if torch.__version__ >= '2.2':
            track_mask = torch.isin(targets, track_classes)
        else:
            track_mask = torch.isin(targets.cpu(), track_classes.cpu()).to(device)
        
        if track_mask.any():
            self.top_bottom_samples.append((
                preds[track_mask],
                targets[track_mask]
            ))

    def _compute_tracked_confusion(self, samples, 
                                 classes: torch.Tensor) -> torch.Tensor:
        """Compute confusion matrix for tracked classes"""
        if not samples:
            return None
            
        all_preds, all_targets = zip(*samples)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        
        # Filter to only the specified classes
        mask = torch.isin(targets, classes)
        if not mask.any():
            return None
            
        preds = preds[mask]
        targets = targets[mask]
        
        # Create mapping to contiguous indices
        unique_classes = classes.unique()
        class_map = {cls.item(): i for i, cls in enumerate(unique_classes)}
        
        mapped_targets = torch.tensor([class_map[t.item()] for t in targets], 
                                    device=targets.device)
        mapped_preds = torch.tensor([class_map[p.item()] for p in preds], 
                                   device=preds.device)
        
        return torchmetrics.functional.confusion_matrix(
            mapped_preds, mapped_targets,
            task='multiclass',
            num_classes=len(unique_classes),
            normalize='true'
        )
        
    def _init_important_metrics(self):
        """Initialize only the important-class-specific metrics"""
        # Using separate metric objects for cleaner reset
        self.val_important_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_important_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average='none')
        self.val_important_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average='none')

    def _validate_device_consistency(self, *tensors: torch.Tensor) -> None:
        """Debug helper to check tensor devices"""
        devices = {t.device for t in tensors if isinstance(t, torch.Tensor)}
        self._logger.debug(f"Devices: {devices}")
        if len(devices) > 1:
            error_msg = f"Device mismatch detected. Found devices: {devices}\n"
            error_msg += "Tensor details:\n"
            for i, t in enumerate(tensors):
                if isinstance(t, torch.Tensor):
                    error_msg += f"  Tensor {i}: device={t.device}, shape={t.shape}, dtype={t.dtype}\n"
            self._logger.debug(error_msg)
            raise RuntimeError(error_msg)

    def _process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets, sequences = batch
        
        # Debug device consistency
        self._logger.debug(f"Input shapes - inputs: {inputs.shape}, targets: {targets.shape}, sequences: {sequences.shape}")
        
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
        final_outputs = outputs.reshape(-1, self.num_classes)
        final_targets = targets.reshape(-1)
        
        self._logger.debug(f"Final shapes - outputs: {final_outputs.shape}, targets: {final_targets.shape}")
        
        final_targets = targets.reshape(-1)
        
        # Final device check before return
        self._validate_device_consistency(outputs, final_targets)
        return outputs, final_targets

    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        self._logger.debug("Now calculating loss")
        loss = self.loss_fn(logits, targets)
        self._logger.debug("Loss calculated")
        preds = logits.argmax(dim=-1)
        
        # Flatten if needed
        if logits.dim() == 3:
            preds = preds.view(-1)
            targets = targets.view(-1)
        
        # Update core metrics
        self.val_acc.update(preds, targets)
        self._update_f1_approximation(preds, targets)
        
        self._logger.debug("Normal metrics calculated")
        
        self.validation_step_outputs.append(loss)
        
        self._logger.debug("Test 2")
        
        # Update important class metrics
        if len(self.important_classes) > 0:
            # 1. Ensure both tensors are on CPU for isin()
            targets_cpu = targets.cpu()
            important_classes_cpu = self.important_classes.cpu()
            
            # 2. Create mask on CPU
            important_mask_cpu = torch.isin(targets_cpu, important_classes_cpu)
            
            # 3. For metric update: keep everything on CPU
            preds_cpu = preds.cpu()
            targets_cpu = targets.cpu()
            
            # 4. Update metric on CPU (temporarily)
            if important_mask_cpu.any():
                self.val_important_acc.to('cpu')
                self.val_important_acc.update(preds_cpu[important_mask_cpu], targets_cpu[important_mask_cpu])
                self.val_important_acc.to(targets.device)  # Move metric back to original device
                
        self._track_class_performance(preds, targets)    
            
        self._logger.debug("Important metrics calculated")
        
        return loss

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        self._logger.debug("Now calculating loss")
        loss = self.loss_fn(logits, targets)
        self._logger.debug("Loss calculated")
        
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
    
    def on_validation_epoch_end(self):
        metrics = {
            'val/loss': torch.stack([x for x in self.validation_step_outputs]).mean(),
            'val/acc': self.val_acc.compute(),
            'val/approx_f1': self._compute_approximate_f1()
        }
        
        # Compute important class metrics
        if len(self.important_classes) > 0:
            metrics['val/important_acc'] = self.val_important_acc.compute()
            
            # Compute confusion matrix for important classes
            important_confmat = self._compute_tracked_confusion(
                self.important_class_samples,
                self.important_classes
            )
            if important_confmat is not None:
                self.important_class_confmat = important_confmat
                # Compute precision/recall/F1 for important classes
                for i, cls in enumerate(self.important_classes):
                    cls_idx = i
                    tp = important_confmat[cls_idx, cls_idx]
                    fp = important_confmat[:, cls_idx].sum() - tp
                    fn = important_confmat[cls_idx, :].sum() - tp
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                    
                    metrics.update({
                        f'val/important_{cls.item()}_precision': precision,
                        f'val/important_{cls.item()}_recall': recall,
                        f'val/important_{cls.item()}_f1': f1
                    })
        
        # Compute top/bottom class metrics
        current_counts = self.class_counts.cpu().numpy()
        top_classes = torch.tensor(np.argpartition(-current_counts, self.top_k)[:self.top_k], 
                    device=self.device)
        bottom_classes = torch.tensor(np.argpartition(current_counts, self.bottom_k)[:self.bottom_k],
                        device=self.device)
        
        tracked_classes = torch.cat([top_classes, bottom_classes])
        top_bottom_confmat = self._compute_tracked_confusion(
            self.top_bottom_samples,
            tracked_classes
        )
        
        if top_bottom_confmat is not None:
            self.top_bottom_confmat = top_bottom_confmat
            num_top = len(top_classes)
            
            # Track metrics for top classes
            for i, cls in enumerate(top_classes):
                cls_idx = i
                tp = top_bottom_confmat[cls_idx, cls_idx]
                fp = top_bottom_confmat[:, cls_idx].sum() - tp
                fn = top_bottom_confmat[cls_idx, :].sum() - tp
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                metrics.update({
                    f'val/top_{i}_acc': top_bottom_confmat[cls_idx, cls_idx] / top_bottom_confmat[cls_idx, :].sum(),
                    f'val/top_{i}_precision': precision,
                    f'val/top_{i}_recall': recall,
                    f'val/top_{i}_f1': f1
                })
            
            # Track metrics for bottom classes
            for j, cls in enumerate(bottom_classes):
                cls_idx = num_top + j
                tp = top_bottom_confmat[cls_idx, cls_idx]
                fp = top_bottom_confmat[:, cls_idx].sum() - tp
                fn = top_bottom_confmat[cls_idx, :].sum() - tp
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                metrics.update({
                    f'val/bottom_{j}_acc': top_bottom_confmat[cls_idx, cls_idx] / top_bottom_confmat[cls_idx, :].sum(),
                    f'val/bottom_{j}_precision': precision,
                    f'val/bottom_{j}_recall': recall,
                    f'val/bottom_{j}_f1': f1
                })
        
        self.log_dict(metrics, prog_bar=True)
        self._reset_metrics()
        self._reset_tracking()

    def _reset_tracking(self):
        """Reset tracking structures for next epoch"""
        self.important_class_samples.clear()
        self.top_bottom_samples.clear()
        self.class_counts.zero_()

    def _reset_metrics(self):
        """Reset metrics"""
        self.val_acc.reset()
        self._f1_samples.clear()
        if len(self.important_classes) > 0:
            self.val_important_acc.reset()
            
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