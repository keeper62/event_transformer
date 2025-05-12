import torchmetrics
import importlib
from typing import Optional, Dict, Any, Tuple
import logging

from models import Transformer, LogTemplateMiner, LogTokenizer
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl


from torchmetrics import Metric
import torch.nn as nn
import torch.nn.functional as F

import evaluate

import os

import logging

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

def setup_logger(name: str | None = None) -> logging.Logger:
    """Setup logger that works with PyTorch Lightning."""
    logger = logging.getLogger(name or __name__)
    
    # Clear existing configuration
    logger.handlers.clear()
    logger.propagate = False  # Critical for PL compatibility
    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:  # Main process only    
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class BLEUScore(Metric):
    """GPU-compatible BLEU score metric with proper tensor handling"""
    def __init__(self, max_n=4, weights=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_n = max_n
        self.weights = weights or [1/max_n] * max_n
        
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("references", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Store tensors directly (GPU-compatible)"""
        # Ensure we're working with sequences (2D tensors)
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if target.ndim == 1:
            target = target.unsqueeze(-1)
            
        self.predictions.append(preds.detach().clone())
        self.references.append(target.detach().clone())

    def compute(self):
        """Calculate BLEU score with proper GPU handling"""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        
        # Move all tensors to CPU at once for efficiency
        preds_cpu = torch.cat(self.predictions).cpu().numpy()
        refs_cpu = torch.cat(self.references).cpu().numpy()
        
        total = 0.0
        for pred, ref in zip(preds_cpu, refs_cpu):
            # Convert arrays to lists of strings
            pred_str = [str(x) for x in pred]
            ref_str = [[str(x) for x in ref]]  # Note double list for references
            
            score = sentence_bleu(
                ref_str, 
                pred_str, 
                weights=self.weights,
                smoothing_function=smoother
            )
            total += score
        
        return torch.tensor(total / len(preds_cpu), device=self.device)

class ROUGELScore(Metric):
    """GPU-compatible ROUGE-L metric with batch processing"""
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.rouge = evaluate.load('rouge')
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("references", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Store batch tensors (GPU-compatible)"""
        # Ensure we're working with sequences (2D tensors)
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if target.ndim == 1:
            target = target.unsqueeze(-1)
            
        self.predictions.append(preds.detach().clone())
        self.references.append(target.detach().clone())

    def compute(self):
        """Batch process ROUGE-L on CPU"""
        if not self.predictions:
            return torch.tensor(0.0, device=self.device)
            
        # Concatenate all batches and move to CPU once
        preds_cpu = torch.cat(self.predictions).cpu().numpy()
        refs_cpu = torch.cat(self.references).cpu().numpy()
        
        # Convert to strings - handling both sequences and single tokens
        if preds_cpu.ndim == 1:
            pred_strs = [str(p) for p in preds_cpu]
            ref_strs = [str(r) for r in refs_cpu]
        else:
            pred_strs = [" ".join(map(str, p)) for p in preds_cpu]
            ref_strs = [" ".join(map(str, r)) for r in refs_cpu]
        
        # Batch compute ROUGE
        results = self.rouge.compute(
            predictions=pred_strs,
            references=ref_strs,
            rouge_types=['rougeL']
        )
        
        # Handle different output formats from evaluate library
        if isinstance(results['rougeL'], dict):
            # Old format: {'rougeL': {'mid': {'fmeasure': float}}}
            rouge_score = results['rougeL']['mid']['fmeasure']
        else:
            # New format: {'rougeL': float}
            rouge_score = results['rougeL']
        
        return torch.tensor(rouge_score, device=self.device)

class ImportantClassWrapper(torchmetrics.Metric):
    def __init__(self, metric, important_classes, prefix='val/important_'):
        super().__init__()
        self.metric = metric
        self.important_classes = important_classes.to('cpu')  # Store on CPU
        self.prefix = prefix

    def update(self, preds, target):
        # Move important_classes to target device
        important_classes = self.important_classes.to(target.device)
        
        # Create mask safely
        mask = torch.zeros_like(target, dtype=torch.bool)
        for cls in important_classes:
            mask = mask | (target == cls)
        
        if mask.any():
            # Ensure preds and target are on same device
            preds = preds.to(target.device)
            self.metric.update(preds[mask], target[mask])

    def compute(self):
        result = self.metric.compute()
        if isinstance(result, dict):
            return {f"{self.prefix}{k}": v for k, v in result.items()}
        return {f"{self.prefix}score": result}

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self._logger = setup_logger(self.__class__.__name__)

    def forward(self, inputs, targets):
        # Ensure tensors are on same device
        targets = targets.to(inputs.device)
        
        # Flatten if needed
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.size(-1))
            targets = targets.reshape(-1)

        # Debug shapes and device
        self._logger.debug(f"FocalLoss - inputs: {inputs.shape} ({inputs.device}), "
                         f"targets: {targets.shape} ({targets.device})")

        # Validate targets
        if (targets < 0).any():
            self._logger.warning(f"Negative targets detected: {targets[targets < 0]}")
            targets = targets.clamp(min=0)
            
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Safe gather with bounds checking
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            max_idx = alpha.size(0) - 1
            targets = targets.clamp(0, max_idx)
            
        # Get log probability of true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        
        # Compute focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            focal_term = alpha_t * focal_term
        
        # Final loss
        loss = -focal_term * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class DataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], test_mode: bool = False, logger=None):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self._setup_complete = False
        
        self._logger = setup_logger(self.__class__.__name__)
        self._logger.debug(f"Initializing DataModule with config: {config}")
        
        # Log device info early
        self._log_device_info()
        
        try:
            # Initialize components
            self._logger.debug("Initializing LogTemplateMiner...")
            self.template_miner = LogTemplateMiner(config['dataset']['drain_path'])
            
            self._logger.debug("Loading template miner state...")
            self.template_miner.load_state()
            
            self._logger.debug("Initializing LogTokenizer...")
            self.tokenizer = LogTokenizer(
                config['tokenizer']['tokenizer_length'], 
                tokenizer_path=config['tokenizer']['tokenizer_path']
            )
            
            # Dynamic dataset class loading
            self._logger.debug(f"Loading dataset class: {config['dataset']['class']}")
            dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
            self.dataset_class = getattr(dataset_module, "Dataset")
            
            self._logger.info("DataModule components initialized successfully")
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def _log_device_info(self):
        """Log comprehensive device and environment information"""
        self._logger.info("===== Environment Configuration =====")
        self._logger.info(f"PyTorch Lightning v{pl.__version__}")
        self._logger.info(f"PyTorch v{torch.__version__}")
        self._logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self._logger.info(f"Current Device: {torch.cuda.current_device()}")
            self._logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
        self._logger.info(f"Num CPUs: {os.cpu_count()}")
        self._logger.info("====================================")

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets for each stage with detailed logging"""
        if self._setup_complete:
            self._logger.debug("Setup already complete, skipping")
            return
            
        self._logger.info(f"Starting dataset setup for stage: {stage}")
        
        try:
            # Initialize full dataset
            self._logger.debug("Creating full dataset...")
            self.full_dataset = self.dataset_class(
                path=self.config['dataset']['path'],
                prediction_steps=self.config['model']['prediction_steps'],
                context_length=self.config['model']['context_length'],
                template_miner=self.template_miner.transform,
                tokenizer=self.tokenizer.transform,
                test_mode=self.test_mode
            )
            self._logger.info(f"Full dataset created with {len(self.full_dataset)} samples")
            
            # Validate dataset samples
            self._logger.debug("Validating dataset samples...")
            self._validate_dataset_shapes(self.full_dataset)
            
            # Split dataset
            self._logger.debug("Splitting dataset...")
            if self.test_mode:
                test_size = min(100, len(self.full_dataset) // 10)
                train_size = len(self.full_dataset) - test_size
                self._logger.info(f"Test mode: splitting into {train_size} train, {test_size} test samples")
                
                self.train_dataset, self.test_dataset = random_split(
                    self.full_dataset, 
                    [train_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
                self.val_dataset = self.test_dataset
            else:
                val_size = min(int(0.2 * len(self.full_dataset)), 4096)
                train_size = len(self.full_dataset) - val_size
                self._logger.info(f"Train mode: splitting into {train_size} train, {val_size} val samples")
                
                self.train_dataset, self.val_dataset = random_split(
                    self.full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )

            # Validate split datasets
            self._logger.debug("Validating split datasets...")
            if hasattr(self, 'train_dataset'):
                self._validate_dataset_shapes(self.train_dataset)
            if hasattr(self, 'val_dataset'):
                self._validate_dataset_shapes(self.val_dataset)
            if hasattr(self, 'test_dataset'):
                self._validate_dataset_shapes(self.test_dataset)
                
            self._setup_complete = True
            self._logger.info("Dataset setup completed successfully")
            
        except Exception as e:
            self._logger.error(f"Dataset setup failed: {str(e)}", exc_info=True)
            raise

    def _validate_dataset_shapes(self, dataset):
        """Enhanced dataset validation with detailed logging"""
        self._logger.debug(f"Validating dataset with {len(dataset)} samples")
        
        try:
            sample_count = min(5, len(dataset))  # Check first 5 samples
            for i in range(sample_count):
                input_window, output_window, sequences = dataset[i]
                
                self._logger.debug(f"Sample {i} shapes - "
                           f"input: {input_window.shape}, "
                           f"output: {output_window.shape}, "
                           f"sequences: {sequences.shape}")
                
                # Validate shapes against config
                assert len(input_window) == self.config['model']['context_length'], \
                    f"Input window length {len(input_window)} != context_length {self.config['model']['context_length']}"
                
                assert len(output_window) == self.config['model']['context_length'], \
                    f"Output window length {len(output_window)} != prediction_length {self.config['model']['context_length']}"
                
                # Validate value ranges
                if hasattr(self.tokenizer, 'vocab_size'):
                    assert output_window.max() < self.tokenizer.vocab_size, \
                        f"Output contains invalid token indices (max: {output_window.max()}, vocab size: {self.tokenizer.vocab_size})"
                
            self._logger.debug(f"Dataset validation passed for {sample_count} samples")
            
        except Exception as e:
            self._logger.error(f"Dataset validation failed on sample {i}: {str(e)}")
            self._logger.error(f"Problematic sample - input: {input_window}, output: {output_window}")
            raise

    def get_class_weights(self, strategy: str = "inverse", max_samples: int = 1000) -> torch.Tensor:
        self._logger.info("Computing class weights...")
        
        if not self._setup_complete:
            self.setup()

        vocab_size = self.config['model']['vocab_size']
        counts = torch.zeros(vocab_size, dtype=torch.long)
        
        # Sample processing with enhanced debugging
        n_samples = min(len(self.train_dataset), max_samples)
        indices = torch.randperm(len(self.train_dataset))[:n_samples]
        
        self._logger.debug(f"Processing {n_samples} samples")
        self._logger.debug(f"Vocab size: {vocab_size}")
        
        try:
            # Process samples with validation
            for i, idx in enumerate(indices):
                _, targets, _ = self.train_dataset[idx]
                
                # Validate targets
                if (targets < 0).any() or (targets >= vocab_size).any():
                    invalid = targets[(targets < 0) | (targets >= vocab_size)]
                    self._logger.error(f"Invalid targets in sample {i}: {invalid.cpu().numpy()}")
                    self._logger.error(f"Target range: {targets.min().item()} to {targets.max().item()}")
                    self._logger.error(f"Vocab size: {vocab_size}")
                    raise ValueError("Invalid target indices detected")
                    
                unique, counts_batch = torch.unique(targets.flatten(), return_counts=True)
                counts[unique] += counts_batch
                
                if i % 100 == 0:  # Progress logging
                    self._logger.debug(f"Processed {i}/{n_samples} samples")
                    
        except Exception as e:
            self._logger.error(f"Failed processing sample {i}: {str(e)}")
            self._logger.error(f"Sample shapes - targets: {targets.shape}")
            self._logger.error(f"Sample values - targets: {targets.cpu().numpy()}")
            raise
        
        # Log class distribution
        self._logger.debug(f"Class distribution - min: {counts.min()}, max: {counts.max()}, mean: {counts.float().mean()}")
        
        counts = counts.float() + 1e-6  # Smoothing
        
        # Weight calculation
        if strategy == "inverse":
            weights = 1.0 / counts
        elif strategy == "sqrt":
            weights = 1.0 / torch.sqrt(counts)
        elif strategy == "balanced":
            weights = (n_samples / vocab_size) / counts
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        weights = weights * (vocab_size / weights.sum())
        
        # Log statistics
        self._logger.info(
            f"Class weights: min={weights.min():.2f}, "
            f"median={weights.median():.2f}, "
            f"max={weights.max():.2f}"
        )
    
        return weights

    def train_dataloader(self) -> DataLoader:
        self._logger.debug("Creating train DataLoader")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        self._logger.debug("Creating val DataLoader")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if not hasattr(self, 'test_dataset'):
            self._logger.debug("No test dataset available")
            return None
        self._logger.debug("Creating test DataLoader")
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """Centralized DataLoader creation with validation"""
        if dataset is None:
            self._logger.warning("Attempted to create DataLoader with None dataset")
            return None
            
        self._logger.debug(f"Creating DataLoader with {len(dataset)} samples (shuffle={shuffle})")
        
        try:
            loader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=shuffle,
                num_workers=self.config['dataset'].get('num_workers', 0),
                pin_memory=torch.cuda.is_available(),
                persistent_workers=self.config['dataset'].get('num_workers', 0) > 0,
                drop_last=shuffle  # Only drop last for training
            )
            self._logger.debug("DataLoader created successfully")
            return loader
        except Exception as e:
            self._logger.error(f"DataLoader creation failed: {str(e)}")
            raise

class TransformerLightning(pl.LightningModule):
    def __init__(self, config: Dict[str, Any], class_weights: torch.Tensor, config_name: Optional[str] = None, 
                important_classes: torch.Tensor | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
        # Setup logger
        self._logger = setup_logger(self.__class__.__name__)
        self._logger.info("Initializing TransformerLightning model")
        
        # Model components
        self._logger.debug(f"Creating Transformer with config: {config['model']}")
        self.model = Transformer(config)
        self.num_classes = config['model']['vocab_size']
        self._logger.info(f"Model initialized with num_classes: {self.num_classes}")
        
        # Important classes setup
        if important_classes is not None:
            self.important_classes = important_classes.long()
            self._logger.debug(f"Important classes received: {self.important_classes.shape} "
                            f"(min: {self.important_classes.min()}, max: {self.important_classes.max()})")
        else:
            self.important_classes = torch.tensor([], dtype=torch.long)
            self._logger.debug("No important classes specified")
        
        self.importance_boost_factor = config['training'].get('importance_boost_factor', 15.0)
        self._logger.debug(f"Using importance boost factor: {self.importance_boost_factor}")
        
        # Initialize metrics
        self._logger.debug("Initializing metrics...")
        self._init_metrics()
        
        # Class weights validation
        self.class_weights = class_weights
        self._logger.debug(f"Class weights shape: {self.class_weights.shape} "
                        f"(device: {self.class_weights.device})")
        
        # Verify class weights match num_classes
        if len(self.class_weights) != self.num_classes:
            error_msg = (f"class_weights length ({len(self.class_weights)}) "
                        f"doesn't match num_classes ({self.num_classes})")
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Verify important_classes are valid
        if len(self.important_classes) > 0:
            invalid = self.important_classes[(self.important_classes < 0) | 
                                            (self.important_classes >= self.num_classes)]
            if len(invalid) > 0:
                error_msg = f"Invalid important_classes indices: {invalid.cpu().numpy()}"
                self._logger.error(error_msg)
                raise ValueError(error_msg)
            self._logger.debug(f"Validated important classes (count: {len(self.important_classes)})")
        
        # Loss function initialization
        self._logger.debug("Initializing FocalLoss with: "
                        f"gamma={config['training'].get('focal_gamma', 2.0)}, "
                        f"reduction='mean'")
        self.loss_fn = FocalLoss(
            alpha=self.class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            reduction='mean'
        )
        self._logger.info("FocalLoss initialized successfully")
        
        # Final initialization check
        self._logger.info("TransformerLightning initialization complete")
        if torch.cuda.is_available():
            self._logger.info(f"Model will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._logger.info("Model will use CPU")
        
    def forward(self, inputs: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, sequences) 

    def _init_metrics(self):
        """Initialize all TorchMetrics for evaluation"""
        metric_args = {
            'task': 'binary'
        }
        self.train_acc = torchmetrics.Accuracy(**metric_args)
        
        # Validation metrics
        self.val_acc = torchmetrics.Accuracy(**metric_args)
        self.val_top5 = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
            top_k=5
        )
        
        # New metrics
        self.val_bleu = BLEUScore(max_n=4)  # Average of BLEU-1 to BLEU-4
        self.val_rouge = ROUGELScore()
        
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
        
        # Update metrics
        self.val_acc.update((preds == targets).float(), torch.ones_like(targets))
        self.val_top5.update(logits, targets)
        
        # Update sequence metrics (BLEU & ROUGE)
        self.val_bleu.update(preds, targets)  # Compare predicted vs target sequences
        self.val_rouge.update(preds, targets)
        
        if hasattr(self, 'val_important'):
            self.val_important(preds, targets)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        # Log all metrics
        metrics = {
            'val/acc': self.val_acc.compute(),
            'val/top5_acc': self.val_top5.compute(),
            'val/bleu': self.val_bleu.compute(),  # Average BLEU-1 to BLEU-4
            'val/rougeL': self.val_rouge.compute(),  # ROUGE-L F1 score
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
        
        self.log_dict(metrics, prog_bar=False, sync_dist=True)
        
        # Reset all metrics
        self.val_acc.reset()
        self.val_top5.reset()
        self.val_bleu.reset()
        self.val_rouge.reset()
        
        if hasattr(self, 'val_important'):
            self.val_important.reset()

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Compute training accuracy (binary)
        preds = logits.argmax(dim=-1)
        is_correct = (preds == targets).float()
        self.train_acc.update(is_correct, torch.ones_like(is_correct))
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        
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