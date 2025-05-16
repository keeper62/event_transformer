import torchmetrics
import importlib
from typing import Optional, Dict, Any, Tuple
import logging

from models import Transformer, LogTemplateMiner, LogTokenizer
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F

import os

import logging

import numpy as np

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

class LightningNGramScore(torchmetrics.Metric):
    def __init__(self, ngram_size=4):
        super().__init__()
        self.ngram_size = ngram_size
        
        # Lightning-managed states
        self.add_state("match_counts", default=torch.zeros(ngram_size), dist_reduce_fx="sum")
        self.add_state("total_counts", default=torch.zeros(ngram_size), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Ensure 2D input
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
        for n in range(1, self.ngram_size + 1):
            # Extract n-grams using unfold
            pred_ngrams = preds.unfold(1, n, 1)   # [batch, ngram_windows, n]
            target_ngrams = targets.unfold(1, n, 1)
            
            # Compare n-grams
            matches = (pred_ngrams == target_ngrams).all(dim=-1)
            self.match_counts[n-1] += matches.sum().float()
            self.total_counts[n-1] += matches.numel()
    
    def compute(self):
        precisions = self.match_counts / self.total_counts.clamp(min=1)
        return torch.exp(torch.mean(torch.log(precisions)))

class RougeL(torchmetrics.Metric):
    def __init__(self, exact_mode=True, fast_threshold=0.9):
        super().__init__()
        self.exact_mode = exact_mode  # False for approximate-only
        self.fast_threshold = fast_threshold
        
        self.add_state("lcs_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pred_len_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_len_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Process 1D sequences with ultra-optimized LCS"""
        # Input validation
        if preds.dim() != 1 or targets.dim() != 1:
            raise ValueError("Inputs must be 1D tensors")
        
        lcs_len = self._lightning_lcs(preds, targets)
        pred_len = len(preds)
        target_len = len(targets)
        
        self.lcs_sum += lcs_len
        self.pred_len_sum += pred_len
        self.target_len_sum += target_len
        self.total += 1

    def _lightning_lcs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Optimized LCS pipeline with all speed tricks"""
        # Early exit for empty sequences
        if len(a) == 0 or len(b) == 0:
            return torch.tensor(0.0, device=a.device)
        
        # Instant match check
        if torch.equal(a, b):
            return torch.tensor(len(a), device=a.device)
        
        # Prefix/suffix shortcut
        min_len = min(len(a), len(b))
        if torch.equal(a[:min_len], b[:min_len]) or torch.equal(a[-min_len:], b[-min_len:]):
            return torch.tensor(min_len, device=a.device)
        
        # Fast path: Content similarity check
        if self._content_similarity(a, b) >= self.fast_threshold:
            return torch.tensor(min_len, device=a.device)
        
        # Fallback to exact LCS if needed
        return self._exact_lcs(a, b) if self.exact_mode else self._approx_lcs(a, b)

    def _content_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Quick similarity estimate using unique elements"""
        unique_match = torch.isin(a.unique(), b.unique()).float().mean()
        len_ratio = min(len(a), len(b)) / max(len(a), len(b))
        return (unique_match + len_ratio) / 2

    def _approx_lcs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Conservative approximation using element matching"""
        # Count matches without order consideration
        a_vals, a_counts = torch.unique(a, return_counts=True)
        b_vals, b_counts = torch.unique(b, return_counts=True)
        
        common = torch.isin(a_vals, b_vals)
        matched = torch.minimum(a_counts[common], b_counts[torch.isin(b_vals, a_vals[common])])
        return matched.sum().float()

    def _exact_lcs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Memory-optimized DP implementation"""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return torch.tensor(0.0, device=a.device)
        
        # Use two rolling rows for O(n) space
        dp = torch.zeros(2, n+1, dtype=torch.long, device=a.device)
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    dp[1, j] = dp[0, j-1] + 1
                else:
                    dp[1, j] = max(dp[0, j], dp[1, j-1])
            dp[0] = dp[1].clone()
        return dp[1, n].float()

    def compute(self) -> torch.Tensor:
        """Compute final ROUGE-L F1 with numerical stability"""
        precision = self.lcs_sum / self.pred_len_sum.clamp(min=1e-6)
        recall = self.lcs_sum / self.target_len_sum.clamp(min=1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return torch.clamp(f1, 0, 1)

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

    def get_class_weights(self, strategy: str = "inverse", max_samples: int = 100000) -> torch.Tensor:
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
        self.seq_len = config['model'].get('seq_len', 512)  # Assuming fixed sequence length
        self._logger.info(f"Model initialized with num_classes: {self.num_classes}, seq_len: {self.seq_len}")
        
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
            #alpha=self.class_weights,
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
        # Training metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.train_top5 = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
            top_k=5
        )
        
        # Validation metrics
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_top5 = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
            top_k=5
        )
        
        # Sequence metrics - Updated to use torchmetrics implementations
        self.val_bleu = LightningNGramScore(ngram_size=4) # BLEU-4
        #self.val_rouge = RougeL(exact_mode=False)

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
        self._logger.debug("Validation step Started!")
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        preds = logits.argmax(dim=-1)

        # Update accuracy metrics
        self._logger.debug("Updating accuracy metrics")
        self.val_acc.update(preds, targets)
        self.val_top5.update(logits, targets)

        self._logger.debug("Updating sequence metrics")
        # Update CPU-based metrics
        self.val_bleu.update(preds, targets)
        #self.val_rouge.update(preds, targets)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self._logger.debug("Validation step finished!")
        return loss


    def on_validation_epoch_end(self):
        """Complete rewrite with guaranteed device safety"""
        metrics = {
            'val/acc': self.val_acc.compute(),
            'val/top5': self.val_top5.compute()
        }


        metrics.update({
            'val/bleu': self.val_bleu.compute(),
        #    'val/rougeL': self.val_rouge.compute()
        })

        # 3. Safe logging (no sync for text metrics)
        self._logger.debug("Logging accuracy")
        self.log_dict({
            k: v for k, v in metrics.items() 
            if k in ['val/acc', 'val/top5']
        }, sync_dist=True)  # Sync GPU metrics
        
        #self._logger.debug("Logging text metrics")
        self.log_dict({
            k: v for k, v in metrics.items() 
        #    if k in ['val/bleu', 'val/rougeL']
        }, sync_dist=True)  # Never sync CPU metrics
     
    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Compute training accuracy
        preds = logits.argmax(dim=-1)
        
        # Update training metrics
        self.train_acc.update(preds, targets)
        self.train_top5.update(logits, targets)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/top5_acc", self.train_top5, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log training metrics at epoch end"""
        try:
            train_acc = self.train_acc.compute()
            train_top5 = self.train_top5.compute()
            
            self.log("train/epoch_acc", train_acc, prog_bar=False, sync_dist=True)
            self.log("train/epoch_top5", train_top5, prog_bar=False, sync_dist=True)
            
            self._logger.info(f"Training Accuracy: {train_acc:.4f}, Top-5 Accuracy: {train_top5:.4f}")
        except Exception as e:
            self._logger.warning(f"Failed to compute training metrics: {str(e)}")
        finally:
            self.train_acc.reset()
            self.train_top5.reset()
    
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