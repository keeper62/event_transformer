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
#from focal_loss.focal_loss import FocalLoss

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

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

class FocalLoss(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target!=self.ignore_index) 
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert not torch.isnan(x).any(), "NaN in predictions"
        assert not torch.isinf(x).any(), "Inf in predictions"
        x = torch.clamp(x, min=1e-7, max=1.0 - 1e-7)
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

class LightningNGramScore(torchmetrics.Metric):
    def __init__(self, ngram_size=4):
        super().__init__()
        self.ngram_size = ngram_size
        
        self.add_state("match_counts", default=torch.zeros(ngram_size), dist_reduce_fx="sum")
        self.add_state("total_counts", default=torch.zeros(ngram_size), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: [batch_size, seq_len, vocab_size] logits
        targets: [batch_size, seq_len] ground truth indices
        """
        # Convert logits to predictions
        preds = preds.argmax(dim=-1)  # [batch_size, seq_len]
        
        # Ensure 2D input
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
        for n in range(1, self.ngram_size + 1):
            pred_ngrams = preds.unfold(1, n, 1)   # [batch, ngram_windows, n]
            target_ngrams = targets.unfold(1, n, 1)
            
            matches = (pred_ngrams == target_ngrams).all(dim=-1)
            self.match_counts[n-1] += matches.sum().float()
            self.total_counts[n-1] += matches.numel()
    
    def compute(self):
        precisions = self.match_counts / self.total_counts.clamp(min=1)
        return torch.exp(torch.mean(torch.log(precisions)))

class DataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], test_mode: bool = False, logger=None):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self._setup_complete = False
        self.split_ratios = config['dataset'].get('split_ratios', [0.8, 0.1, 0.1])
        
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
            full_dataset = self.dataset_class(
                path=self.config['dataset']['path'],
                context_length=self.config['model']['context_length'],
                template_miner=self.template_miner.transform,
                tokenizer=self.tokenizer.transform,
                test_mode=self.test_mode
            )
            self._logger.info(f"Full dataset created with {len(full_dataset)} samples")
            
            # Validate dataset samples
            self._logger.debug("Validating dataset samples...")
            self._validate_dataset_shapes(full_dataset)
            
            # Split dataset
            self._logger.debug("Splitting dataset...")
            train_size = 0.8 
            val_size = 0.2

            # Split the dataset
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )

            # Validate split datasets
            self._logger.debug("Validating split datasets...")
            if hasattr(self, 'train_dataset'):
                self._validate_dataset_shapes(self.train_dataset)
            if hasattr(self, 'val_dataset'):
                self._validate_dataset_shapes(self.val_dataset)
                
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

    def train_dataloader(self) -> DataLoader:
        self._logger.debug("Creating train DataLoader")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        self._logger.debug("Creating val DataLoader")
        return self._create_dataloader(self.val_dataset, shuffle=False)

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
    def __init__(self, config: Dict[str, Any], config_name: Optional[str] = None, 
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
        if config['model'].get('loss_fn', 'focal') == 'focal':
            self._logger.debug("Initializing FocalLoss with: "
                            f"gamma={config['training'].get('focal_gamma', 2.0)}, "
                            f"reduction='mean'")
            self.loss_fn = FocalLoss(
                #alpha=self.class_weights,
                gamma=config['training'].get('focal_gamma', 2.0),
                reduction='mean'
            )
        elif config['model'].get('loss_fn', 'focal') == "crossentropy":
            self._logger.debug("Initializing Cross-entropyloss")
            self.loss_fn = FocalLoss(
                #alpha=self.class_weights,
                gamma=0,
                reduction='mean'
            )
        
        self._logger.info("Loss function initialized successfully")
        
        # Final initialization check
        self._logger.info("TransformerLightning initialization complete")
        if torch.cuda.is_available():
            self._logger.info(f"Model will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._logger.info("Model will use CPU")

        
    def forward(self, inputs: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, sequences) 

    def _init_metrics(self):
        """Initialize metrics using MetricCollection for efficient logging"""
        # Common metrics for all phases
        base_metrics = {
            "acc": torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes),
            "top5": torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5),
        }
        
        # Training metrics
        self.train_metrics = torchmetrics.MetricCollection(
            {**base_metrics},
            prefix="train/"
        )
        
        # Validation metrics
        self.val_metrics = torchmetrics.MetricCollection(
            {
                **base_metrics,
                "bleu": LightningNGramScore(ngram_size=4),
            },
            prefix="val/"
        )

    def _process_batch(self, batch):
        inputs, targets, sequences = batch
        logits = self.model(inputs, sequences)
        return logits.view(-1, logits.size(-1)), targets.view(-1)                   

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(torch.nn.functional.softmax(logits, dim=-1), targets)
        
        # Update and log metrics
        metrics = self.train_metrics(logits, targets)
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(torch.nn.functional.softmax(logits, dim=-1), targets)
        
        # Update metrics
        self.val_metrics.update(logits, targets)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        # Compute all validation metrics
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, sync_dist=True)
        self.val_metrics.reset()
            
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
            min_lr=training_cfg.get('min_lr', 0.1)
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