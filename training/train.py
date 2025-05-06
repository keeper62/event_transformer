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
        
        self._init_metrics()
        
        # Proper device handling for class weights
        if class_weights is not None:
            # Register as buffer to automatically handle device placement
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Focal Loss configuration
        loss_kwargs = {
            'alpha': self.class_weights,  # Now properly device-aware
            'gamma': config['training'].get('focal_gamma', 2.0),
            'reduction': 'mean',
            'label_smoothing': config['training'].get('label_smoothing', 0.1)
        }
        self.loss_fn = self._get_focal_loss(**loss_kwargs)
        
        self.best_val_loss = float('inf')
        self.validation_step_outputs = []

    def _get_focal_loss(self, **kwargs):
        """Device-aware focal loss that ensures all tensors stay on same device."""
        class FocalLoss(torch.nn.Module):
            def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100, label_smoothing=0.0):
                super().__init__()
                # Register alpha as buffer if provided to ensure proper device movement
                if alpha is not None:
                    self.register_buffer('alpha', alpha)
                else:
                    self.alpha = None
                self.gamma = gamma
                self.reduction = reduction
                self.ignore_index = ignore_index
                self.label_smoothing = label_smoothing

            def forward(self, inputs, targets):
                # Ensure inputs and targets are on same device
                if inputs.device != targets.device:
                    targets = targets.to(inputs.device)

                if self.num_classes == 1:
                    # Binary case
                    return F.binary_cross_entropy_with_logits(
                        inputs.squeeze(-1),
                        targets.float(),
                        weight=self.alpha,
                        reduction=self.reduction
                    )
                else:
                    # Multi-class case
                    ce_loss = F.cross_entropy(
                        inputs, 
                        targets,
                        weight=self.alpha,
                        reduction='none',
                        ignore_index=self.ignore_index,
                        label_smoothing=self.label_smoothing
                    )
                    pt = torch.exp(-ce_loss)
                    focal_loss = (1 - pt) ** self.gamma * ce_loss

                    if self.reduction == 'mean':
                        return focal_loss.mean()
                    elif self.reduction == 'sum':
                        return focal_loss.sum()
                    return focal_loss

        # Ensure class weights are on correct device if provided
        if 'alpha' in kwargs and kwargs['alpha'] is not None:
            kwargs['alpha'] = kwargs['alpha'].to(self.device)

        return FocalLoss(**kwargs)

    def forward(self, x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        return self.model(x, sequences)

    def _init_metrics(self):
        """Initialize all metrics without computing them."""
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

    def _process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Get inputs and ensure device consistency
        inputs, targets, sequences = batch
        device = inputs.device

        # Verify all inputs are on same device
        assert targets.device == device, f"Targets on {targets.device} but inputs on {device}"
        assert sequences.device == device, f"Sequences on {sequences.device} but inputs on {device}"

        outputs = []

        # 2. Get config parameters
        config = self.hparams.config['training']
        sampling_mode = config.get('sampling_mode', 'top_k')
        top_k = config.get('top_k', 5)
        top_p = config.get('top_p', 0.9)
        temperature = torch.tensor(config.get('temperature', 1.0), device=device)
        autocast_enabled = config.get('mixed_precision', True) and torch.cuda.is_available()

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            # 3. Initialize state - explicitly clone to same device
            current_inputs = inputs.clone().to(device)
            current_sequences = sequences.clone().to(device)

            for step in range(self.model.n_steps):
                # 4. Forward pass - model should handle device
                logits = self(current_inputs, current_sequences)
                logits_last = logits[:, -1, :]
                outputs.append(logits_last)

                with torch.no_grad():
                    # 5. Apply temperature
                    logits_last = logits_last / temperature

                    # 6. Sampling - all operations stay on device
                    probs = torch.softmax(logits_last, dim=-1)

                    if sampling_mode == 'top_p':
                        # Nucleus sampling
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        # Create mask
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False

                        # Apply mask without scatter
                        sorted_probs = torch.where(
                            sorted_indices_to_remove,
                            torch.tensor(0.0, device=device),
                            sorted_probs
                        )
                        probs = torch.zeros_like(probs, device=device).scatter_(
                            -1, sorted_indices, sorted_probs)
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                        preds = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        # Top-k sampling
                        topk_probs, topk_indices = torch.topk(probs, k=top_k)
                        preds = torch.multinomial(topk_probs, num_samples=1).squeeze(1)
                        preds = torch.gather(topk_indices, -1, preds.unsqueeze(-1)).squeeze(-1)

                    # 7. Update sequences - ensure device consistency
                    current_inputs = torch.cat([
                        current_inputs[:, 1:].to(device),
                        preds.unsqueeze(1).to(device)
                    ], dim=1)

                    # 8. Process templates - explicit device placement
                    pred_templates = []
                    for p in preds:
                        template = self.template_miner.decode_event_id_sequence(p.item())
                        pred_templates.append(template)

                    tokenized = self.tokenizer.batch_transform(pred_templates)
                    tokenized_tensor = torch.tensor(
                        tokenized, 
                        dtype=current_sequences.dtype,
                        device=device
                    )
                    current_sequences = torch.cat([
                        current_sequences[:, 1:].to(device),
                        tokenized_tensor.unsqueeze(1)
                    ], dim=1)

        # 9. Final outputs - ensure device and shape consistency
        outputs_tensor = torch.stack(outputs, dim=1).float().to(device)
        targets = targets.reshape(-1).to(device)

        if self.num_classes > 1:
            outputs_tensor = outputs_tensor.reshape(-1, self.num_classes)
        else:
            outputs_tensor = outputs_tensor.squeeze(-1)

        # Final device check
        assert outputs_tensor.device == device
        assert targets.device == device

        return outputs_tensor, targets

    def on_train_epoch_start(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, logger=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        preds = logits.argmax(dim=-1)
        
        self.log("train/loss_step", loss, on_step=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=-1)
        
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': targets,
            'logits': logits
        })
        
        self.log("val/loss_step", loss, on_step=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs or self.trainer.sanity_checking:
            return
            
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        
        correct = (all_preds == all_targets).long()
        truth = torch.ones_like(correct)
        
        self.val_acc.update(all_preds, all_targets)
        self.val_top5_acc.update(all_logits, all_targets)
        self.val_f1.update(correct, truth)
        self.val_precision.update(correct, truth)
        self.val_recall.update(correct, truth)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        self.log_dict({
            "val/loss": avg_loss,
            "val/acc": self.val_acc.compute(),
            "val/top5_acc": self.val_top5_acc.compute(),
            "val/f1": self.val_f1.compute(),
            "val/precision": self.val_precision.compute(),
            "val/recall": self.val_recall.compute(),
            "val/best_loss": self.best_val_loss,
            "lr": current_lr,
        }, prog_bar=True, sync_dist=True)
        
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        self.log_dict({
            "scheduler/num_bad_epochs": scheduler.num_bad_epochs,
            "scheduler/cooldown_counter": scheduler.cooldown_counter,
        }, logger=True, sync_dist=True)
        
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