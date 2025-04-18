import torchmetrics
import importlib
from models import Transformer, LogTokenizer, compute_class_weights
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, config, test_mode=False):
        super().__init__()
        self.config = config
        self.tokenizer = LogTokenizer(config['dataset']['drain_path'])
        self.test_mode = test_mode
        
        dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
        self.dataset_class = getattr(dataset_module, "Dataset")
        self.train_dataset = None
        self.val_dataset = None
        self.setup_f = False

    def setup(self, stage=None):
        if not self.setup_f:
            dataset = self.dataset_class(
                path=self.config['dataset']['path'], 
                prediction_steps=self.config['model']['prediction_steps'],
                context_length=self.config['model']['context_length'],
                transform=self.tokenizer.transform, 
                test_mode=self.test_mode
            )

            self.tokenizer.load_state()

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
            self.setup_f = True

    def get_class_weights(self):
        if self.train_dataset is None:
            self.setup()

        return compute_class_weights(self.train_dataset, self.config['model']['vocab_size'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

class TransformerLightning(pl.LightningModule):
    def __init__(self, config, config_name, test_mode=False, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(config)
        self.config_name = config_name
        self.test_mode = test_mode
        
        if class_weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1, ignore_index=0)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
        
        self.num_classes = config['model']['vocab_size']

        # Define metrics — will move them to CPU later
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average='micro')
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average='micro')
        
        self.train_top5_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)
        self.val_top5_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='micro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='micro')

    def forward(self, x, timestamps):
        return self.model(x, timestamps)

    def _process_batch(self, batch):
        inputs, targets, timestamps = batch
        outputs = self(inputs, timestamps)
        return (
            outputs.view(-1, outputs.size(-1)),  # [B*T, C]
            targets.view(-1)                     # [B*T]
        )

    def training_step(self, batch, batch_idx):
        if self.test_mode and batch_idx > 5:
            return None

        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        
        # Update metrics
        self.train_accuracy.update(logits, targets)
        self.train_top5_acc.update(logits, targets)
        self.train_f1.update(logits, targets)

        # Log loss only
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        # Compute and log metrics manually
        self.log("train/accuracy", self.train_accuracy.compute())
        self.log("train/top5_accuracy", self.train_top5_acc.compute())
        self.log("train/f1_micro", self.train_f1.compute(), prog_bar=True)

        # Always reset!
        self.train_accuracy.reset()
        self.train_top5_acc.reset()
        self.train_f1.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)

        self.val_accuracy.update(logits, targets)
        self.val_top5_acc.update(logits, targets)
        self.val_f1.update(logits, targets)

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            self.log("val/accuracy", self.val_accuracy.compute())
            self.log("val/top5_accuracy", self.val_top5_acc.compute())
            self.log("val/f1_micro", self.val_f1.compute(), prog_bar=True)

        self.val_accuracy.reset()
        self.val_top5_acc.reset()
        self.val_f1.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.config['training'].get('lr', 1e-5)
        )
