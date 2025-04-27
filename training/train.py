import torchmetrics
import importlib
from models import Transformer, LogTemplateMiner, LogTokenizer, compute_class_weights
from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, config, test_mode=False):
        super().__init__()
        self.config = config
        self.template_miner = LogTemplateMiner(config['dataset']['drain_path'])
        self.tokenizer = LogTokenizer(config['dataset']['vocab_path'])
        self.test_mode = test_mode
        
        dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
        self.dataset_class = getattr(dataset_module, "Dataset")
        self.train_dataset = None
        self.val_dataset = None
        self.setup_f = False
        
        self.template_miner.load_state()

    def setup(self, stage=None):
        if not self.setup_f:
            dataset = self.dataset_class(
                path=self.config['dataset']['path'], 
                prediction_steps=self.config['model']['prediction_steps'],
                context_length=self.config['model']['context_length'],
                transform=self.template_miner.transform, 
                tokenizer=self.tokenizer,
                test_mode=self.test_mode
            )

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

import pytorch_lightning as pl
import torch
import torchmetrics

class TransformerLightning(pl.LightningModule):
    def __init__(self, config, config_name, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(config)
        self.config_name = config_name

        if class_weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1, ignore_index=0)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

        self.num_classes = config['model']['vocab_size']

        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average='micro')
        self.val_top5_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)

        # Binary metrics on correctness
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")

    def forward(self, x):
        return self.model(x)

    def _process_batch(self, batch):
        inputs, targets = batch  # inputs: [batch, seq_len]

        outputs = []

        for step in range(self.model.n_steps):
            logits = self(inputs)  # [batch, seq_len, vocab_size]
            logits_last = logits[:, -1, :]  # [batch, vocab_size]

            preds = logits_last.argmax(dim=-1)  # [batch]

            outputs.append(logits_last)  # Save logits

            # Shift input: remove first token, append prediction
            inputs = torch.cat([inputs[:, 1:], preds.unsqueeze(1)], dim=1)  # [batch, seq_len]

        outputs = torch.stack(outputs, dim=1)  # [batch, n_steps, vocab_size]

        return outputs.view(-1, outputs.size(-1)), targets.view(-1)
        
    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        preds = torch.argmax(logits, dim=-1)

        correct = (preds == targets).long()
        truth = torch.ones_like(correct)

        self.val_accuracy.update(logits, targets)
        self.val_top5_acc.update(logits, targets)
        self.val_f1.update(correct, truth)
        self.val_precision.update(correct, truth)
        self.val_recall.update(correct, truth)

        return loss

    def on_validation_epoch_end(self):
        self.log("val/accuracy", self.val_accuracy.compute(), sync_dist=True)
        self.log("val/top5_accuracy", self.val_top5_acc.compute(), sync_dist=True)
        self.log("val/f1", self.val_f1.compute(), sync_dist=True)
        self.log("val/precision", self.val_precision.compute(), sync_dist=True)
        self.log("val/recall", self.val_recall.compute(), sync_dist=True)

        self.val_accuracy.reset()
        self.val_top5_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.config['training'].get('lr', 1e-5)
        )
