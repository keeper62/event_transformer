import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import logging

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMLightning(pl.LightningModule):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int = 128, 
                 hidden_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout: float = 0.3, 
                 seq_len: int = 512, 
                 important_classes: torch.Tensor | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=['important_classes'])
        
        # Setup logger
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO)
        self._logger.info("Initializing LSTMGRULightning model")
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        self._logger.info(f"Initialized LSTM with hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Important classes
        self.important_classes = important_classes if important_classes is not None else torch.tensor([], dtype=torch.long)
        self.importance_boost_factor = 15.0
        self._logger.info(f"Important classes set: {self.important_classes.tolist() if len(self.important_classes) > 0 else 'None'}")
        
        # Loss
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics
        self._init_metrics()
        
    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.seq_len)
        logits = self.fc(output)
        return logits

    def _init_metrics(self):
        self.train_metrics = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(task='multiclass', num_classes=self.vocab_size),
            "top5": torchmetrics.Accuracy(task='multiclass', num_classes=self.vocab_size, top_k=5)
        }, prefix="train/")
        
        self.val_metrics = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(task='multiclass', num_classes=self.vocab_size),
            "top5": torchmetrics.Accuracy(task='multiclass', num_classes=self.vocab_size, top_k=5)
        }, prefix="val/")
        
    def _process_batch(self, batch):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        return logits.view(-1, logits.size(-1)), targets.view(-1)
        
    def training_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        metrics = self.train_metrics(logits, targets)
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        logits, targets = self._process_batch(batch)
        loss = self.loss_fn(logits, targets)
        self.val_metrics.update(logits, targets)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
        
    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}
        }

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            self._logger.info(f"Batch {batch_idx} memory usage:")
            if torch.cuda.is_available():
                self._logger.info(f"GPU Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                self._logger.info(f"GPU Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            else:
                import psutil
                process = psutil.Process()
                self._logger.info(f"CPU Memory Used: {process.memory_info().rss/1e9:.2f}GB")
                self._logger.info(f"System Available: {psutil.virtual_memory().available/1e9:.2f}GB")
