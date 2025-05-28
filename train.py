import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models import load_config
from training import DataModule, TransformerLightning

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

# Usage:
logger = setup_logger(__name__)

important_errors = torch.tensor([42, 101], dtype=torch.long)

def setup_environment(seed: int = 42) -> None:
    """Set up the training environment with reproducibility."""
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    sys.path.append(os.path.abspath("."))
    torch.set_float32_matmul_precision('medium')

def get_callbacks(config: Dict[str, Any], config_name: str, test_mode: bool = False) -> list:
    """Create and return a list of callbacks for the trainer."""
    if test_mode:
        return []
    
    callbacks = []
    
    # Model checkpoint callback - improved configuration
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{config_name}-{{epoch}}-{{val/loss:.2f}}",  # Use config name directly
        monitor=config['training'].get('monitor', 'val/loss'),
        mode="min",
        save_top_k=config['training'].get('save_top_k', 1),
        save_last=True,
        auto_insert_metric_name=False,  # Cleaner filenames
        every_n_epochs=1,
        save_on_train_epoch_end=False  # Validate at end of validation epoch
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback if configured
    if config['training'].get('early_stopping', {}).get('enabled', False):
        early_stop_callback = EarlyStopping(
            monitor=config['training'].get('monitor', 'val/loss'),
            patience=config['training']['early_stopping']['patience'],
            mode="min",
            verbose=True,
            min_delta=config['training']['early_stopping'].get('min_delta', 0.0)
        )
        callbacks.append(early_stop_callback)
    
    return callbacks

def train_with_config(
    config: Dict[str, Any],
    config_name: str,
    num_accelerators: int,
    num_nodes: int,
    accelerator: str,
    test_mode: bool = False
) -> Dict[str, Any]:
    """Train the model with the given configuration.
    """
    try:
        # Initialize data module
        logger.debug("Initializing DataModule")
        data_module = DataModule(config, test_mode=test_mode)
        
        # Get vocabulary sizes
        config['model']['vocab_size'] = data_module.template_miner.get_vocab_size()
        logger.debug(f"Vocab size template miner: {config['model']['vocab_size']}")
        
        config['tokenizer']['vocab_size'] = data_module.tokenizer.get_vocab_size()
        logger.debug(f"Vocab size tokenizer: {config['tokenizer']['vocab_size']} samples")
        
        # Initialize model
        logger.debug("Initializing model")
        model = TransformerLightning(
            config, 
            config_name, 
            important_classes=important_errors,
        )

        # Configure logger and callbacks
        logger_obj = TensorBoardLogger("logs/", name=config_name) if not test_mode else None
        callbacks = get_callbacks(config, config_name, test_mode)
        
        # Create training trainer
        train_trainer = pl.Trainer(
            max_epochs=1 if test_mode else config['training'].get('num_epochs', 10),
            devices=num_accelerators,
            accelerator=accelerator,
            num_nodes=num_nodes,
            strategy='ddp_find_unused_parameters_true' if accelerator=='gpu' else 'auto',
            logger=logger_obj,
            callbacks=callbacks,
            gradient_clip_val=config['training'].get('gradient_clip_val', None),
            deterministic=True,
            enable_checkpointing=not test_mode,
            log_every_n_steps=config['training'].get('log_interval', 50),
            fast_dev_run=test_mode,
            overfit_batches=config['training'].get('overfit_batches', 0),
            enable_progress_bar=int(os.getenv("LOCAL_RANK", 0)) == 0,
            precision='16-mixed',
            accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1)
        )

        logger.info(f"Starting training with config: {config_name}")
        train_trainer.fit(model, datamodule=data_module)
        
        if not test_mode:
        # Save final model
            save_dir = Path("saved_models")
            save_dir.mkdir(exist_ok=True)
            model_path = save_dir / f"trained_model_{config_name}_final.ckpt"
            train_trainer.save_checkpoint(str(model_path))
            logger.info(f"Final model saved to {model_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

def main() -> None:
    """Enhanced main function with better result handling."""
    parser = argparse.ArgumentParser(
        description="Train Transformer model with specified configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu", "tpu"])
    parser.add_argument("--num_accelerators", type=int, default=1)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--notes", type=str, default="None")

    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    setup_environment()
    
    try:
        config, config_name = load_config(args.config), Path(args.config).stem
        logger.info(f"Training with configuration: {config_name}")
        
        train_with_config(
            config['base_config'], 
            config_name, 
            args.num_accelerators, 
            args.num_nodes, 
            args.accelerator, 
            test_mode=args.test_mode
        )
        
        logger.info("Training completed successfully.")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()