import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models import load_config, LogTemplateMiner, LogTokenizer
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

important_errors = torch.tensor([42, 101, 567, 1423, 3500, 4582], dtype=torch.long)

def setup_environment(seed: int = 42) -> None:
    """Set up the training environment with reproducibility."""
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    sys.path.append(os.path.abspath("."))
    torch.set_float32_matmul_precision('medium')

def log_device_info():
    """Log comprehensive device and environment information"""
    logger.info("===== Environment Configuration =====")
    logger.info(f"PyTorch Lightning v{pl.__version__}")
    logger.info(f"PyTorch v{torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Current Device: {torch.cuda.current_device()}")
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"Num CPUs: {os.cpu_count()}")
    logger.info("====================================")

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
) -> None:
    """Train the model with the given configuration."""
    log_device_info()
    
    try:
        # Initialize data module
        logger.debug("Initializing DataModule")
        data_module = DataModule(config, test_mode=test_mode)
        
        # Initialize components
        logger.debug("Initializing LogTemplateMiner...")
        template_miner = LogTemplateMiner(config['dataset']['drain_path'])
            
        logger.debug("Loading template miner state...")
        template_miner.load_state()
            
        logger.debug("Initializing LogTokenizer...")
        tokenizer = LogTokenizer(
            config['tokenizer']['tokenizer_length'], 
            tokenizer_path=config['tokenizer']['tokenizer_path']
        )
        
        # Get vocabulary sizes
        config['model']['vocab_size'] = template_miner.get_vocab_size()
        logger.debug(f"Vocab size template miner: {config['model']['vocab_size']}")
        
        config['tokenizer']['vocab_size'] = tokenizer.get_vocab_size()
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
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=config['training'].get('num_epochs', 10),
            devices=num_accelerators,
            accelerator=accelerator,
            num_nodes=num_nodes,
            strategy='ddp_find_unused_parameters_true',
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
        trainer.fit(model, datamodule=data_module)
        
        # Handle testing based on available checkpoints
        checkpoint_callback = next(
            (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), 
            None
        )
        
        if test_mode:
            logger.info("Running in test mode")
            trainer.test(model=model, datamodule=data_module)
        else:
            # Save final model
            save_dir = Path("saved_models")
            save_dir.mkdir(exist_ok=True)
            model_path = save_dir / f"trained_model_{config_name}_final.ckpt"
            trainer.save_checkpoint(str(model_path))
            logger.info(f"Final model saved to {model_path}")
            
            # Determine which model to test
            if checkpoint_callback and checkpoint_callback.best_model_path:
                logger.info(f"Testing with best checkpoint: {checkpoint_callback.best_model_path}")
                trainer.test(ckpt_path="best", datamodule=data_module)
            else:
                logger.info("No best checkpoint found, testing with final model")
                trainer.test(model=model, datamodule=data_module)

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

def main() -> None:
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train Transformer model with specified configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base_config.yaml", 
        help="Path to the config file."
    )
    parser.add_argument(
        "--num_nodes", 
        type=int, 
        default=1, 
        help="Number of distributed nodes."
    )
    parser.add_argument(
        "--accelerator", 
        type=str, 
        default="auto",  # Changed from "cpu" to "auto" for better flexibility
        choices=["auto", "cpu", "gpu", "tpu"],
        help="Which accelerator to use."
    )
    parser.add_argument(
        "--num_accelerators", 
        type=int, 
        default=1, 
        help="Number of GPUs or CPUs to use."
    )
    parser.add_argument(
        "--test_mode", 
        action="store_true", 
        help="Enable test mode with a small dataset."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with additional logging."
    )

    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.INFO)
        logger.debug("Debug mode enabled")
    
    setup_environment()
    
    try:
        config, config_name = load_config(args.config), Path(args.config).stem
        logger.info(f"Training with configuration: {config_name}")
        logger.info(f"Arguments: {vars(args)}")
        
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