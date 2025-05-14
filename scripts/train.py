"""
Training script for neural network models.
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig
from src.dataset import get_dataloader
from src.models import BaseModel
from src.utils.trainer import Trainer


def train_model(config_path, output_dir):
    """
    Train a model with the specified configuration.
    
    Args:
        config_path: Path to the configuration file or model directory
        output_dir: Directory to save the trained model
    """
    # Load configuration
    config = ModelConfig.from_pretrained(config_path)
    
    # Print configuration
    print(f"Training with configuration: {config}")
    
    # Create data loaders
    train_loader = get_dataloader(
        "data/train",
        batch_size=config.batch_size
    )
    
    val_loader = None
    if os.path.exists("data/val"):
        val_loader = get_dataloader(
            "data/val",
            batch_size=config.batch_size,
            shuffle=False
        )
    
    # Initialize model
    model = BaseModel(config)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config.learning_rate
    )
    
    # Create checkpoint directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    history = trainer.train(
        num_epochs=config.epochs,
        save_dir=os.path.join(output_dir, "checkpoints")
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_losses"], label="Train Loss")
    if val_loader is not None:
        plt.plot(history["val_losses"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, "training_loss.png"))
    
    # Save the final model
    model.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a neural network model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to configuration file or directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/base_model",
        help="Directory to save the trained model"
    )
    args = parser.parse_args()
    
    # Train model
    train_model(args.config, args.output_dir)


if __name__ == "__main__":
    main()
