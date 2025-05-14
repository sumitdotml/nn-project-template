"""
Evaluation script for neural network models.
"""

import argparse
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.dataset import get_dataloader
from src.models import BaseModel


def evaluate_model(config_path, checkpoint_path):
    """
    Evaluate a model with the specified configuration and checkpoint.
    
    Args:
        config_path: Path to the configuration file
        checkpoint_path: Path to the model checkpoint
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create data loader
    test_loader = get_dataloader(
        config["data"]["test_path"] if "test_path" in config["data"] else config["data"]["val_path"],
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )
    
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = BaseModel(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Evaluate model
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            preds = torch.argmax(output, dim=1)
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate a neural network model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    args = parser.parse_args()
    
    # Evaluate model
    evaluate_model(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
