"""
Prediction script for neural network models.
"""

import argparse
import sys
import os
import torch
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.models import BaseModel


def predict(model, data, device):
    """
    Make predictions with a model.
    
    Args:
        model: Model to use for predictions
        data: Input data
        device: Device to use for predictions
        
    Returns:
        Model predictions
    """
    # Move data to device
    data = data.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        output = model(data)
    
    return output


def load_model(config_path, checkpoint_path):
    """
    Load a model with the specified configuration and checkpoint.
    
    Args:
        config_path: Path to the configuration file
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        Loaded model and device
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = BaseModel(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, device


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Make predictions with a neural network model")
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
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.npy",
        help="Path to save predictions"
    )
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.config, args.checkpoint)
    
    # Load data (this implementation is a placeholder - customize for your data format)
    # For example, you might load a numpy array, image, etc.
    try:
        data = torch.from_numpy(np.load(args.input))
    except:
        print(f"Error loading data from {args.input}. Please customize this script for your data format.")
        return
    
    # Make predictions
    predictions = predict(model, data, device)
    
    # Save predictions (customize for your output format)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, predictions.cpu().numpy())
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
