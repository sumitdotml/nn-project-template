"""
Base model template for neural network architectures.
"""

import torch
import torch.nn as nn
from src.config import ModelConfig


class BaseModel(nn.Module):
    """
    Base model class for neural network architectures.
    This is a simple template - customize according to your needs.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model.
        
        Args:
            config: ModelConfig instance with model parameters
        """
        super().__init__()
        self.config = config
        
        # Define model layers based on configuration
        # This is a minimal example - modify for your architecture
        self.layers = nn.ModuleList([
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        ])
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and configuration to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        
        # Create directory if it doesn't exist, ensuring save_directory is within /outputs/
        if not save_directory.startswith('outputs/'):
            save_directory = os.path.join('outputs', save_directory)
            
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Load model from a pretrained directory or model ID.
        
        Args:
            model_path: Path to the model directory or model ID
            
        Returns:
            Loaded model
        """
        import os
        from src.config import ModelConfig
        
        # Load configuration
        config = ModelConfig.from_pretrained(model_path)
        
        # Initialize model with config
        model = cls(config)
        
        # Try to load weights if available
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.isfile(model_file):
            model.load_state_dict(torch.load(model_file))
        elif os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
        
        return model
