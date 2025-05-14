"""
Tests for the model implementation.
"""

import unittest
import sys
import os
import torch

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import BaseModel
from src.config import DEFAULT_CONFIG


class TestModel(unittest.TestCase):
    """Test cases for the model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DEFAULT_CONFIG["model"]
        self.model = BaseModel(self.config)
        
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        # Create a random input tensor
        batch_size = 2
        input_dim = self.config["input_dim"]
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        expected_shape = (batch_size, self.config["output_dim"])
        self.assertEqual(output.shape, expected_shape)
    
    def test_save_and_load(self):
        """Test saving and loading the model."""
        # Create a temporary directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Define a checkpoint path
        checkpoint_path = os.path.join(temp_dir, "model.pt")
        
        # Save the model
        self.model.save_checkpoint(checkpoint_path)
        
        # Load the model
        loaded_model = BaseModel.load_from_checkpoint(checkpoint_path)
        
        # Check that the loaded model has the same parameters
        for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
