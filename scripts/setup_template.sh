#!/bin/bash

# Project component creator script
# This script helps you create new components for your neural network project

echo "Neural Network Project Component Creator"
echo "========================================"

# Function to create a new model
create_model() {
    local model_name=$1
    local model_file="src/models/${model_name}.py"
    
    if [ -f "$model_file" ]; then
        echo "Error: Model $model_name already exists at $model_file"
        return 1
    fi
    
    echo "Creating new model: $model_name"
    
    # Create the model file
    cat > "$model_file" << EOL
"""
$model_name model implementation.
"""

import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class ${model_name^}(BaseModel):
    """
    ${model_name^} model implementation.
    Extends the BaseModel class.
    """
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # TODO: Define custom model architecture here
        # This overrides the simple architecture from BaseModel
        self.layers = nn.ModuleList([
            # Define your layers here
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.2)),
            nn.Linear(config["hidden_dim"], config["output_dim"])
        ])
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # TODO: Implement custom forward pass logic
        for layer in self.layers:
            x = layer(x)
        return x
EOL

    # Update the __init__.py file
    if ! grep -q "from src.models.${model_name} import ${model_name^}" "src/models/__init__.py"; then
        # Add import to __init__.py
        sed -i "" "s/from src.models.base_model import BaseModel/from src.models.base_model import BaseModel\nfrom src.models.${model_name} import ${model_name^}/" "src/models/__init__.py"
    fi
    
    # Create a config file for the model
    local config_file="configs/${model_name}.yaml"
    if [ ! -f "$config_file" ]; then
        cat > "$config_file" << EOL
# Configuration for ${model_name^} model
model:
  type: ${model_name}
  input_dim: 784  # Customize for your data
  hidden_dim: 256
  output_dim: 10
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 20
  optimizer: adam
  weight_decay: 1e-5

data:
  train_path: data/train
  val_path: data/val
  test_path: data/test
EOL
    fi
    
    echo "Model $model_name created successfully!"
    echo "Files created:"
    echo "- $model_file"
    echo "- $config_file"
    echo ""
    echo "Usage:"
    echo "  python scripts/train.py --config configs/${model_name}.yaml"
}

# Function to create a new dataset
create_dataset() {
    local dataset_name=$1
    local dataset_file="src/datasets/${dataset_name}.py"
    
    # Create datasets directory if it doesn't exist
    mkdir -p "src/datasets"
    
    if [ -f "$dataset_file" ]; then
        echo "Error: Dataset $dataset_name already exists at $dataset_file"
        return 1
    fi
    
    # Create __init__.py if it doesn't exist
    if [ ! -f "src/datasets/__init__.py" ]; then
        echo '"""
Dataset implementations.
"""' > "src/datasets/__init__.py"
    fi
    
    echo "Creating new dataset: $dataset_name"
    
    # Create the dataset file
    cat > "$dataset_file" << EOL
"""
$dataset_name dataset implementation.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class ${dataset_name^}Dataset(Dataset):
    """
    ${dataset_name^} dataset implementation.
    """
    
    def __init__(self, data_path, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.transform = transform
        
        # TODO: Load your data here
        self.data = []
        self.targets = []
        self._load_data()
        
    def _load_data(self):
        """
        Load data from disk.
        """
        # TODO: Implement data loading logic
        pass
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Get a data item.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (data, target)
        """
        # TODO: Implement data retrieval logic
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(data)
            
        return data, target


def get_${dataset_name}_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the ${dataset_name} dataset.
    
    Args:
        data_path: Path to the data
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads for loading data
        
    Returns:
        DataLoader instance
    """
    dataset = ${dataset_name^}Dataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
EOL

    # Update the __init__.py file
    if ! grep -q "from src.datasets.${dataset_name} import" "src/datasets/__init__.py"; then
        # Add import to __init__.py
        echo "from src.datasets.${dataset_name} import ${dataset_name^}Dataset, get_${dataset_name}_dataloader" >> "src/datasets/__init__.py"
    fi
    
    echo "Dataset $dataset_name created successfully!"
    echo "Files created:"
    echo "- $dataset_file"
    echo ""
    echo "Usage example:"
    echo "  from src.datasets.${dataset_name} import get_${dataset_name}_dataloader"
    echo "  dataloader = get_${dataset_name}_dataloader('path/to/data')"
}

# Function to create a new utility
create_utility() {
    local util_name=$1
    local util_file="src/utils/${util_name}.py"
    
    if [ -f "$util_file" ]; then
        echo "Error: Utility $util_name already exists at $util_file"
        return 1
    fi
    
    echo "Creating new utility: $util_name"
    
    # Create the utility file
    cat > "$util_file" << EOL
"""
$util_name utility functions.
"""

# TODO: Implement utility functions here
def example_function():
    """
    Example function.
    
    Returns:
        Dummy value
    """
    return True
EOL

    # Update the __init__.py file
    if ! grep -q "from src.utils.${util_name} import" "src/utils/__init__.py"; then
        # Add import to __init__.py
        echo "from src.utils.${util_name} import *" >> "src/utils/__init__.py"
    fi
    
    echo "Utility $util_name created successfully!"
    echo "File created: $util_file"
}

# Function to create a new experiment script
create_experiment() {
    local exp_name=$1
    local exp_file="scripts/${exp_name}.py"
    
    if [ -f "$exp_file" ]; then
        echo "Error: Experiment script $exp_name already exists at $exp_file"
        return 1
    fi
    
    echo "Creating new experiment script: $exp_name"
    
    # Create the experiment file
    cat > "$exp_file" << EOL
"""
$exp_name experiment script.
"""

import argparse
import sys
import os
import yaml
import torch
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.dataset import get_dataloader
from src.models import BaseModel
from src.utils.trainer import Trainer


def run_experiment(config_path):
    """
    Run the experiment with the specified configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    print(f"Running {exp_name} experiment with config from {config_path}")
    
    # TODO: Implement your experiment logic here
    
    # Example: Create data loaders
    train_loader = get_dataloader(
        config["data"]["train_path"],
        batch_size=config["training"]["batch_size"]
    )
    
    # Example: Initialize model
    model = BaseModel(config["model"])
    
    # TODO: Add your custom experiment code
    print("Experiment complete!")


def main():
    parser = argparse.ArgumentParser(description="${exp_name^} experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Run the experiment
    run_experiment(args.config)


if __name__ == "__main__":
    main()
EOL
    
    echo "Experiment script $exp_name created successfully!"
    echo "File created: $exp_file"
    echo ""
    echo "Usage:"
    echo "  python scripts/${exp_name}.py --config configs/your_config.yaml"
}

# Function to create folders and files for a new project
setup_project() {
    # Create essential directories if they don't exist
    mkdir -p data/{train,val,test}
    mkdir -p checkpoints
    mkdir -p docs/plots
    
    # Create an example Markdown file in data directory
    if [ ! -f "data/README.md" ]; then
        cat > "data/README.md" << EOL
# Data Directory

This directory is for storing datasets for your neural network project.

## Structure

- \`train/\`: Training data
- \`val/\`: Validation data
- \`test/\`: Test data

## Adding Data

Place your dataset files in the appropriate subdirectories.
For very large datasets, consider adding them to .gitignore and providing download instructions here.
EOL
    fi
    
    # Create an example Markdown file in checkpoints directory
    if [ ! -f "checkpoints/README.md" ]; then
        cat > "checkpoints/README.md" << EOL
# Checkpoints Directory

This directory stores model checkpoints from training.

## Structure

Checkpoints are typically organized by experiment name:

\`\`\`
checkpoints/
├── experiment1/
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── final_model.pt
└── experiment2/
    ├── epoch_1.pt
    └── final_model.pt
\`\`\`

## Usage

When training a model, checkpoints will be saved here automatically.
EOL
    fi
    
    echo "Project directories set up successfully!"
}

# Main menu
show_menu() {
    echo ""
    echo "Please choose an option:"
    echo "1) Create a new model"
    echo "2) Create a new dataset"
    echo "3) Create a new utility"
    echo "4) Create a new experiment script"
    echo "5) Setup project directories"
    echo "q) Quit"
    echo ""
    echo -n "Enter your choice: "
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            echo -n "Enter model name (lowercase, no spaces): "
            read model_name
            create_model "$model_name"
            ;;
        2)
            echo -n "Enter dataset name (lowercase, no spaces): "
            read dataset_name
            create_dataset "$dataset_name"
            ;;
        3)
            echo -n "Enter utility name (lowercase, no spaces): "
            read util_name
            create_utility "$util_name"
            ;;
        4)
            echo -n "Enter experiment name (lowercase, no spaces): "
            read exp_name
            create_experiment "$exp_name"
            ;;
        5)
            setup_project
            ;;
        q|Q)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done 