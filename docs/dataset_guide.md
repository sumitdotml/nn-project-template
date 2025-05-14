# Dataset Guide: Understanding the Relationship Between `dataset.py` and the `data/` Directory

This guide explains how the `src/dataset.py` file and the `data/` directory work together in this neural network template, and the steps you should take to adapt them for your specific project.

## Overview

The relationship between `dataset.py` and the `data/` directory is a fundamental part of the data loading pipeline:

- **`dataset.py`**: Contains the code for loading, processing, and batching your data.
- **`data/` directory**: Contains the actual data files that will be loaded by the code in `dataset.py`.

## Structure of `data/` Directory

The `data/` directory is organized into subdirectories that correspond to different data splits:

```
data/
├── train/         # Training data
├── val/           # Validation data (optional)
└── test/          # Test data (optional)
```

- **`train/`**: Contains the data used for training your models.
- **`val/`**: Contains the data used for validating your models during training.
- **`test/`**: Contains the data used for final evaluation of your models.

## Current Implementation in `dataset.py`

The `dataset.py` file contains two main components:

1. **`CustomDataset` class**: A template PyTorch Dataset class that you need to customize for your specific data.
2. **`get_dataloader` function**: A utility function that creates PyTorch DataLoaders from your dataset.

The current implementation is a minimal template that you need to adapt for your specific data format.

## How the Training Script Uses These Components

In `scripts/train.py`, the data loading process follows these steps:

1. The script calls `get_dataloader("data/train", batch_size=config.batch_size)` to get a DataLoader for the training data.
2. If a `data/val` directory exists, it also creates a validation DataLoader.
3. These DataLoaders are passed to the Trainer, which uses them during the training process.

## Steps for Using Your Own Data

### 1. Preparing Your Data Directory

1. Place your training data in the `data/train/` directory.
2. If you have validation data, place it in the `data/val/` directory.
3. If you have test data, place it in the `data/test/` directory.

The data can be in any format (images, text files, CSV files, etc.) as long as your `CustomDataset` implementation can read it.

### 2. Customizing `CustomDataset` in `dataset.py`

Modify the `CustomDataset` class in `src/dataset.py` to work with your specific data format:

```python
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Load your data
        # Example: images, text, tabular data, etc.
        # self.data = ...
        # self.labels = ...
        
    def __len__(self):
        # Return the size of your dataset
        return len(self.data)
        
    def __getitem__(self, idx):
        # Get and process a single item
        item = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            item = self.transform(item)
            
        return item, label
```

Focus on implementing these three methods:
- `__init__`: Load your data from `data_path`.
- `__len__`: Return the number of samples in your dataset.
- `__getitem__`: Return a single sample and its label (or target value).

### 3. Optionally Customizing `get_dataloader`

If your dataset requires special handling (like custom collation functions), you may need to modify the `get_dataloader` function:

```python
def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4):
    # Create your dataset with any special arguments
    dataset = CustomDataset(data_path, transform=your_transform)
    
    # Create a DataLoader with any special arguments
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # Add any other arguments as needed
        # collate_fn=your_collate_fn,
        # pin_memory=True,
    )
    return dataloader
```

## Common Data Types and Implementations

### Image Classification

```python
import os
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Assume directory structure: data_path/class_name/image.jpg
        self.image_files = []
        self.labels = []
        classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        for class_name in classes:
            class_path = os.path.join(data_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Text Classification

```python
import pandas as pd
import torch

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.df = pd.read_csv(data_path)  # Assumes a CSV with 'text' and 'label' columns
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
        
        encoding['labels'] = torch.tensor(label)
        return encoding
```

### Time Series

```python
import numpy as np
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, seq_length=50, prediction_window=10):
        # Load data from a CSV or numpy file
        self.data = np.load(data_path)  # Shape: [timesteps, features]
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.data) - (seq_length + prediction_window) + 1):
            self.sequences.append(self.data[i:i+seq_length])
            self.targets.append(self.data[i+seq_length:i+seq_length+prediction_window])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert to torch tensors
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return sequence, target
```