# Neural Network Architecture Template

A barebones template with a clean directory structure for neural network architecture projects. This template provides a starting point for implementing and experimenting with various neural network architectures.

## Features

- Organized directory structure
- Modular code organization
- Basic model template
- Standardized configuration system
- Training, evaluation, and prediction scripts
- Testing framework
- Complete transformer architecture components

## Directory Structure

- `src/`: Core source code
  - `models/`: Model architectures
    - `attention.py`: Attention mechanisms (self, cross, multi-head)
    - `embeddings.py`: Token and positional embeddings
    - `feedforward.py`: Feed-forward networks
    - `normalization.py`: Layer normalization
    - `transformer.py`: Encoder and decoder layers
    - `base_model.py`: Base model class
  - `utils/`: Utility functions
  - `dataset.py`: Data loading and processing
  - `config.py`: Configuration handling
- `scripts/`: Scripts for training, evaluation, and prediction
- `configs/`: Configuration files
- `data/`: Dataset files
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for exploration
- `docs/`: Documentation
- `outputs/`: Output directory for trained models, results, and checkpoints

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone <YOUR_REPOSITORY_URL_HERE>
   cd <YOUR_PROJECT_DIRECTORY_NAME>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv # if using uv, do: uv venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

   # or if using uv
   uv pip install -r requirements.txt
   ```

## Usage

### Configuration

The template uses a JSON-based configuration system similar to Hugging Face Transformers. Configuration files are typically stored in the `configs/` directory (e.g., `configs/default.json`).

**For a detailed guide on creating and using configuration files, including how to adapt configurations from libraries like Hugging Face Transformers, please see [`configs/README.md`](configs/README.md).**

An example `configs/default.json` might look like this:

```json
{
  "model_type": "base_model",
  "input_dim": 784,
  "hidden_dim": 256,
  "output_dim": 10,
  "dropout": 0.2,
  "batch_size": 32,
  "learning_rate": 1e-3,
  "epochs": 10,
  "optimizer": "adam",
  "weight_decay": 1e-5,
  "num_layers": 12,
  "activation_function": "relu"
}
```

The configuration is managed through the `ModelConfig` class in `src/config.py`, which provides a Hugging Face-style API:

```python
# Load a configuration
from src.config import ModelConfig

# Default configuration that will load configs/default.json
config = ModelConfig()

# From a json file
config = ModelConfig.from_json_file("path/to/json/file.json")

# From a pretrained model directory
config = ModelConfig.from_pretrained("outputs/my_model")

# From a dictionary
config = ModelConfig.from_dict({"input_dim": 100, "hidden_dim": 50, "output_dim": 10})

# Create with custom values
config = ModelConfig(input_dim=100, hidden_dim=50, output_dim=10)

# Save configuration
config.save_pretrained("outputs/my_model")
```

### Training

Train a model using a configuration:

```bash
python scripts/train.py --config configs/default.json --output_dir outputs/my_model
```

### Model API

The model follows a Hugging Face-style API for easy saving and loading:

```python
from src.models import BaseModel
from src.config import ModelConfig

# Load a pretrained model
model = BaseModel.from_pretrained("outputs/my_model")

# Or create a new one
config = ModelConfig(input_dim=100, hidden_dim=50, output_dim=10)
model = BaseModel(config)

# Make predictions
import torch
inputs = torch.randn(1, 100)
outputs = model(inputs)

# Save the model
model.save_pretrained("outputs/new_model")
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --config configs/my_model.yaml --checkpoint checkpoints/my_model/final_model.pt
```

### Prediction

Make predictions with a trained model:

```bash
python scripts/predict.py --config configs/my_model.yaml --checkpoint checkpoints/my_model/final_model.pt --input data/test/sample.npy --output predictions.npy
```

## Working with Data

The template includes a flexible system for loading and processing data using PyTorch's Dataset and DataLoader classes.

**For a detailed guide on working with datasets, including examples for different data types, please see [`docs/dataset_guide.md`](docs/dataset_guide.md).**

The basic structure is:
1. Place your data files in the appropriate subdirectories of `data/` (train, val, test)
2. Customize the `CustomDataset` class in `src/dataset.py` to load and process your specific data format
3. Use the `get_dataloader` function to create batches for training and evaluation

## Transformers Components

The template includes a complete implementation of transformer architecture components:

```python
from src.models import (
    # Embedding components
    TokenEmbedding, PositionalEncoding, TransformerEmbeddings,
    
    # Attention mechanisms
    ScaledDotProductAttention, MultiHeadAttention, SelfAttention, CrossAttention,
    
    # Other components
    FeedForward, LayerNorm,
    
    # Transformer layers
    EncoderLayer, DecoderLayer, TransformerEncoder, TransformerDecoder,
    
    # Complete models
    Transformer, BaseModel
)

# Example: Create a transformer model
config = ModelConfig(
    hidden_dim=512, 
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_attention_heads=8
)
transformer = Transformer(config)
```

For examples of how to use these components, see the example notebook at [`notebooks/transformer_example.ipynb`](notebooks/transformer_example.ipynb).

## Customization

### Custom Model

Create a custom model by extending the base model:

```python
# src/models/custom_model.py
from src.models.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers
        
    def forward(self, x):
        # Implement custom forward pass
        return x
```

Update `src/models/__init__.py` to import your custom model:

```python
from src.models.base_model import BaseModel
from src.models.custom_model import CustomModel
```

### Custom Dataset

Implement your custom dataset by modifying `src/dataset.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- PyTorch for the deep learning framework
- Open-source neural network architectures that inspired this template

## Project Helper Script

The template includes a helper script (`scripts/setup_template.sh`) to quickly create new project components:

```bash
# Run the helper script
./scripts/setup_template.sh
```

This interactive script allows you to:
1. Create a new model (extends BaseModel)
2. Create a new dataset
3. Create a new utility function
4. Create a new experiment script
5. Set up project directories and documentation

This is particularly useful for extending the template with your own implementations.
