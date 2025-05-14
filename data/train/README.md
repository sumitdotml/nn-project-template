# Training Data Directory

Place your training dataset files in this directory. The `src/dataset.py` will load data from this location.

## Data Format

The default `CustomDataset` class in `src/dataset.py` expects you to implement the data loading logic. For example:

1. For image classification: Place your images in this directory, possibly organized by class in subdirectories.
2. For text tasks: Store text files or preprocessed token files here.
3. For tabular data: CSV, JSON, or other structured data files.

## Implementation

To use your data, you'll need to update the `CustomDataset.__init__` and `CustomDataset.__getitem__` methods in `src/dataset.py` to properly load and process your specific data format. 