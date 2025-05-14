"""
Models subpackage.
Contains model architectures and building blocks like embeddings, normalization layers, etc.
"""
from .base_model import BaseModel
from .embeddings import TokenEmbedding, PositionalEncoding, TransformerEmbeddings
from .feedforward import FeedForward
from .normalization import LayerNorm # This imports our config-aware LayerNorm wrapper
from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention
)
from .transformer import (
    EncoderLayer,
    DecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Transformer
)

# Future considerations:
# - A model factory function: def create_model(config): ...
# - Import specific attention mechanisms if they are added as separate modules.

__all__ = [
    "BaseModel",
    "TokenEmbedding",
    "PositionalEncoding",
    "TransformerEmbeddings",
    "FeedForward",
    "LayerNorm",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SelfAttention",
    "CrossAttention",
    "EncoderLayer",
    "DecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
]
