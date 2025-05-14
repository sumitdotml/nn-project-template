"""
Implements embedding layers, such as token embeddings and positional embeddings.
"""
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x is expected to be a tensor of token indices [batch_size, seq_len]
        return self.embedding(x) * math.sqrt(self.embed_dim) # Scaling by sqrt(d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape [1, max_len, d_model] for broadcasting
        self.register_buffer('pe', pe) # Not a model parameter, but should be part of state_dict

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # self.pe is [1, max_len, d_model]
        # x is [batch_size, seq_len, d_model]
        # We need self.pe[:, :x.size(1), :] which is [1, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = getattr(config, 'vocab_size', 30522) # Example: BERT vocab size
        self.d_model = getattr(config, 'hidden_dim', 512)
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 512)
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        self.layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        self.token_embeddings = TokenEmbedding(self.vocab_size, self.d_model)
        self.position_embeddings = PositionalEncoding(self.d_model, self.max_position_embeddings, self.dropout_rate)
        
        # Optional: Layer Normalization after summing embeddings, common in many Transformer architectures
        self.layer_norm = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        Args:
            input_ids (torch.Tensor): Tensor of input token indices, shape [batch_size, seq_len].
            token_type_ids (torch.Tensor, optional): Tensor of token type IDs for segment embeddings, 
                                                     shape [batch_size, seq_len]. Defaults to None.
            position_ids (torch.Tensor, optional): Tensor of position IDs. If None, positions are created
                                                 sequentially. Shape [batch_size, seq_len]. Defaults to None.
        Returns:
            torch.Tensor: The combined embeddings, shape [batch_size, seq_len, d_model].
        """
        # input_ids: [batch_size, seq_len]
        
        x = self.token_embeddings(input_ids) # [batch_size, seq_len, d_model]
        
        # Note: Original Transformer and BERT add positional embeddings.
        # Some architectures might use learned positional embeddings instead.
        x = self.position_embeddings(x)      # [batch_size, seq_len, d_model]
        
        # If token_type_ids are provided (e.g., for BERT-like models with segment embeddings):
        # if token_type_ids is not None:
        #   if not hasattr(self, 'token_type_embeddings'):
        #       # Initialize token_type_embeddings if not already present
        #       # Typically, num_token_types is small (e.g., 2 for BERT)
        #       num_token_types = getattr(config, 'num_token_types', 2)
        #       self.token_type_embeddings = nn.Embedding(num_token_types, self.d_model)
        #   type_embeddings = self.token_type_embeddings(token_type_ids)
        #   x = x + type_embeddings
            
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x 