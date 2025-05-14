"""
Implements attention mechanisms for transformer architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in 'Attention Is All You Need'.
    
    This computes: softmax(Q * K^T / sqrt(d_k)) * V
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the scaled dot-product attention.
        
        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len_q, d_k]
            key: Key tensor of shape [batch_size, num_heads, seq_len_k, d_k]
            value: Value tensor of shape [batch_size, num_heads, seq_len_v, d_v] (seq_len_k == seq_len_v)
            mask: Optional mask tensor of shape [batch_size, 1, 1, seq_len_k] or [batch_size, 1, seq_len_q, seq_len_k]
                  to mask out certain positions. Values should be 0 (masked) or 1 (unmasked).
            
        Returns:
            attention_output: Result of attention operation of shape [batch_size, num_heads, seq_len_q, d_v]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        # Calculate the attention scores
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Apply mask if provided
        if mask is not None:
            # Fill masked positions with a large negative number before softmax
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Multiply by values
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len_q, d_v]
        
        return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in 'Attention Is All You Need'.
    
    This splits the embedding dimension into multiple heads, applies scaled dot-product
    attention independently, and then concatenates the results and projects them.
    """
    def __init__(self, config):
        super().__init__()
        # Extract config parameters
        self.d_model = getattr(config, 'hidden_dim', 512)
        self.num_heads = getattr(config, 'num_attention_heads', 8)
        self.dropout_rate = getattr(config, 'attention_dropout', 0.1)
        
        # Check if d_model is divisible by num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
            
        self.d_k = self.d_model // self.num_heads  # Dimension of each head
        self.d_v = self.d_model // self.num_heads  # Dimension of each head's values
        
        # Linear projections
        self.wq = nn.Linear(self.d_model, self.d_model)  # Query projection
        self.wk = nn.Linear(self.d_model, self.d_model)  # Key projection
        self.wv = nn.Linear(self.d_model, self.d_model)  # Value projection
        self.wo = nn.Linear(self.d_model, self.d_model)  # Output projection
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=self.dropout_rate)
        
        # Optional dropout after output projection
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            batch_size: Batch size
            
        Returns:
            Tensor of shape [batch_size, num_heads, seq_len, depth]
        """
        # Reshape x to [batch_size, seq_len, num_heads, depth]
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose to [batch_size, num_heads, seq_len, depth]
        return x.transpose(1, 2)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the multi-head attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model] (seq_len_k == seq_len_v)
            mask: Optional mask tensor of shape [batch_size, 1, seq_len_q, seq_len_k]
                  to mask out certain positions. Values should be 0 (masked) or 1 (unmasked).
            
        Returns:
            output: Result of attention operation of shape [batch_size, seq_len_q, d_model]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # Linear projections and split heads
        q = self.split_heads(self.wq(query), batch_size)  # [batch_size, num_heads, seq_len_q, d_k]
        k = self.split_heads(self.wk(key), batch_size)    # [batch_size, num_heads, seq_len_k, d_k]
        v = self.split_heads(self.wv(value), batch_size)  # [batch_size, num_heads, seq_len_v, d_v]
        
        # Scaled dot-product attention
        attention_output, attention_weights = self.attention(q, k, v, mask)
        
        # Reshape back to [batch_size, seq_len_q, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.wo(attention_output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention module where query, key, and value come from the same source.
    This is a wrapper around MultiHeadAttention for convenience.
    """
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            output: Result of attention operation
            attention_weights: Attention weights
        """
        return self.mha(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-attention module where query comes from one source, while key and value come from another.
    This is useful in encoder-decoder attention.
    This is a wrapper around MultiHeadAttention for convenience.
    """
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        
    def forward(self, query, key_value, mask=None):
        """
        Forward pass of the cross-attention.
        
        Args:
            query: Query tensor, typically from decoder of shape [batch_size, seq_len_q, d_model]
            key_value: Key and value tensors, typically from encoder of shape [batch_size, seq_len_kv, d_model]
            mask: Optional mask tensor
            
        Returns:
            output: Result of attention operation
            attention_weights: Attention weights
        """
        return self.mha(query, key_value, key_value, mask) 