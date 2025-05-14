"""
Implements transformer encoder and decoder layers, as described in 'Attention Is All You Need'.
"""
import torch
import torch.nn as nn
from .attention import MultiHeadAttention, SelfAttention, CrossAttention
from .feedforward import FeedForward
from .normalization import LayerNorm


class EncoderLayer(nn.Module):
    """
    Transformer encoder layer, consisting of self-attention and feed-forward networks,
    with residual connections and layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        
        # Self-attention sublayer
        self.self_attention = SelfAttention(config)
        self.attention_norm = LayerNorm(config)
        
        # Feed-forward sublayer
        self.feed_forward = FeedForward(config)
        self.ff_norm = LayerNorm(config)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the encoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask for the self-attention
            
        Returns:
            Output tensor of the same shape as input
        """
        # Self-attention sublayer with residual connection and normalization
        attention_output, _ = self.self_attention(x, mask)
        residual_output = x + self.dropout1(attention_output)
        norm_output = self.attention_norm(residual_output)
        
        # Feed-forward sublayer with residual connection and normalization
        ff_output = self.feed_forward(norm_output)
        residual_output = norm_output + self.dropout2(ff_output)
        norm_output = self.ff_norm(residual_output)
        
        return norm_output


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer, consisting of self-attention, cross-attention, 
    and feed-forward networks, with residual connections and layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        
        # Self-attention sublayer (masked)
        self.self_attention = SelfAttention(config)
        self.self_attention_norm = LayerNorm(config)
        
        # Cross-attention sublayer (encoder-decoder attention)
        self.cross_attention = CrossAttention(config)
        self.cross_attention_norm = LayerNorm(config)
        
        # Feed-forward sublayer
        self.feed_forward = FeedForward(config)
        self.ff_norm = LayerNorm(config)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
    def forward(self, x, enc_output, target_mask=None, encoder_mask=None):
        """
        Forward pass of the decoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, target_seq_len, hidden_dim]
            enc_output: Encoder output of shape [batch_size, source_seq_len, hidden_dim]
            target_mask: Optional mask for the self-attention to prevent attending to future tokens
            encoder_mask: Optional mask for the cross-attention to prevent attending to padded encoder positions
            
        Returns:
            Output tensor of the same shape as input [batch_size, target_seq_len, hidden_dim]
        """
        # Self-attention sublayer (masked)
        sa_output, _ = self.self_attention(x, target_mask)
        residual_output = x + self.dropout1(sa_output)
        norm_output = self.self_attention_norm(residual_output)
        
        # Cross-attention sublayer (encoder-decoder attention)
        ca_output, _ = self.cross_attention(norm_output, enc_output, encoder_mask)
        residual_output = norm_output + self.dropout2(ca_output)
        norm_output = self.cross_attention_norm(residual_output)
        
        # Feed-forward sublayer
        ff_output = self.feed_forward(norm_output)
        residual_output = norm_output + self.dropout3(ff_output)
        norm_output = self.ff_norm(residual_output)
        
        return norm_output


class TransformerEncoder(nn.Module):
    """
    Full transformer encoder, consisting of multiple encoder layers.
    """
    def __init__(self, config):
        super().__init__()
        self.num_layers = getattr(config, 'num_encoder_layers', 6)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(self.num_layers)])
        
        # Final layer normalization (common in transformers like BERT)
        self.norm = LayerNorm(config)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the transformer encoder.
        
        Args:
            x: Input tensor, usually token embeddings, of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask for the self-attention
            
        Returns:
            Output tensor representing encoded sequence
        """
        # Pass through each encoder layer in sequence
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final normalization
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Full transformer decoder, consisting of multiple decoder layers.
    """
    def __init__(self, config):
        super().__init__()
        self.num_layers = getattr(config, 'num_decoder_layers', 6)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_layers)])
        
        # Final layer normalization
        self.norm = LayerNorm(config)
        
    def forward(self, x, enc_output, target_mask=None, encoder_mask=None):
        """
        Forward pass of the transformer decoder.
        
        Args:
            x: Input tensor, usually token embeddings, of shape [batch_size, target_seq_len, hidden_dim]
            enc_output: Encoder output of shape [batch_size, source_seq_len, hidden_dim]
            target_mask: Optional mask for the self-attention to prevent attending to future tokens
            encoder_mask: Optional mask for the cross-attention to prevent attending to padded encoder positions
            
        Returns:
            Output tensor representing the decoded sequence
        """
        # Pass through each decoder layer in sequence
        for layer in self.layers:
            x = layer(x, enc_output, target_mask, encoder_mask)
        
        # Apply final normalization
        return self.norm(x)


class Transformer(nn.Module):
    """
    Complete transformer model with encoder and decoder.
    """
    def __init__(self, config):
        super().__init__()
        # Extract relevant config parameters
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.src_vocab_size = getattr(config, 'src_vocab_size', 30000)
        self.tgt_vocab_size = getattr(config, 'tgt_vocab_size', 30000)
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        self.max_len = getattr(config, 'max_position_embeddings', 512)
        
        # Embeddings
        # Note: You can import and use the TransformerEmbeddings class from embeddings.py
        from .embeddings import TransformerEmbeddings
        self.encoder_embedding = TransformerEmbeddings(config)
        
        # If target vocabulary is different, you might want separate embedding parameters
        # self.decoder_embedding = TransformerEmbeddings(config) # Uncomment if needed
        # For simplicity, we'll use the same embeddings for now
        self.decoder_embedding = self.encoder_embedding
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        # Final output projection for generating token probabilities
        self.output_projection = nn.Linear(self.hidden_dim, self.tgt_vocab_size)
        
    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        """
        Forward pass of the transformer.
        
        Args:
            src_tokens: Source token indices of shape [batch_size, src_seq_len]
            tgt_tokens: Target token indices of shape [batch_size, tgt_seq_len]
            src_mask: Optional mask for padding in source sequences
            tgt_mask: Mask for preventing attention to future tokens in target sequence
            
        Returns:
            Output tensor with token probabilities of shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Encoder
        encoder_input = self.encoder_embedding(src_tokens)
        encoder_output = self.encoder(encoder_input, src_mask)
        
        # Decoder
        decoder_input = self.decoder_embedding(tgt_tokens)
        decoder_output = self.decoder(decoder_input, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary size
        output = self.output_projection(decoder_output)
        
        return output
        
    def generate_square_subsequent_mask(self, size):
        """
        Generate a mask for preventing attention to future tokens (for decoding).
        
        Args:
            size: Size of the square mask
            
        Returns:
            Mask tensor of shape [size, size] with ones in the lower triangle and zeros elsewhere
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0  # Convert to boolean where True (1) means attend, False (0) means mask 