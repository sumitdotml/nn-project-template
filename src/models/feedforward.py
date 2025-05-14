"""
Implements the feedforward network (FFN) typically found in Transformer models.
"""
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # These would typically come from the config object
        # e.g., d_model = config.hidden_dim
        #       d_ff = config.ff_dim or 4 * config.hidden_dim

        d_model = getattr(config, 'hidden_dim', 512)
        d_ff = getattr(config, 'ff_dim', 4 * d_model) # Common expansion factor
        dropout = getattr(config, 'dropout', 0.1)
        activation_function_name = getattr(config, 'activation_function', 'relu')

        if activation_function_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_function_name.lower() == 'gelu':
            activation = nn.GELU()
        # Add more activation functions as needed (e.g., swiglu, geglu)
        else:
            # Fallback or error for unknown activation
            print(f"Warning: Unknown activation function '{activation_function_name}' for FeedForward. Using ReLU.")
            activation = nn.ReLU()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout_1 = nn.Dropout(dropout) # Dropout can be applied after activation
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout) # Or dropout before the final output

    def forward(self, x):
        # Common implementation: x -> linear -> activation -> dropout -> linear -> dropout
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.w_2(x)
        x = self.dropout_2(x)
        return x 