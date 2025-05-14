"""
Implements normalization layers, primarily Layer Normalization,
configured via the global config object.
"""
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    A Layer Normalization module that uses parameters from the project's
    configuration object (e.g., hidden_dim for normalized_shape, layer_norm_eps for epsilon).
    It wraps torch.nn.LayerNorm.
    """
    def __init__(self, config, normalized_shape=None):
        super().__init__()
        
        if normalized_shape is None:
            # Default to hidden_dim if not specified
            if not hasattr(config, 'hidden_dim'):
                raise ValueError(
                    "LayerNorm requires 'hidden_dim' in config if normalized_shape is not explicitly provided."
                )
            normalized_shape = config.hidden_dim
        
        # Use a common epsilon value for transformers (e.g., from BERT), allow override from config
        eps = getattr(config, 'layer_norm_eps', 1e-12)

        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.norm(x)

# Note: You could also create a custom LayerNorm implementation here if needed for specific research purposes,
# but using nn.LayerNorm is generally recommended for performance and stability.
# Example custom LayerNorm (for illustration - usually use nn.LayerNorm):
# class CustomLayerNorm(nn.Module):
#     def __init__(self, features, eps=1e-6):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(features))  # learnable scale
#         self.beta = nn.Parameter(torch.zeros(features)) # learnable shift
#         self.eps = eps
# 
#     def forward(self, x):
#         # x shape: (batch_size, ..., features)
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.gamma * (x - mean) / (std + self.eps) + self.beta 