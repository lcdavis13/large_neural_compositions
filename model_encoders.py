import torch
import torch.nn as nn
import math
    

class IdEmbedder(nn.Module):
    def __init__(self, data_dim, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(data_dim+1, embed_dim)

    def forward(self, ids):
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        return self.embed(ids)
    

class AbundanceEncoder_LearnedFourier(nn.Module):
    def __init__(self, encode_dim: int, num_frequencies: int = None):
        """
        encode_dim: Output encoding dimension
        num_frequencies: Number of Fourier feature pairs to extract. Defaults to encode_dim // 2
        Initializes with exponentially spaced frequencies in [1, e] scaled by 2π for richer harmonic coverage
        """
        super().__init__()
        self.encode_dim = encode_dim
        self.num_frequencies = num_frequencies if num_frequencies is not None else encode_dim // 2

        # Initialize W_r ∈ R^{num_frequencies × 1} with exponentially increasing frequencies
        initial_frequencies = torch.arange(self.num_frequencies).unsqueeze(-1)  # Shape: [num_frequencies, 1]
        initial_frequencies = torch.exp(initial_frequencies) * (2 * math.pi) # exponentiate and scale to 2pi

         # Shape: [num_frequencies, 1]
        self.freq = nn.Parameter(initial_frequencies)  # Learnable frequencies

        # projection matrix
        self.proj = nn.Linear(2 * self.num_frequencies, encode_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [*, 1] or [*] where each element is a relative abundance value.
        Returns: Tensor of shape [*, encode_dim]
        """
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.size(-1) != 1:
            x = x.unsqueeze(-1)  # Ensure shape [* 1]

        # Compute angles: [*, num_frequencies]
        angles = x @ self.freq.T  # [* 1] @ [1, num_frequencies] -> [* num_frequencies]

        # Compute sine and cosine features
        sin_feats = torch.sin(angles)
        cos_feats = torch.cos(angles)

        # Concatenate to form full feature vector: [*, 2 * num_frequencies]
        fourier_feats = torch.cat([cos_feats, sin_feats], dim=-1)

        # Normalize by number of frequencies
        normalized_feats = fourier_feats / math.sqrt(2 * self.num_frequencies)

        # Linear projection: [*, d]
        return self.proj(normalized_feats)
    

class Decoder(nn.Module):
    def __init__(self, encode_dim: int):
        """
        encode_dim: Input encoding dimension
        Just a linear layer to decode the abundance features back to a single value (either abundance or fitness)
        """
        super().__init__()
        self.decode = nn.Linear(encode_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [*, encode_dim]
        Returns: Tensor of shape [*]
        """
        return self.decode(x).squeeze(-1)