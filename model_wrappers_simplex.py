import torch
import torch.nn as nn
import model_encoders as encoders
from model_maskedSoftmax import MaskedSoftmax


class SimplexModel(nn.Module):
    """
    Explicit composition model. Just wraps the core model in a masked softmax.
    """
    def __init__(self, core_model):
        super().__init__()

        self.core_model = core_model
        self.masked_softmax = MaskedSoftmax()
        
    def forward(self, x):
        h = self.core_model(x)
        h = self.masked_softmax(h, x)
        return h
    
    
class SimplexModel_IdEmbed(nn.Module):
    """
    Explicit composition model with preprocessing to embed IDs for the input, and linearly decode output to an abundance vector.
    Core model is expected to return the same shape as input (i.e. enriched embeddings).
    After decoding, this applies a softmax and a learned "blend" skip gate which interpolates between input and output.
    The gate is initialized such that the model starts as an identity function.
    """
    def __init__(self, core_model, data_dim, embed_dim):
        super().__init__()
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)

        self.core_model = core_model

        self.decode = encoders.Decoder(embed_dim)
        self.masked_softmax = MaskedSoftmax()
        

    def forward(self, x, ids):
        # preprocessing
        h = self.embed(ids)

        # core model
        h = self.core_model(h)

        # postprocessing
        h = self.decode(h)
        h = self.masked_softmax(h, x)

        return h
    
