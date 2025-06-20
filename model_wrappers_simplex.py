import torch
import torch.nn as nn
import model_encoders as encoders
import model_skipgates as skips
from model_maskedSoftmax import MaskedSoftmax


class ResidualSimplexModel(nn.Module):
    """
    Explicit composition model. Wraps the core model in a softmax and a learned "blend" skip gate which interpolates between input and output.
    The gate is initialized such that the model starts as an identity function.
    """
    def __init__(self, core_model):
        super().__init__()

        self.core_model = core_model
        self.masked_softmax = MaskedSoftmax()
        self.blendskip = skips.BlendSkip()

    def forward(self, x):
        h = self.core_model(x)
        h = self.masked_softmax(h, x)
        h = self.blendskip(h, x)
        return h
    
    
class ResidualSimplexModel_IdEmbed(nn.Module):
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
        self.blendskip = skips.BlendSkip()
        

    def forward(self, x, ids):
        # preprocessing
        h = self.embed(ids)

        # core model
        h = self.core_model(h)

        # postprocessing
        h = self.decode(h)
        h = self.masked_softmax(h, x)
        h = self.blendskip(h, x)

        return h
    
    
# class ResidualSimplexModel_IdEmbed_XEncode(nn.Module):
#     """
#     Explicit composition model with an ID embedder and a learned Fourier abundance encoder.
#     The utility of the abundance encoder for explicit models is questionable, because the abundance is equal for all OTUs in the input. The only information it would actually encode is: how many OTUs are present. And adding that to every embedding seems more likely to cause interference than any benefit.
#     """"
#     def __init__(self, core_model, data_dim, embed_dim):
#         super().__init__()
#         self.USES_CONDENSED = True

#         self.embed = encoders.IdEmbedder(data_dim, embed_dim)
#         self.encode = encoders.AbundanceEncoder_LearnedFourier(embed_dim)

#         self.core_model = core_model

#         self.decode = encoders.Decoder(embed_dim)
#         self.softmax = MaskedSoftmax()
#         self.blendskip = skips.BlendSkip()
        

#     def forward(self, x, ids):
#         # preprocessing
#         h = self.embed(ids) + self.encode(x)

#         # core model
#         h = self.core_model(h)

#         # postprocessing
#         h = self.decode(h)
#         h = self.softmax(h, x)
#         h = self.blendskip(h, x)

#         return h
    
