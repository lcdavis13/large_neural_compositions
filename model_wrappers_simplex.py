import torch
import torch.nn as nn
import model_encoders as encoders
from model_maskedSoftmax import MaskedSoftmax
from model_normedLog import normed_log
import model_skipgates as skips


class SimplexModel(nn.Module):
    """
    Explicit composition model. 
    Output of core model is added to log of input (residual/skip in log space) 
    Then it is converted to simplex (and exponentiated) with masked softmax.
    """
    def __init__(self, core_model, learnable_skip):
        super().__init__()

        self.core_model = core_model

        if learnable_skip:
            self.skip_gate = skips.GateSkip()
        else:
            self.skip_gate = skips.StaticSkip()
            
        self.masked_softmax = MaskedSoftmax()
        
    def forward(self, x):
        logx = normed_log(x)
        h = self.core_model(x)
        h = self.skip_gate(h, logx)
        h = self.masked_softmax(h, x)
        return h
    
    
class SimplexModel_IdEmbed(nn.Module):
    """
    Explicit composition model with preprocessing to embed IDs for the input, and linearly decode output to an abundance vector.
    Core model is expected to return the same shape as input (i.e. enriched embeddings).
    
    After decoding, this applies a masked softmax which preserves the zero pattern of the input.
    """
    def __init__(self, core_model, data_dim, embed_dim, learnable_skip):
        super().__init__()
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)

        self.core_model = core_model

        if learnable_skip:
            self.skip_gate = skips.GateSkip()
        else:
            self.skip_gate = skips.StaticSkip()

        self.decode = encoders.Decoder(embed_dim)
        self.masked_softmax = MaskedSoftmax()
        

    def forward(self, x, ids):
        # preprocessing
        h = self.embed(ids)
        logx = normed_log(x)

        # core model
        h = self.core_model(h)

        # postprocessing
        h = self.decode(h)
        h = self.skip_gate(h, logx)
        h = self.masked_softmax(h, x)

        return h
    

    
class SimplexModel_IdEmbed_NoDecode(nn.Module):
    """
    Explicit composition model with preprocessing to embed IDs for the input. 
    Core model must return the final solution - wrapper applies no decoding, softmax, etc.
    """
    def __init__(self, core_model, data_dim, embed_dim):
        super().__init__()
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)

        self.core_model = core_model
        

    def forward(self, x, ids):
        # preprocessing
        h = self.embed(ids)

        # core model
        h = self.core_model(x, h)

        return h
    
