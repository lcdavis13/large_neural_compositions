import torch
import torch.nn as nn
import model_encoders as encoders
import model_skipgates as skips
from ode_solver import odeint


class ODEFunc_Replicator_CustomFitness(nn.Module):
    def __init__(self, fitness_fn):
        super().__init__()

        self.gate = skips.ZeroGate()
        self.fn = fitness_fn
    
    def forward(self, t, x):
        # eval fitness function
        fitness = self.fn(x)

        # Replicator dynamics
        xT_fx = torch.sum(x * fitness, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fitness - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        # zero gate to start with zero derivative (identity function) before training
        dxdt = self.gate(dxdt)  # B x N
        
        return dxdt


class ODEFunc_Replicator_CustomFitness_IdEmbed_XEncode(nn.Module):
    def __init__(self, fitness_fn, embed_dim):
        super().__init__()
        
        self.encode = encoders.AbundanceEncoder_LearnedFourier(embed_dim)

        self.fn = fitness_fn

        self.decode = encoders.Decoder(embed_dim)
        self.gate = skips.ZeroGate()
    
    def forward(self, t, x, embeddings):
        # Preprocessing: encode abundances, add to embeddings
        h = self.encode(x) + embeddings  # B x N x embed_dim
        
        # eval and decode fitness function
        h = self.fn(h)
        fitness = self.decode(h)
        
        # Replicator dynamics
        xT_fx = torch.sum(x * fitness, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fitness - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        # zero gate to start with zero derivative (identity function) before training
        dxdt = self.gate(dxdt)  # B x N
        
        return dxdt


class ODEFunc_Replicator_CustomFitness_IdEmbed(nn.Module):
    def __init__(self, fitness_fn):
        super().__init__()

        self.fn = fitness_fn

        self.gate = skips.ZeroGate()
    
    def forward(self, t, x, embeddings):
        print(x.shape)
        print(embeddings.shape)
        print(t.shape)

        # eval fitness function, passing embeddings for custom handling
        fitness = self.fn(x, embeddings)
        
        # Replicator dynamics
        print(x.shape)
        print(fitness.shape)
        xT_fx = torch.sum(x * fitness, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fitness - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        # zero gate to start with zero derivative (identity function) before training
        dxdt = self.gate(dxdt)  # B x N
        
        return dxdt


class Replicator_CustomFitness(nn.Module):
    """
    Replicator dynamics with a custom fitness function.
    """
    def __init__(self, fitness_fn):
        super().__init__()

        self.USES_ODEINT = True

        self.ode_func = ODEFunc_Replicator_CustomFitness(fitness_fn)
    
    def forward(self, t, x):
        y = odeint(self.ode_func, x, t)
        return y
    

class Replicator_CustomFitness_IdEmbed_XEncode(nn.Module):
    """
    Replicator dynamics with a custom fitness function, with id embeddings added to encoded abundances.
    "Fitness" function is expected to return the same shape as input, and will be linearly decoded to produce fitnesses.
    An optional enrichment function can be applied to the embeddings before passing them to the fitness function (e.g., a transformer to encode possible OTU interactions before abundance encodings are added).
    """
    def __init__(self, fitness_fn, data_dim, embed_dim, enrich_fn=None):
        super().__init__()

        self.USES_ODEINT = True
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)
        self.enrich_fn = enrich_fn

        self.ode_func = ODEFunc_Replicator_CustomFitness_IdEmbed_XEncode(fitness_fn, embed_dim)
    
    def forward(self, t, x, ids):
        # preprocess embeddings
        embeddings = self.embed(ids)
        if self.enrich_fn is not None:
            embeddings = self.enrich_fn(embeddings)

        # ODE
        y = odeint(lambda t,x: self.ode_func(t,x,embeddings), x, t)
        
        return y
    

class Replicator_CustomFitness_IdEmbed(nn.Module):
    """
    Replicator dynamics with a custom fitness function, with id embeddings passed directly to the fitness function alongside raw abundances.
    Unlike other model wrappers, the fitness function is expected to return the final fitnesses directly to allow custom architectures to produce the fitness.
    An optional enrichment function can be applied to the embeddings before passing them to the fitness function (e.g., a transformer to encode possible OTU interactions before abundance encodings are added).
    """
    def __init__(self, fitness_fn, data_dim, embed_dim, enrich_fn=None):
        super().__init__()

        self.USES_ODEINT = True
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)
        self.enrich_fn = enrich_fn

        self.ode_func = ODEFunc_Replicator_CustomFitness_IdEmbed(fitness_fn)
    
    def forward(self, t, x, ids):
        # preprocess embeddings
        embeddings = self.embed(ids)
        if self.enrich_fn is not None:
            embeddings = self.enrich_fn(embeddings)

        # ODE
        y = odeint(lambda t,x: self.ode_func(t,x,embeddings), x, t)
        
        return y
    
