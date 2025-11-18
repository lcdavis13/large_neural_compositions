import torch

"""
Helper functions for performing a Hofbauer lift on compositional data.
The Hofbauer lift allows a replicator system to have equal expressivity (and orbit equivalence) to the generalized Lotka-Volterra (gLV) system.
It involves adding an extra species dimension that acts as a proxy for the total biomass. The interactions from the biomass proxy onto the other species fulfills the role of a intrinsic reproductive rate in the gLV dynamics.
Note that the resulting system is orbit-equivalent but not strictly equivalent. That is, the effective timescale will be warped as a function of biomass.
Also note that it is critical for the biomass term to ALWAYS have zero fitness, regardless of what your fitness function predicts; use _mask_biomass_fitness to ensure this.
"""


def hofbauer_augment_state(x, biomass_init: float = 1.0):
    """
    Add an extra 'biomass proxy' species and renormalize onto the simplex.

    Given x of shape (..., N) with sum_i x_i = 1, we construct a state p in Δ^{N+1}
    corresponding to initial biomass B = biomass_init via:

        p_i = x_i * B / (1 + B)     (i = 1..N)
        p_{N+1} = 1 / (1 + B)

    For biomass_init = 1, this is p_i = x_i / 2 and p_{N+1} = 1/2.
    """
    factor_species = biomass_init / (1.0 + biomass_init)
    biomass_comp = x.new_full(x[..., :1].shape, 1.0 / (1.0 + biomass_init))
    x_species = x * factor_species
    p = torch.cat([x_species, biomass_comp], dim=-1)
    return p


def hofbauer_collapse_state(p, eps: float = 1e-8):
    """
    After integration, drop the biomass dimension and renormalize
    back to an N-dimensional composition.

    From Hofbauer coordinates:

        x_i = p_i / (1 - p_{N+1})

    where p_{N+1} is the biomass proxy coordinate.
    """
    species = p[..., :-1]
    biomass_comp = p[..., -1:]
    denom = 1.0 - biomass_comp
    denom = denom.clamp_min(eps)
    x = species / denom
    return x


def mask_biomass_fitness(fitness):
    """
    Mask the fitness of the biomass proxy species to always be zero.

    Assumes fitness has shape (..., N+1) and the last dimension corresponds to
    the biomass species.
    """
    if fitness.size(-1) < 1:
        return fitness
    fitness = fitness.clone()
    fitness[..., -1] = 0.0
    return fitness