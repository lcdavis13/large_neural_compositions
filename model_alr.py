import torch


def hofbauer_state_to_alr(x_ext: torch.Tensor) -> torch.Tensor:
    """
    x_ext: B x (N+1) Hofbauer composition.

    returns: z: B x N, ALR coords.
             For inactive species (mask_main == 0) we return z_i = 0 by convention.
        mask_main: B x N, 1 for active species, 0 for species that must stay exactly zero.
    """
    # Extract the zero vs nonzero pattern to a new tensor of 0 and 1
    zero_mask = (x_ext[..., :-1] > 0).to(x_ext.dtype)  # B x N


    x_ref = x_ext[..., -1:]          # B x 1 (assumed > 0)
    x_main = x_ext[..., :-1]         # B x N

    z = torch.zeros_like(x_main)

    # Only compute log(x_i / x_ref) where the species is truly active (mask == 1)
    active = zero_mask.bool()

    # broadcast x_ref to same shape as x_main for indexing
    x_ref_broadcast = x_ref.expand_as(x_main)

    z[active] = torch.log(x_main[active] / x_ref_broadcast[active])

    # For inactive species, z_i is left at 0. They are never used by the inverse.
    return z, zero_mask


def alr_to_hofbauer_state(z: torch.Tensor,
                                 zero_mask: torch.Tensor) -> torch.Tensor:
    """
    z: B x N, ALR coords for main species. Entries where zero_mask == 0 are ignored.
    zero_mask: B x N, 1 for active species, 0 for species that must stay exactly zero.

    returns: B x (N+1) Hofbauer composition (sum = 1, >= 0),
             with x_i = 0 wherever zero_mask == 0, and biomass as ref species.
    """
    # Exponentiate only active coordinates
    exp_z = torch.exp(z) * zero_mask  # inactive dims -> 0

    # Sum over active main species
    s = exp_z.sum(dim=-1, keepdim=True)  # B x 1

    # Biomass/reference species: x_ref = 1 / (1 + sum_i exp(z_i))
    x_ref = 1.0 / (1.0 + s)              # B x 1

    # Main species: x_i = exp(z_i) * x_ref for active ones, 0 for inactive ones
    x_main = exp_z * x_ref               # B x N, inactive remain 0

    x_ext = torch.cat([x_main, x_ref], dim=-1)  # B x (N+1)

    # This is an exact simplex point with zeros exactly at masked-out dims.
    return x_ext
