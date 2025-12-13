import torch
import torch.nn as nn
import model_encoders as encoders
import model_skipgates as skips
from ode_solver import odeint
import model_hofbauer as hof
import model_alr as alr


# Because lambda functions (or any non-module) don't work with torchdiffeq.odeint_adjoint
# in the current version, we define a "lambda module" for passing extra arguments to the
# ODE function.
class ODELambda(nn.Module):
    def __init__(self, base_func):
        super().__init__()
        self.base_func = base_func      # e.g. your ODEFunc_...
        self._extra_args = ()
        self._extra_kwargs = {}

    def set_context(self, *args, **kwargs):
        """
        Store any extra arguments/kwargs needed for this forward pass.
        These will be passed to base_func(t, x, *args, **kwargs).
        """
        self._extra_args = args
        self._extra_kwargs = kwargs

    def forward(self, t, x):
        # t, x come from odeint; everything else comes from stored context
        return self.base_func(t, x, *self._extra_args, **self._extra_kwargs)


# -------------------------------------------------------------------------
# ODE function classes
# -------------------------------------------------------------------------

class ODEFunc_Replicator_CustomFitness(nn.Module):
    def __init__(self, fitness_fn, learnable_skip, use_hofbauer: bool, use_lnorm: bool = False):
        super().__init__()

        if learnable_skip:
            self.gate = skips.ZeroGate()
        else:
            self.gate = nn.Identity()

        # if use_lnorm:
        #     self.lnorm = nn.LayerNorm((data_dim))
        # else:
        #     self.lnorm = nn.Identity()


        self.fn = fitness_fn
        self.use_hofbauer = use_hofbauer

    def forward(self, t, x):
        # eval fitness function
        fitness = self.fn(x)

        # Hofbauer masking: biomass species has fitness 0
        if self.use_hofbauer:
            fitness = hof.mask_biomass_fitness(fitness)

        # Replicator dynamics
        xT_fx = torch.sum(x * fitness, dim=-1, keepdim=True)  # B x 1
        diff = fitness - xT_fx                                # B x N (or N+1)
        dxdt = x * diff                                       # B x N (or N+1)

        # zero gate to start with zero derivative (identity function) before training
        dxdt = self.gate(dxdt)  # B x N (or N+1)

        return dxdt


class ODEFunc_Replicator_CustomFitness_Energy(nn.Module):
    def __init__(self, fitness_fn, fitness_dim, learnable_skip):
        super().__init__()

        if learnable_skip:
            self.gate = skips.ZeroGate()
        else:
            self.gate = nn.Identity()

        self.fn = fitness_fn
        
        self.pot_fn = nn.Linear(fitness_dim, 1, bias=False)

    def forward(self, t, x):
        mask = (x > 0)  # or pass in an explicit mask from outside
        mask = mask.to(x.dtype)

        with torch.enable_grad():
            # Ensure x requires grad (needed for autograd to compute gradient)
            x = x.clone().detach().requires_grad_(True) 
            x = x * mask  # non-op, but it enforces zero fitness off-support in our computed gradients

            # Compute scalar potential Φ(x): shape (B,) or (B,1)
            fitness = self.fn(x)
            potential = self.pot_fn(fitness)

            # Sum over our potentials to get a single scalar for batch (autograd will separate it back out)
            potential_scalar = potential.sum()

            # Compute gradient dΦ/dx
            (fitness,) = torch.autograd.grad(
                outputs=potential_scalar,
                inputs=x,
                create_graph=True,
            )
            
        fitness = -fitness

        # Replicator dynamics
        xT_fx = torch.sum(x * fitness, dim=-1, keepdim=True)  # B x 1
        diff = fitness - xT_fx                                # B x N (or N+1)
        dxdt = x * diff                                       # B x N (or N+1)

        # zero gate to start with zero derivative (identity function) before training
        dxdt = self.gate(dxdt)  # B x N (or N+1)

        return dxdt
    

class ODEFunc_Replicator_CustomFitness_IdEmbed_XEncode(nn.Module):
    def __init__(self, fitness_fn, embed_dim, use_logx, learnable_skip,
                 use_hofbauer: bool):
        super().__init__()

        self.encode = encoders.AbundanceEncoder_LearnedFourier(embed_dim, use_logx)
        self.fn = fitness_fn
        self.decode = encoders.Decoder(embed_dim)

        if learnable_skip:
            self.gate = skips.ZeroGate()
        else:
            self.gate = nn.Identity()

        self.use_hofbauer = use_hofbauer

    def forward(self, t, x, embeddings):
        # Preprocessing: encode abundances, add to embeddings
        h = self.encode(x) + embeddings  # B x N x embed_dim (N or N+1)

        # eval and decode fitness function
        h = self.fn(h)
        fitness = self.decode(h)  # B x N (or N+1)

        # Hofbauer masking: biomass species has fitness 0
        if self.use_hofbauer:
            fitness = hof.mask_biomass_fitness(fitness)

        # Replicator dynamics
        xT_fx = torch.sum(x * fitness, dim=-1, keepdim=True)  # B x 1
        diff = fitness - xT_fx                                # B x N (or N+1)
        dxdt = x * diff                                       # B x N (or N+1)

        # zero gate to start with zero derivative (identity function) before training
        dxdt = self.gate(dxdt)

        return dxdt


class ODEFunc_Replicator_CustomFitness_IdEmbed(nn.Module):
    def __init__(self, fitness_fn, learnable_skip, use_hofbauer: bool):
        super().__init__()

        self.fn = fitness_fn

        if learnable_skip:
            self.gate = skips.ZeroGate()
        else:
            self.gate = nn.Identity()

        self.use_hofbauer = use_hofbauer

    def forward(self, t, x, embeddings):
        # eval fitness function, passing embeddings for custom handling
        fitness = self.fn(x, embeddings)  # B x N (or N+1)

        # Hofbauer masking: biomass species has fitness 0
        if self.use_hofbauer:
            fitness = hof.mask_biomass_fitness(fitness)

        # Replicator dynamics
        xT_fx = torch.sum(x * fitness, dim=-1, keepdim=True)  # B x 1
        diff = fitness - xT_fx                                # B x N (or N+1)
        dxdt = x * diff                                       # B x N (or N+1)

        # zero gate to start with zero derivative (identity function) before training
        dxdt = self.gate(dxdt)

        return dxdt


# -------------------------------------------------------------------------
# Wrapper models
# -------------------------------------------------------------------------

class Replicator_CustomFitness(nn.Module):
    """
    Replicator dynamics with a custom fitness function.

    The fitness function is built inside this wrapper from a constructor
    so we can choose N vs (N+1) depending on Hofbauer mode.
    """
    def __init__(self, fitness_fn_ctor, data_dim, learnable_skip,
                 use_hofbauer: bool, energy_based: bool = False, use_lnorm: bool = False):
        """
        Parameters
        ----------
        fitness_fn_ctor : callable
            A constructor or lambda that takes `num_species` and returns
            an nn.Module fitness function. Example:
                lambda dim: core.Linear(dim)
        data_dim : int
            Number of original species (N).
        learnable_skip : bool
            As before.
        use_hofbauer : bool
            If True, construct the fitness function for N+1 dimensions
            and run in Hofbauer space internally.
        """
        super().__init__()

        self.USES_ODEINT = True
        self.use_hofbauer = use_hofbauer
        self.data_dim = data_dim

        # Choose the number of species the fitness model should see
        num_species = data_dim + 1 if use_hofbauer else data_dim

        # Build the fitness model from the constructor
        self.fitness_model = fitness_fn_ctor(num_species)

        # ODE function uses that fitness model
        if energy_based:
            self.ode_func = ODEFunc_Replicator_CustomFitness_Energy(
                fitness_fn=self.fitness_model,
                fitness_dim=num_species,
                learnable_skip=learnable_skip,
                use_lnorm=use_lnorm,
            )
        else:
            self.ode_func = ODEFunc_Replicator_CustomFitness(
                fitness_fn=self.fitness_model,
                learnable_skip=learnable_skip,
                use_hofbauer=use_hofbauer,
                use_lnorm=use_lnorm,
            )

    def forward(self, t, x):
        """
        x: B x N composition (sum_i x_i = 1)
        returns: T x B x N composition
        """
        if self.use_hofbauer:
            # Augment with biomass proxy species
            x0 = hof.hofbauer_augment_state(x, biomass_init=1.0)  # B x (N+1)

            # Integrate in Hofbauer space
            y_ext = odeint(self.ode_func, x0, t)  # T x B x (N+1)

            # Collapse back to N-dimensional composition
            y = hof.hofbauer_collapse_state(y_ext)  # T x B x N
        else:
            y = odeint(self.ode_func, x, t)

        return y


class Replicator_CustomFitness_IdEmbed_XEncode(nn.Module):
    """
    Replicator dynamics with a custom fitness function, with id embeddings
    added to encoded abundances.

    The "fitness" function is expected to return the same shape as input, and
    will be linearly decoded to produce fitnesses.

    Optional Hofbauer formulation (use_hofbauer=True) adds a biomass proxy
    species internally.
    """
    def __init__(
        self,
        core_fitness_fn,
        data_dim,
        embed_dim,
        learnable_skip,
        use_logx,
        use_hofbauer: bool,
        enrich_fn=None,
    ):
        super().__init__()

        self.USES_ODEINT = True
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)
        self.enrich_fn = enrich_fn
        self.use_hofbauer = use_hofbauer

        self.ode_func_raw = ODEFunc_Replicator_CustomFitness_IdEmbed_XEncode(
            fitness_fn=core_fitness_fn,
            embed_dim=embed_dim,
            use_logx=use_logx,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

        # Persistent lambda-module wrapper
        self.ode_func = ODELambda(self.ode_func_raw)

    def forward(self, t, x, ids):
        """
        x: B x N composition
        ids: B x N (indices fed to IdEmbedder)
        returns: T x B x N composition
        """
        # preprocess embeddings
        embeddings = self.embed(ids)  # B x N x embed_dim
        if self.enrich_fn is not None:
            embeddings = self.enrich_fn(embeddings)

        if self.use_hofbauer:
            # Step 1: augment with biomass proxy species
            x0 = hof.hofbauer_augment_state(x, biomass_init=1.0)  # B x (N+1)

            # biomass embedding: simple choice is a zero vector; could be learned instead
            biomass_embed = embeddings.new_zeros(embeddings[..., :1, :].shape)  # B x 1 x embed_dim
            embeddings_ext = torch.cat([embeddings, biomass_embed], dim=1)      # B x (N+1) x embed_dim

            # ODE integration in Hofbauer space
            self.ode_func.set_context(embeddings_ext)
            y_ext = odeint(
                self.ode_func,
                x0,
                t,
            )

            # Step 4: collapse back to N-dimensional composition
            y = hof.hofbauer_collapse_state(y_ext)
        else:
            # ODE in standard N-dimensional replicator space
            self.ode_func.set_context(embeddings)
            y = odeint(
                self.ode_func,
                x,
                t,
            )

        return y


class Replicator_CustomFitness_IdEmbed(nn.Module):
    """
    Replicator dynamics with a custom fitness function, with id embeddings
    passed directly to the fitness function alongside raw abundances.

    The fitness function is expected to return the final fitnesses directly.

    Optional Hofbauer formulation (use_hofbauer=True) adds a biomass proxy
    species internally.
    """
    def __init__(
        self,
        fitness_fn,
        data_dim,
        embed_dim,
        learnable_skip,
        use_hofbauer: bool,
        enrich_fn=None,
    ):
        super().__init__()

        self.USES_ODEINT = True
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)
        self.enrich_fn = enrich_fn
        self.use_hofbauer = use_hofbauer

        self.ode_func_raw = ODEFunc_Replicator_CustomFitness_IdEmbed(
            fitness_fn=fitness_fn,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

        # Persistent lambda-module wrapper
        self.ode_func = ODELambda(self.ode_func_raw)

    def forward(self, t, x, ids):
        """
        x: B x N composition
        ids: B x N
        returns: T x B x N composition
        """
        # preprocess embeddings
        embeddings = self.embed(ids)  # B x N x embed_dim
        if self.enrich_fn is not None:
            embeddings = self.enrich_fn(embeddings)

        if self.use_hofbauer:
            # Step 1: augment x
            x0 = hof.hofbauer_augment_state(x, biomass_init=1.0)  # B x (N+1)

            # biomass embedding: zero vector (could be made learnable if desired)
            biomass_embed = embeddings.new_zeros(embeddings[..., :1, :].shape)  # B x 1 x embed_dim
            embeddings_ext = torch.cat([embeddings, biomass_embed], dim=1)      # B x (N+1) x embed_dim

            # Integrate in Hofbauer space
            self.ode_func.set_context(embeddings_ext)
            y_ext = odeint(
                self.ode_func,
                x0,
                t,
            )

            # Step 4: collapse back to N-dimensional composition
            y = hof.hofbauer_collapse_state(y_ext)
        else:
            # Standard N-dimensional space
            self.ode_func.set_context(embeddings)
            y = odeint(
                self.ode_func,
                x,
                t,
            )

        return y


# -------------------------------------------------------------------------
# ALR + Hofbauer ODE function classes
# -------------------------------------------------------------------------

class ODEFunc_ALRReplicator_CustomFitness(nn.Module):
    """
    Replicator dynamics in ALR coordinates, with Hofbauer lifting.

    State is z (ALR): z_i = log(x_i / x_biomass), i = 0..N-1,
    where the biomass (Hofbauer extra dim) is the reference component.
    """
    def __init__(self, fitness_fn, learnable_skip: bool):
        super().__init__()

        if learnable_skip:
            self.gate = skips.ZeroGate()
        else:
            self.gate = nn.Identity()

        self.fn = fitness_fn  # expects x_ext: B x (N+1)

    def forward(self, t, z, zero_mask):
        """
        z: B x N (ALR coords)
        returns: dz/dt in ALR coords, B x N
        """
        # Convert ALR state to Hofbauer composition
        x = alr.alr_to_hofbauer_state(z, zero_mask)  # B x (N+1)

        # eval fitness function
        fitness = self.fn(x)
        
        # Replicator in ALR:
        # dz_i/dt = f_i(x) - f_ref(x) = f_i(x) - 0, where ref is biomass = last component
        dzdt = fitness[..., :-1]       # B x N
        # enforce zero off-support
        dzdt = dzdt * zero_mask

        # zero gate to start with zero derivative (identity) before training
        dzdt = self.gate(dzdt)

        return dzdt


class ODEFunc_ALRReplicator_CustomFitness_IdEmbed_XEncode(nn.Module):
    """
    ALR replicator with id embeddings + abundance encoding, in Hofbauer space.

    State is z (ALR). We:
    - convert z -> x_ext (Hofbauer),
    - encode x_ext, add embeddings,
    - run core fitness fn + decoder to get fitness on x_ext,
    - apply Hofbauer mask implicitly via ALR reference,
    - evolve z via dz_i = f_i - f_ref.
    """
    def __init__(self, fitness_fn, embed_dim, use_logx, learnable_skip: bool):
        super().__init__()

        self.encode = encoders.AbundanceEncoder_LearnedFourier(embed_dim, use_logx)
        self.fn = fitness_fn            # core fitness network, expects encoded+emb
        self.decode = encoders.Decoder(embed_dim)

        if learnable_skip:
            self.gate = skips.ZeroGate()
        else:
            self.gate = nn.Identity()

    def forward(self, t, z, embeddings, zero_mask):
        """
        z: B x N (ALR coords)
        embeddings: B x (N+1) x embed_dim (including biomass entry)
        """
        # Convert ALR state to Hofbauer composition
        x_ext = alr.alr_to_hofbauer_state(z, zero_mask)  # B x (N+1)

        # Preprocessing: encode abundances, add to embeddings
        h = self.encode(x_ext) + embeddings  # B x (N+1) x embed_dim

        # eval and decode fitness function
        h = self.fn(h)
        fitness = self.decode(h)  # B x (N+1)

        # ALR replicator dynamics
        dzdt = fitness[..., :-1]    # B x N
        # enforce zero off-support
        dzdt = dzdt * zero_mask

        # zero gate
        dzdt = self.gate(dzdt)

        return dzdt


class ODEFunc_ALRReplicator_CustomFitness_IdEmbed(nn.Module):
    """
    ALR replicator with id embeddings passed directly to fitness function.

    State is z (ALR). Fitness fn sees full Hofbauer composition + embeddings
    and returns fitness on each species (including biomass).
    """
    def __init__(self, fitness_fn, learnable_skip: bool):
        super().__init__()

        self.fn = fitness_fn  # expects (x_ext, embeddings) -> fitness B x (N+1)

        if learnable_skip:
            self.gate = skips.ZeroGate()
        else:
            self.gate = nn.Identity()

    def forward(self, t, z, embeddings, zero_mask):
        """
        z: B x N (ALR coords)
        embeddings: B x (N+1) x embed_dim (including biomass entry)
        """
        # Convert ALR state to Hofbauer composition
        x_ext = alr.alr_to_hofbauer_state(z, zero_mask)  # B x (N+1)

        # eval fitness function, passing embeddings
        fitness = self.fn(x_ext, embeddings)  # B x (N+1)

        # ALR replicator dynamics
        dzdt = fitness[..., :-1]     # B x N
        # enforce zero off-support
        dzdt = dzdt * zero_mask

        # zero gate
        dzdt = self.gate(dzdt)

        return dzdt


# -------------------------------------------------------------------------
# ALR + Hofbauer wrappers
# -------------------------------------------------------------------------

class ALR_Replicator_CustomFitness(nn.Module):
    """
    Replicator dynamics with a custom fitness function, in ALR coords
    with Hofbauer lifting.

    - The fitness function is built for N+1 species (original N + biomass).
    - ODE is solved in ALR (B x N), using biomass as ALR reference.
    """
    def __init__(self, fitness_fn_ctor, data_dim: int, learnable_skip: bool, energy_based: bool = False):
        """
        Parameters
        ----------
        fitness_fn_ctor : callable
            Constructor taking `num_species` and returning an nn.Module fitness fn.
            Called with `data_dim + 1` (Hofbauer dimension).
        data_dim : int
            Number of original species (N).
        learnable_skip : bool
            Whether to start with a zero-gated derivative.
        """
        super().__init__()

        self.USES_ODEINT = True
        self.data_dim = data_dim

        # Hofbauer: N original + 1 biomass
        num_species = data_dim + 1

        # Build the fitness model in Hofbauer dimension
        self.fitness_model = fitness_fn_ctor(num_species)

        # ODE function in ALR coordinates
        if energy_based:
            self.ode_func_raw = ODEFunc_ALRReplicator_CustomFitness_Energy(
                fitness_fn=self.fitness_model,
                learnable_skip=learnable_skip,
            )
        else:
            self.ode_func_raw = ODEFunc_ALRReplicator_CustomFitness(
                fitness_fn=self.fitness_model,
                learnable_skip=learnable_skip,
            )
        
        # Lambda-module wrapper for passing zero_mask
        self.ode_func = ODELambda(self.ode_func_raw)

    def forward(self, t, x):
        """
        x: B x N composition (sum_i x_i = 1)
        returns: T x B x N composition
        """
        # Step 1: augment with biomass proxy species in Hofbauer space
        x0_ext = hof.hofbauer_augment_state(x, biomass_init=1.0)  # B x (N+1)

        # Step 2: convert Hofbauer composition to ALR coords
        z0, zero_mask = alr.hofbauer_state_to_alr(x0_ext)  # B x N

        # Step 3: integrate in ALR space
        self.ode_func.set_context(zero_mask)
        z_traj = odeint(self.ode_func, z0, t)  # T x B x N

        # Step 4: map ALR traj back to N-dim composition
        y = alr.alr_to_hofbauer_state(z_traj, zero_mask)  # T x B x (N+1)
        x_final = hof.hofbauer_collapse_state(y)          # T x B x N

        return x_final


class ALR_Replicator_CustomFitness_IdEmbed_XEncode(nn.Module):
    """
    ALR replicator with custom fitness fn, id embeddings, and abundance encoding.

    - Always uses Hofbauer (N+1 species, biomass is reference).
    - ODE state is ALR (B x N).
    - Fitness model uses encoded abundances + embeddings in Hofbauer space.
    """
    def __init__(
        self,
        core_fitness_fn,
        data_dim: int,
        embed_dim: int,
        learnable_skip: bool,
        use_logx: bool,
        enrich_fn=None,
    ):
        super().__init__()

        self.USES_ODEINT = True
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)
        self.enrich_fn = enrich_fn
        self.data_dim = data_dim

        self.ode_func_raw = ODEFunc_ALRReplicator_CustomFitness_IdEmbed_XEncode(
            fitness_fn=core_fitness_fn,
            embed_dim=embed_dim,
            use_logx=use_logx,
            learnable_skip=learnable_skip,
        )

        # Lambda-module wrapper for embeddings_ext + zero_mask
        self.ode_func = ODELambda(self.ode_func_raw)

    def forward(self, t, x, ids):
        """
        x: B x N composition
        ids: B x N (indices fed to IdEmbedder)
        returns: T x B x N composition
        """
        # preprocess embeddings (N species)
        embeddings = self.embed(ids)  # B x N x embed_dim
        if self.enrich_fn is not None:
            embeddings = self.enrich_fn(embeddings)

        # Step 1: augment x with biomass proxy species in Hofbauer space
        x0_ext = hof.hofbauer_augment_state(x, biomass_init=1.0)  # B x (N+1)

        # biomass embedding: zero vector (could be made learnable)
        biomass_embed = embeddings.new_zeros(embeddings[..., :1, :].shape)  # B x 1 x embed_dim
        embeddings_ext = torch.cat([embeddings, biomass_embed], dim=1)      # B x (N+1) x embed_dim

        # Step 2: convert Hofbauer composition to ALR coords
        z0, zero_mask = alr.hofbauer_state_to_alr(x0_ext)  # B x N

        # Step 3: integrate in ALR space with fixed embeddings_ext
        self.ode_func.set_context(embeddings_ext, zero_mask)
        z_traj = odeint(
            self.ode_func,
            z0,
            t,
        )  # T x B x N

        # Step 4: map ALR traj back to N-dim composition
        y = alr.alr_to_hofbauer_state(z_traj, zero_mask)  # T x B x (N+1)
        x_final = hof.hofbauer_collapse_state(y)          # T x B x N

        return x_final                       # T x B x N


class ALR_Replicator_CustomFitness_IdEmbed(nn.Module):
    """
    ALR replicator with a custom fitness function and id embeddings passed
    directly into the fitness function.

    - Always uses Hofbauer (N+1 species, biomass is reference).
    - ODE state is ALR (B x N).
    - Fitness fn should output fitness on all N+1 species given (x_ext, embeddings_ext).
    """
    def __init__(
        self,
        fitness_fn,
        data_dim: int,
        embed_dim: int,
        learnable_skip: bool,
        enrich_fn=None,
    ):
        super().__init__()

        self.USES_ODEINT = True
        self.USES_CONDENSED = True

        self.embed = encoders.IdEmbedder(data_dim, embed_dim)
        self.enrich_fn = enrich_fn
        self.data_dim = data_dim

        self.ode_func_raw = ODEFunc_ALRReplicator_CustomFitness_IdEmbed(
            fitness_fn=fitness_fn,
            learnable_skip=learnable_skip,
        )

        # Lambda-module wrapper for embeddings_ext + zero_mask
        self.ode_func = ODELambda(self.ode_func_raw)

    def forward(self, t, x, ids):
        """
        x: B x N composition
        ids: B x N
        returns: T x B x N composition
        """
        # preprocess embeddings (N species)
        embeddings = self.embed(ids)  # B x N x embed_dim
        if self.enrich_fn is not None:
            embeddings = self.enrich_fn(embeddings)

        # Step 1: augment x with biomass proxy species in Hofbauer space
        x0_ext = hof.hofbauer_augment_state(x, biomass_init=1.0)  # B x (N+1)

        # biomass embedding: zero vector (could be made learnable)
        biomass_embed = embeddings.new_zeros(embeddings[..., :1, :].shape)  # B x 1 x embed_dim
        embeddings_ext = torch.cat([embeddings, biomass_embed], dim=1)      # B x (N+1) x embed_dim

        # Step 2: convert Hofbauer composition to ALR coords
        z0, zero_mask = alr.hofbauer_state_to_alr(x0_ext)  # B x N

        # Step 3: integrate in ALR space with fixed embeddings_ext
        self.ode_func.set_context(embeddings_ext, zero_mask)
        z_traj = odeint(
            self.ode_func,
            z0,
            t,
        )  # T x B x N

        # Step 4: map ALR traj back to N-dim composition
        y = alr.alr_to_hofbauer_state(z_traj, zero_mask)  # T x B x (N+1)
        x_final = hof.hofbauer_collapse_state(y)          # T x B x N

        return x_final
