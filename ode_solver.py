# ode_solver.py

import os
import torch

try:
    if os.getenv("SOLVER") == "torchdiffeq":
        from torchdiffeq import odeint as torchdiffeq_odeint
    elif os.getenv("SOLVER") == "torchdiffeq_memsafe":
        from torchdiffeq import odeint_adjoint as torchdiffeq_odeint
    elif os.getenv("SOLVER") in ["torchode", "torchode_memsafe"]:
        import torchode as to
    else:
        raise ValueError("Invalid SOLVER environment variable")
except ImportError as e:
    raise ImportError(f"Error importing specified solver package: {e}")


def odeint(func, y0, t):
    """
    Dynamically selects an ODE solver based on the SOLVER environment variable, and solves the fixed point ODE.

    Syntax identical to torchdiffeq.odeint
    """
    
    solver = os.getenv("SOLVER")
    
    if solver == "torchdiffeq" or solver == "torchdiffeq_memsafe":
        # Using torchdiffeq's odeint or odeint_adjoint
        return torchdiffeq_odeint(func, y0, t)
    
    elif solver == "torchode" or solver == "torchode_memsafe":
        # Attempting to solve a bug where after hundreds of epochs it decides to not track gradients anymore
        if not y0.requires_grad:
            y0 = y0.clone().detach().requires_grad_(True)
        
        # Using torchode with equivalent settings
        term = to.ODETerm(func)
        step_method = to.Tsit5(term=term)  # Tsit5 as step method
        step_size_controller = to.PIDController(
            atol=1e-6, rtol=1e-3, pcoeff=0.2, icoeff=0.5, dcoeff=0.0, term=term
        )
        
        # Choose controller type based on solver selection
        if solver == "torchode":
            adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
        elif solver == "torchode_memsafe":
            adjoint = to.BacksolveAdjoint(term, step_method, step_size_controller) # not working yet
        else:
            raise ValueError("Invalid solver type for torchode.")
        
        # Prepare the problem with repeated time evaluations for batch consistency
        batch_size = y0.shape[0]
        t_eval = t.repeat((batch_size, 1))
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        
        # Solve the problem
        sol = adjoint.solve(problem)
        
        # Return the solution in a format compatible with torchdiffeq
        sol_ys = sol.ys.transpose(0, 1)
        
        # Attempting to solve a bug where after hundreds of epochs it decides to not track gradients anymore
        if not sol_ys.requires_grad:
            sol_ys = sol_ys.clone().detach().requires_grad_(True)
        return sol_ys
    
    else:
        raise ValueError(
            "Invalid SOLVER environment variable.")
