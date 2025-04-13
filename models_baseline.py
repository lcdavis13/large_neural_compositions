import torch
import torch.nn as nn

import models_cnode
from ode_solver import odeint



class ReturnInput(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self):
        super().__init__()
    
    def fit(self, X, Y):
        pass  # No fitting or training needed

    
    def streaming_fit(self, dataset, device):
        pass
    
    def forward(self, x):
        return x


class ConstOutput(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))

    def fit(self, X, Y):
        self.f = nn.Parameter(Y.mean(dim=0).detach())

    def streaming_fit(self, dataset, device):
        sum_Y = None
        count_Y = 0

        for batch in dataset:
            Y_batch = batch['y'].to(device)

            if sum_Y is None:
                sum_Y = Y_batch.sum(dim=0)
            else:
                sum_Y += Y_batch.sum(dim=0)

            count_Y += Y_batch.shape[0]

        if count_Y > 0:
            self.f = nn.Parameter((sum_Y / count_Y).detach())
        else:
            raise ValueError("Dataset appears to be empty.")


    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        f_repeated = self.f.unsqueeze(0).expand(batch_size, -1)  # Repeat f for each batch element
        return f_repeated


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None

    def fit(self, X, Y):
        # Add bias term
        ones = torch.ones(X.shape[0], 1, device=X.device)
        X_aug = torch.cat([X, ones], dim=1)  # (samples, features + 1)

        # Closed-form solution: beta = (X^T X)^-1 X^T Y
        beta, *_ = torch.linalg.lstsq(X_aug, Y)

        beta = beta[:X_aug.shape[1]]  # keep only relevant part

        self.A = beta[:-1].T  # (output_dim, input_dim)
        self.b = beta[-1].T   # (output_dim)

    def streaming_fit(self, dataset, device):
        XtX = None
        XtY = None

        for batch in dataset:
            X_batch = batch['x0'].to(device)
            Y_batch = batch['y'].to(device)

            ones = torch.ones(X_batch.shape[0], 1, device=device)
            X_aug = torch.cat([X_batch, ones], dim=1)

            if XtX is None:
                XtX = X_aug.T @ X_aug
                XtY = X_aug.T @ Y_batch
            else:
                XtX += X_aug.T @ X_aug
                XtY += X_aug.T @ Y_batch

        beta = torch.linalg.solve(XtX, XtY)

        self.A = beta[:-1].T
        self.b = beta[-1].T


    def forward(self, X):
        if self.A is None or self.b is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.A.T + self.b


class ConstOutputFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))


    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        f = self.f.unsqueeze(0).expand(batch_size, -1)  # Repeat f for each batch element
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(f)  # This will hold the output
        
        for i in range(batch_size):
            f_selected = f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back

        return y


class LinearRegressionMaskNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None

    def forward(self, X):
        if self.A is None or self.b is None:
            raise ValueError("Model is not fitted yet.")
        
        y_pred = X @ self.A.T + self.b  # (batch_size, output_dim)

        # Mask and normalize
        mask = X != 0  # shape (batch_size, input_dim)
        y_out = torch.zeros_like(y_pred)

        for i in range(X.shape[0]):
            masked = y_pred[i, mask[i]]
            if masked.numel() > 0:
                normalized = masked / masked.sum()
                y_out[i, mask[i]] = normalized

        return y_out


class SingleLayerPerceptron(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, t, x):
        h = self.f1(x)
        h = self.relu(h)
        h = self.f2(h)
        
        return h
    
    
class SLPFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(f)  # This will hold the output
        
        for i in range(batch_size):
            f_selected = f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back
        
        return y


class SLPSumFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        f = x + f
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(f)  # This will hold the output
        
        for i in range(batch_size):
            f_selected = f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back
        
        return y


class SLPMultFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M, identity_gate):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)

        if identity_gate:
            self.gateA = nn.Parameter(torch.tensor(0.0))
            self.gateB = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('gateA', torch.tensor(1.0))
            self.register_buffer('gateB', torch.tensor(0.0))
    
    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        f = x * f

        gated_f = self.gateA*f + self.gateB*x
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(gated_f)  # This will hold the output
        
        for i in range(batch_size):
            f_selected = gated_f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back
        
        return y


class SLPMultSumFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        f = (x * f) + x
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(f)  # This will hold the output
        
        for i in range(batch_size):
            f_selected = f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back
        
        return y


class cNODE1_singlestep(nn.Module):
    # cNODE1, but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    
    def __init__(self, N, bias, init_zero, identity_gate):
        super().__init__()
        self.func = models_cnode.ODEFunc_cNODE1(N, bias, init_zero=init_zero, identity_gate=identity_gate)
    
    def forward(self, x):
        dxdt = self.func([0.0], x)
        return x + dxdt


class SLPODE(nn.Module):
    # use odeint to train a single layer perceptron's fixed point
    def __init__(self, N, M):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = SingleLayerPerceptron(N, M)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class ODEFunc_cNODE0(nn.Module):
    # identical to ConstReplicator, except in ODE form; it returns the derivative instead of the next state.
    def __init__(self, N, init_zero=True, identity_gate=False):
        super().__init__()
        self.f = nn.Parameter(torch.rand(N))
        
        if not identity_gate:
            self.register_buffer('gate', torch.tensor(1.0)) # Make sure this has the same name as the parameter version
        
        if init_zero:
            # Initialize weights and biases to zero (this is the original approach from the paper)
            nn.init.zeros_(self.f)

            if identity_gate:
                self.gate = nn.Parameter(torch.tensor(1.0)) # modified identity_gate for when init_zero is used. Otherwise there would be symmetry and model couldn't learn. In the hypothetical where identity_gate gate provides some benefit beyond the initialization to zero, this can capture that benefit when init_zero is used.
        else:
            if identity_gate:
                self.gate = nn.Parameter(torch.tensor(0.0)) # regular identity_gate, when init_zero isn't used
    

    def forward(self, t, x):
        fx = self.f.expand(x.size(0), -1)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return self.gate*dxdt  # B x N


class cNODE0(nn.Module):
    # cNODE where "F(x)" does not depend on x. In other words, it learns a fixed fitness value for each species regardless of which species are actually present.
    def __init__(self, N, init_zero, identity_gate):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODE0(N, init_zero=init_zero, identity_gate=identity_gate)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class cNODE0_singlestep(nn.Module):
    # Identical to cNODE0 but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    def __init__(self, N, init_zero, identity_gate):
        super().__init__()
        self.func = ODEFunc_cNODE0(N, init_zero=init_zero, identity_gate=identity_gate)
    
    def forward(self, x):
        dxdt = self.func([0.0], x)
        return x + dxdt