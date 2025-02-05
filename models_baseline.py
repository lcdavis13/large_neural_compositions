import torch
import torch.nn as nn
import torch.nn.functional as F

import models_cnode
from ode_solver import odeint


class ReturnInput(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
    

class ReturnTrainingSampleMixture(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_samples, Y_train):
        """
        :param input_dim: Dimensionality of X
        :param hidden_dim: Number of hidden units in classifier
        :param num_samples: Number of training samples (N)
        :param output_dim: Dimensionality of Y
        :param Y_train: Tensor of shape (N, D) containing Y values
        """
        super().__init__()

        self.num_samples = num_samples  # N
        self.Y_train = nn.Parameter(Y_train, requires_grad=False)  # Store training Y values (N, D) as non-trainable

        # Classifier network that predicts logits over indices
        self.classifier = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_samples)  # Output logits for N classes
        )

    def forward(self, x):
        """
        Predicts Y values using soft index retrieval.
        :param x: Input tensor (batch_size, input_dim)
        :return: Predicted Y tensor (batch_size, output_dim)
        """
        # Predict class logits
        logits = self.classifier(x)  # (batch_size, N)

        # Convert logits to soft probabilities
        soft_indices = F.softmax(logits, dim=1)  # (batch_size, N)

        # Retrieve Y as weighted sum of all Y_train samples (differentiable)
        retrieved_Y = torch.matmul(soft_indices, self.Y_train)  # (batch_size, D)

        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(retrieved_Y)  # This will hold the output
        
        batch_size = x.shape[0]
        for i in range(batch_size):
            f_selected = retrieved_Y[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back

        return y 


class SingleConst(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self):
        super().__init__()
        self.f = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return self.f.expand(x.shape)


class SingleConstFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self):
        super().__init__()
        self.f = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        f = self.f.expand(x.shape)
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(f)  # This will hold the output
        
        for i in range(x.shape[0]):
            f_selected = f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back
        
        return y


class ConstOutput(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        f_repeated = self.f.unsqueeze(0).expand(batch_size, -1)  # Repeat f for each batch element
        return f_repeated


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
        f = x * f
        
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
    
    
class SingleLayer(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x):
        return self.f(x)


class SingleLayerMultiplied(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return y  # B x N


class SingleLayerFiltered(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x):  # x' = x*f(x)
        fx = self.f(x)  # B x N

        ones = torch.zeros_like(x)
        y = torch.mul(ones, fx)  # B x N
        
        return y  # B x N


class SingleLayerSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x + f(x)
        fx = self.f(x)  # B x N
        
        y = x + fx  # B x N
        
        return y  # B x N


class SingleLayerMultipliedSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x + x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return x + y  # B x N


class SingleLayerFilteredSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x + x*f(x)
        fx = self.f(x)  # B x N
        
        ones = torch.zeros_like(x)
        y = torch.mul(ones, fx)  # B x N
        
        return x + y  # B x N


class cNODE1_singlestep(nn.Module):
    # cNODE1, but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    
    def __init__(self, N, bias):
        super().__init__()
        self.func = models_cnode.ODEFunc_cNODE1(N, bias)
    
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
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):
        fx = self.f.expand(x.size(0), -1)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE0(nn.Module):
    # cNODE where "F(x)" does not depend on x. In other words, it learns a fixed fitness value for each species regardless of which species are actually present.
    def __init__(self, N):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODE0(N)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class cNODE0_singlestep(nn.Module):
    # Identical to cNODE0 but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    def __init__(self, N):
        super().__init__()
        self.func = ODEFunc_cNODE0(N)
    
    def forward(self, x):
        dxdt = self.func([0.0], x)
        return x + dxdt