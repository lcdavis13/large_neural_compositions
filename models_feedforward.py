import torch
import torch.nn as nn
import math


class GatedSkipConnection(nn.Module):
    def __init__(self, identity_gate: bool = True):
        super(GatedSkipConnection, self).__init__()
        self.identity_gate = identity_gate
        
        if self.identity_gate:
            # Learnable parameters initialized to implement identity: y = x1
            self.a = nn.Parameter(torch.tensor(0.0))
            self.b = nn.Parameter(torch.tensor(1.0))
        else:
            # Non-learnable constants
            self.register_buffer('a', torch.tensor(1.0))
            self.register_buffer('b', torch.tensor(1.0))

    def forward(self, x0, x1):
        return self.a * x0 + self.b * x1



class UniformFFN(nn.Module):
    def __init__(self, in_out_dim, num_params, num_hidden_layers, dropout, identity_gate): 
        super().__init__()
        self.in_out_dim = in_out_dim
        L = num_hidden_layers

        # Solve for hidden width d
        a = L
        b = 2 * self.in_out_dim
        c = -num_params
        d = int((-b + math.sqrt(b**2 - 4*a*c)) / (2*a)) if a != 0 else num_params // (2*self.in_out_dim)

        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Input layer
        self.layers.append(nn.Linear(self.in_out_dim, d))

        # Hidden layers
        for _ in range(L):
            self.layers.append(nn.Linear(d, d))
            self.skips.append(GatedSkipConnection(identity_gate))

        # Output layer
        self.layers.append(nn.Linear(d, self.in_out_dim))

        # Final gated skip: from input to output
        self.skips.append(GatedSkipConnection(identity_gate))

    def forward(self, x):
        x_input = x  # Save original input for the final skip

        x = self.layers[0](x)
        x = self.activation(x)
        x = self.dropout(x)

        for i in range(1, len(self.layers) - 1):
            x0 = self.layers[i](x)
            x0 = self.activation(x0)
            x0 = self.dropout(x0)
            x = self.skips[i - 1](x0, x)

        x_out = self.layers[-1](x)
        x_final = self.skips[-1](x_out, x_input)  # Final skip: input to output
        return x_final





def compute_linear_tapered_widths(N, P, total_layers):
    if total_layers < 1:
        raise ValueError("total_layers must be >= 1")

    if total_layers % 2 == 1:
        k = (total_layers + 1) // 2 + 1

        def total_params(delta):
            widths = [N + i * delta for i in range(k)]
            return 2 * sum(widths[i] * widths[i+1] for i in range(k - 1))

        low, high = 0.0, N * 10.0
        for _ in range(100):
            mid = (low + high) / 2
            if total_params(mid) < P:
                low = mid
            else:
                high = mid

        delta = (low + high) / 2
        first_half = [round(N + i * delta) for i in range(k)]
        second_half = first_half[-2::-1]
        return first_half + second_half

    else:
        k = total_layers // 2

        def total_params(delta):
            widths = [N + i * delta for i in range(k + 1)]
            return 2 * sum(widths[i] * widths[i+1] for i in range(k)) + widths[-1] ** 2

        low, high = 0.0, N * 10.0
        for _ in range(100):
            mid = (low + high) / 2
            if total_params(mid) < P:
                low = mid
            else:
                high = mid

        delta = (low + high) / 2
        first_half = [round(N + i * delta) for i in range(k + 1)]
        second_half = first_half[::-1]
        return first_half + second_half



def compute_exponential_tapered_widths(N, P, total_layers):
    if total_layers < 1:
        raise ValueError("total_layers must be >= 1")

    def total_params(r, k, even):
        if even:
            series_sum = sum(N**2 * r**(2*i + 1) for i in range(k))
            return 2 * series_sum + (N * r**k) ** 2
        else:
            series_sum = sum(N**2 * r**(2*i + 1) for i in range(k - 1))
            return 2 * series_sum

    if total_layers % 2 == 1:
        k = (total_layers + 1) // 2 + 1
        even = False
    else:
        k = total_layers // 2
        even = True

    low, high = 1.01, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        current = total_params(mid, k, even)
        if current < P:
            low = mid
        else:
            high = mid

    r = (low + high) / 2
    if even:
        first_half = [round(N * r**i) for i in range(k + 1)]
        second_half = first_half[::-1]
    else:
        first_half = [round(N * r**i) for i in range(k)]
        second_half = first_half[-2::-1]

    return first_half + second_half



class TaperedFFNBase(nn.Module):
    def __init__(self, in_out_dim, num_params, num_hidden_layers, dropout=0.0, identity_gate=True, width_fn=None):
        super().__init__()
        assert width_fn is not None, "You must provide a width function."
        N = in_out_dim
        widths = width_fn(N, num_params, num_hidden_layers)

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.skips = nn.ModuleDict()

        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i+1]))
            self.activations.append(nn.Sequential(nn.ReLU(), nn.Dropout(dropout)))

        for i in range(len(widths) // 2):
            j = len(widths) - 2 - i  # skip connections mirror internal layers
            if widths[i] == widths[j + 1]:
                self.skips[str(i)] = GatedSkipConnection(identity_gate)

    def forward(self, x):
        intermediates = {}
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            x_new = act(layer(x))
            j = str(len(self.layers) - 1 - i)
            if str(i) in self.skips and j in intermediates:
                x = self.skips[str(i)](x_new, intermediates[j])
            else:
                intermediates[str(i)] = x
                x = x_new
        return x


class LinearTaperedFFN(TaperedFFNBase):
    def __init__(self, in_out_dim, num_params, num_hidden_layers, dropout=0.0, identity_gate=True):
        super().__init__(in_out_dim, num_params, num_hidden_layers, dropout, identity_gate, width_fn=compute_linear_tapered_widths)


class ExponentialTaperedFFN(TaperedFFNBase):
    def __init__(self, in_out_dim, num_params, num_hidden_layers, dropout=0.0, identity_gate=True):
        super().__init__(in_out_dim, num_params, num_hidden_layers, dropout, identity_gate, width_fn=compute_exponential_tapered_widths)




if __name__ == "__main__":
    import pandas as pd

    test_configs = [
        (100, 100000, 2),
        (128, 200000, 3),
        (64, 50000, 4),
        (128, 300000, 5),
        (64, 50000, 6),
        (128, 300000, 7),
    ]

    def extract_layer_widths(model):
        """Return a list of layer sizes including input and output"""
        dims = []
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                dims.append(layer.in_features)
        if isinstance(model.layers[-1], nn.Linear):
            dims.append(model.layers[-1].out_features)
        return dims

    results = []
    for in_out_dim, num_params, num_layers in test_configs:
        u_model = UniformFFN(in_out_dim, num_params, num_layers, dropout=0.0, identity_gate=True)
        l_model = LinearTaperedFFN(in_out_dim, num_params, num_layers, dropout=0.0, identity_gate=True)
        e_model = ExponentialTaperedFFN(in_out_dim, num_params, num_layers, dropout=0.0, identity_gate=True)

        row = {
            'in_out_dim': in_out_dim,
            'target_params': num_params,
            'num_layers': num_layers,
            'Uniform': sum(p.numel() for p in u_model.parameters()),
            'LinearTapered': sum(p.numel() for p in l_model.parameters()),
            'ExponentialTapered': sum(p.numel() for p in e_model.parameters()),
        }
        results.append(row)

        if num_layers in (4, 5):
            u_widths = extract_layer_widths(u_model)
            l_widths = extract_layer_widths(l_model)
            e_widths = extract_layer_widths(e_model)
            print(f"\nLayer widths for UniformFFN (L={num_layers}): {u_widths}")
            print(f"Layer widths for LinearTaperedFFN (L={num_layers}): {l_widths}")
            print(f"Layer widths for ExponentialTaperedFFN (L={num_layers}): {e_widths}")

    df = pd.DataFrame(results)
    print("\nParameter Summary:")
    print(df.to_string(index=False))
