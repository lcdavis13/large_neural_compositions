import torch
import torch.nn as nn
import math

class UniformFFN(nn.Module):
    def __init__(self, in_out_dim, num_params, num_hidden_layers, dropout=0.0):
        super().__init__()
        N, L = in_out_dim, num_hidden_layers

        a = (L - 1)
        b = 2 * N
        c = -num_params
        d = int((-b + math.sqrt(b**2 - 4*a*c)) / (2*a)) if a != 0 else num_params // (2*N)

        layers = [nn.Linear(N, d), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(L - 1):
            layers += [nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(d, N))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def compute_linear_tapered_widths(N, P, total_layers):
    k = (total_layers - 1) // 2

    def total_params(w):
        delta = (w - N) / k
        widths = [N + i * delta for i in range(k + 1)]
        total = sum(int(widths[i]) * int(widths[i + 1]) for i in range(k))
        total *= 2
        if total_layers % 2 == 0:
            total += int(widths[-1]) ** 2
        return total

    low, high = N, N * 10
    for _ in range(100):
        mid = (low + high) / 2
        if total_params(mid) < P:
            low = mid
        else:
            high = mid

    delta = (high - N) / k
    first_half = [int(round(N + i * delta)) for i in range(k + 1)]
    second_half = first_half[::-1] if total_layers % 2 == 0 else first_half[-2::-1]
    return first_half + second_half


def compute_exponential_tapered_widths(N, P, total_layers):
    k = (total_layers - 1) // 2

    def total_params(r):
        widths = [N * (r ** i) for i in range(k + 1)]
        total = sum(int(widths[i]) * int(widths[i + 1]) for i in range(k))
        total *= 2
        if total_layers % 2 == 0:
            total += int(widths[-1]) ** 2
        return total

    low, high = 1.0, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        if total_params(mid) < P:
            low = mid
        else:
            high = mid

    r = high
    first_half = [int(round(N * (r ** i))) for i in range(k + 1)]
    second_half = first_half[::-1] if total_layers % 2 == 0 else first_half[-2::-1]
    return first_half + second_half


class LinearTaperedFFN(nn.Module):
    def __init__(self, in_out_dim, num_params, num_hidden_layers, dropout=0.0):
        super().__init__()
        N = in_out_dim
        widths = compute_linear_tapered_widths(N, num_params, num_hidden_layers + 2)[1:-1]
        widths = [N] + widths + [N]

        layers = []
        for i in range(len(widths) - 1):
            layers += [nn.Linear(widths[i], widths[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        self.model = nn.Sequential(*layers[:-2])

    def forward(self, x):
        return self.model(x)


class ExponentialTaperedFFN(nn.Module):
    def __init__(self, in_out_dim, num_params, num_hidden_layers, dropout=0.0):
        super().__init__()
        N = in_out_dim
        widths = compute_exponential_tapered_widths(N, num_params, num_hidden_layers + 2)[1:-1]
        widths = [N] + widths + [N]

        layers = []
        for i in range(len(widths) - 1):
            layers += [nn.Linear(widths[i], widths[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        self.model = nn.Sequential(*layers[:-2])

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    import pandas as pd

    test_configs = [
        (100, 100000, 2),
        (128, 200000, 3),
        (64, 50000, 4),
        (256, 300000, 5),
        (64, 50000, 6),
        (256, 300000, 7),
    ]

    def extract_layer_widths(model):
        """Return a list of layer sizes including input and output"""
        dims = []
        for layer in model.model:
            if isinstance(layer, nn.Linear):
                dims.append(layer.in_features)
        dims.append(model.model[-1].out_features)
        return dims

    results = []
    for in_out_dim, num_params, num_layers in test_configs:
        u_model = UniformFFN(in_out_dim, num_params, num_layers)
        l_model = LinearTaperedFFN(in_out_dim, num_params, num_layers)
        e_model = ExponentialTaperedFFN(in_out_dim, num_params, num_layers)

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
            l_widths = extract_layer_widths(l_model)
            e_widths = extract_layer_widths(e_model)
            print(f"\nLayer widths for LinearTaperedFFN (L={num_layers}): {l_widths}")
            print(f"Layer widths for ExponentialTaperedFFN (L={num_layers}): {e_widths}")

    df = pd.DataFrame(results)
    print("\nParameter Summary:")
    print(df.to_string(index=False))
