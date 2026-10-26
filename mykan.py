import math

import torch
from torch import nn
from torch.nn import functional as F

def parse_bases(names):
    aliases = {
        'original': 'bspline',
        'b-spline': 'bspline',
        'b_spline': 'bspline',
        'spline': 'bspline',
        'bspline': 'bspline',
        'fourier': 'fourier',
        'rbf': 'rbf',
        'bernstein': 'bernstein',
        'polynomial': 'polynomial',
    }
    if not isinstance(names, str):
        names = '+'.join(names)
    names = names.replace(',', '+').split('+')
    return tuple(dict.fromkeys(aliases[name.strip().lower().removeprefix('kan_')] for name in names))

class KANLayer(nn.Module):
    def __init__(self, inputdim, outdim, bases=('rbf',), basis_size=5, spline_order=3):
        super().__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.bases = parse_bases(bases)
        self.basis_size = basis_size
        self.spline_order = spline_order
        self.base_weight = nn.Parameter(torch.empty(outdim, inputdim))
        self.basis_weight = nn.Parameter(torch.ones(outdim, inputdim))
        self.coeffs = nn.ParameterDict()
        nn.init.xavier_uniform_(self.base_weight)
        for basis in self.bases:
            size = {
                'bspline': basis_size + spline_order,
                'fourier': 2 * basis_size,
                'rbf': basis_size,
                'bernstein': basis_size + 1,
                'polynomial': basis_size + 1,
            }[basis]
            coefficients = nn.Parameter(torch.empty(outdim, inputdim, size))
            nn.init.normal_(coefficients, std=1.0 / math.sqrt(inputdim * size))
            self.coeffs[basis] = coefficients
        if 'bspline' in self.bases:
            grid = torch.arange(-spline_order, basis_size + spline_order + 1, dtype=torch.float32)
            self.register_buffer('grid', -1.0 + 2.0 * grid / basis_size)
        if 'fourier' in self.bases:
            self.register_buffer('harmonics', torch.arange(1, basis_size + 1, dtype=torch.float32))
        if 'rbf' in self.bases:
            centers = torch.linspace(-1.0, 1.0, basis_size).repeat(inputdim, 1)
            self.centers = nn.Parameter(centers)
            self.log_rho = nn.Parameter(torch.full((inputdim, basis_size), math.log(math.expm1(1.0))))
        if 'bernstein' in self.bases:
            self.register_buffer('powers', torch.arange(basis_size + 1))
            self.register_buffer('binomial', torch.tensor([math.comb(basis_size, k) for k in range(basis_size + 1)], dtype=torch.float32))

    def basis_values(self, x, basis):
        if basis == 'bspline':
            values = ((x.unsqueeze(-1) >= self.grid[:-1]) & (x.unsqueeze(-1) < self.grid[1:])).to(x.dtype)
            for order in range(1, self.spline_order + 1):
                left = (x.unsqueeze(-1) - self.grid[:-(order + 1)]) / (self.grid[order:-1] - self.grid[:-(order + 1)])
                right = (self.grid[order + 1:] - x.unsqueeze(-1)) / (self.grid[order + 1:] - self.grid[1:-order])
                values = left * values[..., :-1] + right * values[..., 1:]
            return values
        if basis == 'fourier':
            angles = x.unsqueeze(-1) * self.harmonics
            return torch.cat((angles.cos(), angles.sin()), dim=-1)
        if basis == 'rbf':
            rho = F.softplus(self.log_rho) + torch.finfo(x.dtype).eps
            return torch.exp(-rho * (x.unsqueeze(-1) - self.centers).square())
        if basis == 'bernstein':
            z = x.sigmoid().unsqueeze(-1)
            return self.binomial * z.pow(self.powers) * (1.0 - z).pow(self.basis_size - self.powers)
        return torch.stack([x.pow(k) for k in range(self.basis_size + 1)], dim=-1)

    def forward(self, x):
        shape = x.shape[:-1] + (self.outdim,)
        x = x.reshape(-1, self.inputdim)
        output = F.linear(F.silu(x), self.base_weight)
        for basis in self.bases:
            values = self.basis_values(x, basis)
            weight = self.coeffs[basis] * self.basis_weight.unsqueeze(-1)
            output = output + F.linear(values.flatten(1), weight.flatten(1))
        return output.reshape(shape)

class KANNetwork(nn.Sequential):
    def __init__(self, widths, bases=('rbf',), basis_size=5, spline_order=3):
        super().__init__(*(KANLayer(a, b, bases, basis_size, spline_order) for a, b in zip(widths[:-1], widths[1:])))
