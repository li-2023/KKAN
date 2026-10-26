import torch
from torch import nn

def conjugate(x):
    return torch.cat((x[..., :1, :], -x[..., 1:, :]), dim=-2)

def cayley_dickson(x, y):
    components = x.shape[-2]
    if components != y.shape[-2] or components < 1 or components & (components - 1):
        raise ValueError()
    if components == 1:
        return x * y
    a, b = x.chunk(2, dim=-2)
    c, d = y.chunk(2, dim=-2)
    return torch.cat((cayley_dickson(a, c) - cayley_dickson(conjugate(d), b),
                      cayley_dickson(a, d) + cayley_dickson(b, conjugate(c))), dim=-2)

def multiplication_table(components=8):
    basis = torch.eye(components).unsqueeze(-1)

    return cayley_dickson(basis[:, None], basis[None, :]).squeeze(-1).permute(2, 0, 1).contiguous()

class HypercomplexLinear(nn.Module):

    def __init__(self, width, components=8):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(components, width, width))
        self.register_buffer('table', multiplication_table(components), persistent=False)
        for component in self.weight:
            nn.init.xavier_uniform_(component)

    def forward(self, x):

        blocks = torch.einsum('cab,aoi->cobi', self.table, self.weight)
        output = torch.nn.functional.linear(x.flatten(-2), blocks.flatten(0, 1).flatten(1))
        return output.reshape(*x.shape[:-2], self.weight.shape[0], self.weight.shape[1])

class ComponentDropout(nn.Module):

    def __init__(self, probability=0.1):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError()
        self.probability = probability

    def forward(self, x):
        keep = 1.0 - self.probability
        if self.training:
            mask = x.new_empty((*x.shape[:-1], 1)).bernoulli_(keep)
            return x * mask
        return x * keep
