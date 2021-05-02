import math
from typing import Optional

import torch
from torch import nn, Tensor


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def gelu(x: Tensor) -> Tensor:
    alpha = math.sqrt(2.0 / math.pi)
    tensor_type = x.dtype
    x = x.float()
    x = 0.5 * x * (1.0 + torch.tanh(alpha * (x + 0.044715 * torch.pow(x, 3.0))))
    x = x.type(tensor_type)
    return x


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return swish(x)


class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)


ACTIVATIONS_MAPPER = {
    'relu': nn.ReLU(),
    're': nn.ReLU(),
    'tanh': nn.Tanh(),
    'tan': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'sigm': nn.Sigmoid(),
    'sig': nn.Sigmoid(),
    'swish': Swish(),
    'swi': Swish(),
    'gelu': GELU(),
    'ge': GELU()
}


class GLU(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation_type: str = 'relu', shared: bool = False):
        super().__init__()
        self.gate_activation = get_activation(activation_type=activation_type)
        self.gate = nn.Linear(in_features=in_features, out_features=out_features)
        self.projection = nn.Linear(in_features=in_features, out_features=out_features)
        if shared:
            self.projection = self.gate

    def forward(self, x: Tensor) -> Tensor:
        x = self.gate_activation(self.gate(x)) * self.projection(x)
        return x


def get_activation(activation_type: Optional[str] = None,
                   in_features: Optional[int] = None,
                   out_features: Optional[int] = None):

    if activation_type is None:
        return nn.Identity()

    if activation_type.endswith('glu'):

        assert in_features is not None, 'set hidden_dim for activation'

        if out_features is None:
            out_features = in_features

        activation_type = activation_type[:-3]

        if not activation_type:
            activation_type = 'sigmoid'

        activation_module = GLU(in_features=in_features,
                                out_features=out_features,
                                activation_type=activation_type)

    else:

        activation_module = ACTIVATIONS_MAPPER.get(activation_type, nn.Identity())

    return activation_module
