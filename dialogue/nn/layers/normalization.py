from typing import Optional

import torch
from torch import nn, Tensor


class BatchNorm1d(nn.BatchNorm1d):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         affine=affine,
                         track_running_stats=track_running_stats)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self,
                 normalized_shape: int,
                 partial: float = -1.,
                 eps: float = 1e-8,
                 bias: bool = False):
        """
            Root Mean Square Layer Normalization
        :param normalized_shape: model size
        :param partial: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super().__init__()

        self.eps = eps
        self.normalized_shape = normalized_shape
        self.partial = partial
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.register_parameter('scale', self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(normalized_shape))
            self.register_parameter('offset', self.offset)

    def forward(self, x: Tensor) -> Tensor:

        if self.partial < 0. or self.partial > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.normalized_shape
        else:
            partial_size = int(self.normalized_shape * self.partial)
            partial_x, _ = torch.split(x, [partial_size, self.normalized_shape - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

    def extra_repr(self):
        return f'{self.normalized_shape,}, partial={self.partial}, eps={self.eps}'


NORMS_MAPPER = {
    'batch_norm': nn.BatchNorm1d,
    'bn': nn.BatchNorm1d,
    'layer_norm': nn.LayerNorm,
    'ln': nn.LayerNorm,
    'rms_norm': RMSNorm,
    'rms': RMSNorm,
}


def get_normalization(normalized_shape: int, normalization_type: Optional[str] = None):

    normalization_module = NORMS_MAPPER.get(normalization_type, nn.LayerNorm)
    normalization_layer = normalization_module(normalized_shape)

    return normalization_layer
