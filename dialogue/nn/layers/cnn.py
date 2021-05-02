from typing import Optional, Tuple, Union, List

import torch.nn.functional as F
from torch import nn, Tensor

from dialogue.nn.layers import activation, common, normalization


class CausalCNN(nn.Module):

    def __init__(self,
                 model_dim: int,
                 kernel_size: int,
                 normalization_type: str = 'ln',
                 dropout: float = 0.1,
                 activation_type: str = 'relu',
                 output_dim: Optional[int] = None):
        super().__init__()

        output_dim = output_dim if output_dim is not None else model_dim

        self.normalization = normalization.get_normalization(normalized_shape=model_dim,
                                                             normalization_type=normalization_type)
        self.dropout = common.SpatialDropout(p=dropout)
        self.layer = nn.Conv1d(in_channels=model_dim, out_channels=output_dim, kernel_size=kernel_size)
        self.activation = activation.get_activation(activation_type=activation_type)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:

        x = self.normalization(x)
        x = self.dropout(x)
        x = common.embedding_masking(x, pad_mask=pad_mask)

        x = F.pad(input=x.transpose(1, 2), pad=[self.layer.kernel_size[0] - 1, 0])

        x = self.layer(x).transpose(1, 2)

        x = self.activation(x)

        x = common.embedding_masking(x, pad_mask=pad_mask)

        return x


class ResidualCausalCNN(CausalCNN):

    def __init__(self,
                 model_dim: int,
                 kernel_size: int,
                 normalization_type: str = 'ln',
                 dropout: float = 0.1,
                 activation_type: str = 'relu'):
        super().__init__(
            model_dim=model_dim,
            kernel_size=kernel_size,
            normalization_type=normalization_type,
            dropout=dropout,
            activation_type=activation_type,
            output_dim=model_dim
        )

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:

        x = super().forward(x=x, pad_mask=pad_mask) + x

        return x


class IncreasedCausalCNN(nn.Module):

    def __init__(self,
                 model_dim: int,
                 kernel_size_increase: int,
                 inner_dim: Optional[int] = None,
                 kernel_size_decrease: Optional[int] = None,
                 normalization_type: str = 'ln',
                 dropout: float = 0.3,
                 activation_type: str = 'relu'):
        super().__init__()

        inner_dim = inner_dim if not None else model_dim
        kernel_size_decrease = kernel_size_decrease if not None else kernel_size_increase

        self.increase_cnn = CausalCNN(model_dim=model_dim,
                                      kernel_size=kernel_size_increase,
                                      normalization_type=normalization_type,
                                      dropout=dropout,
                                      activation_type=activation_type,
                                      output_dim=inner_dim)

        self.decrease_cnn = CausalCNN(model_dim=inner_dim,
                                      kernel_size=kernel_size_decrease,
                                      normalization_type=normalization_type,
                                      dropout=dropout,
                                      activation_type=activation_type,
                                      output_dim=model_dim)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:

        x = self.decrease_cnn(self.increase_cnn(x, pad_mask), pad_mask) + x

        return x


class ParallelCausalCNN(nn.Module):

    def __init__(self,
                 model_dim: int,
                 kernel_sizes: List[Union[int, Tuple[int, int]]],
                 inner_dim: Optional[int] = None,
                 normalization_type: str = 'ln',
                 dropout: float = 0.3,
                 activation_type: str = 'relu'):
        super().__init__()

        self.layers = nn.ModuleList()

        for ks in kernel_sizes:

            if isinstance(ks, tuple):

                layer = IncreasedCausalCNN(model_dim=model_dim,
                                           kernel_size_increase=ks[0],
                                           inner_dim=inner_dim,
                                           kernel_size_decrease=ks[1],
                                           normalization_type=normalization_type,
                                           dropout=dropout,
                                           activation_type=activation_type)

            else:

                layer = CausalCNN(model_dim=model_dim,
                                  kernel_size=ks,
                                  normalization_type=normalization_type,
                                  dropout=dropout,
                                  activation_type=activation_type,
                                  output_dim=model_dim)

            self.layers.append(layer)

        self.normalization = normalization.get_normalization(normalized_shape=model_dim,
                                                             normalization_type=normalization_type)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:

        x_cnn = [layer(x, pad_mask=pad_mask) for layer in self.layers]

        for part in x_cnn:
            x += part

        return x
