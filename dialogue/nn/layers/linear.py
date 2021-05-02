from typing import Optional, Sequence

from torch import nn, Tensor

from dialogue.nn.layers import activation, normalization


class Linear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 normalization_type: Optional[str] = 'bn',
                 dropout: float = 0.3,
                 activation_type: Optional[str] = None,
                 residual_as_possible: bool = True):
        super().__init__()

        if residual_as_possible and in_features == out_features:
            self.residual = True
        else:
            self.residual = False

        self.layers = nn.Sequential()

        if normalization_type is not None:
            self.layers.add_module('normalization', normalization.get_normalization(
                normalized_shape=in_features,
                normalization_type=normalization_type))

        if dropout:
            self.layers.add_module('dropout', nn.Dropout(p=dropout))

        if activation_type and 'glu' in activation_type:
            self.layers.add_module(activation_type, activation.get_activation(activation_type=activation_type,
                                                                              in_features=in_features,
                                                                              out_features=out_features))
        else:
            self.layers.add_module('linear', nn.Linear(in_features=in_features, out_features=out_features))

        if activation_type and 'glu' not in activation_type:
            self.layers.add_module('activation', activation.get_activation(activation_type=activation_type))

    def forward(self, x: Tensor) -> Tensor:

        x_projected = self.layers(x)

        if self.residual:
            x_projected = x_projected + x

        return x_projected

    def extra_repr(self) -> str:
        return f'(residual={self.residual})'


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 sizes: Sequence[int],
                 normalization_type: Optional[str] = 'bn',
                 dropout: float = 0.15,
                 activation_type: Optional[str] = 'relu',
                 residual_as_possible: bool = True,
                 last_layer_activation_type: Optional[str] = None,
                 last_layer_dropout: Optional[float] = None,
                 last_layer_residual: bool = False):
        super().__init__()

        self.layers = nn.Sequential()

        for n, i_size in enumerate(range(len(sizes) - 1)):

            if i_size + 2 == len(sizes):
                current_activation_type = last_layer_activation_type
                residual = last_layer_residual
                layer_dropout = last_layer_dropout if last_layer_dropout is not None else dropout
            else:
                current_activation_type = activation_type
                residual = residual_as_possible
                layer_dropout = dropout

            self.layers.add_module(f'layer_{i_size + 1}',
                                   Linear(in_features=sizes[i_size],
                                          out_features=sizes[i_size + 1],
                                          normalization_type=normalization_type,
                                          dropout=layer_dropout,
                                          activation_type=current_activation_type,
                                          residual_as_possible=residual))

    def forward(self, x):

        x = self.layers(x)

        return x


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 increased_dim: int,
                 normalization_type: Optional[str] = 'rms',
                 dropout: float = 0.1,
                 activation_type: str = 'geglu'):
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module('increase', nn.Sequential())
        self.layers.add_module('decrease', nn.Sequential())

        if 'glu' in activation_type:
            self.layers.increase.add_module('glu',
                                            activation.get_activation(activation_type=activation_type,
                                                                      in_features=model_dim,
                                                                      out_features=increased_dim))
        else:
            self.layers.increase.add_module('linear', nn.Linear(in_features=model_dim, out_features=increased_dim))
            self.layers.increase.add_module('activation', activation.get_activation(
                activation_type=activation_type,
                in_features=model_dim,
                out_features=increased_dim
            ))

        if normalization_type is not None:
            self.layers.increase.add_module('normalization', normalization.get_normalization(
                normalized_shape=increased_dim,
                normalization_type=normalization_type
            ))

        self.layers.decrease.add_module('dropout', nn.Dropout(dropout))
        self.layers.decrease.add_module('linear', nn.Linear(in_features=increased_dim, out_features=model_dim))
        self.layers.decrease.add_module('output_dropout', nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [*, *, model_dim]
        :return: [*, *, model_dim]
        """

        x = self.layers(x)

        return x
