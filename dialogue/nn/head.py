from typing import Optional, Tuple, Union, List

from torch import nn

from dialogue import io
from dialogue.nn.layers import linear


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 sizes: Union[Tuple[int], List[int]],
                 normalization_type: Optional[str] = 'bn',
                 dropout: float = 0.1,
                 activation_type: Optional[str] = 'relu',
                 residual_as_possible: bool = True,
                 last_layer_activation_type: Optional[nn.Module] = None,
                 last_layer_dropout: Optional[float] = 0,
                 last_layer_residual: bool = False):
        super().__init__()

        self.encoder = linear.MultiLayerPerceptron(
            sizes=sizes,
            normalization_type=normalization_type,
            dropout=dropout,
            activation_type=activation_type,
            residual_as_possible=residual_as_possible,
            last_layer_activation_type=last_layer_activation_type,
            last_layer_dropout=last_layer_dropout,
            last_layer_residual=last_layer_residual
        )

    def forward(self, batch: io.Batch) -> io.Batch:

        batch[io.TYPES.head] = self.encoder(batch[io.TYPES.aggregation])

        return batch
