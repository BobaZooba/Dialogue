from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dialogue.nn.layers import common, normalization


class ResidualBidirectionalLSTM(nn.Module):

    def __init__(self, model_dim: int, dropout: float = 0.3, num_layers: int = 1):
        super().__init__()

        self.normalization_layer = normalization.get_normalization(normalized_shape=model_dim,
                                                                   normalization_type='rms')

        self.rnn_dropout = common.SpatialDropout(p=dropout)

        self.lstm = nn.LSTM(input_size=model_dim,
                            hidden_size=model_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.projection_dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(in_features=model_dim * 2, out_features=model_dim)

    def forward(self,
                x: Tensor,
                pad_mask: Tensor) -> Tensor:

        residual = x

        x = common.embedding_masking(self.normalization_layer(x), pad_mask)

        x = self.rnn_dropout(x)

        x_packed = pack_padded_sequence(x,
                                        pad_mask.sum(1),
                                        batch_first=True,
                                        enforce_sorted=False)

        x_packed, _ = self.lstm(x_packed)

        x, _ = pad_packed_sequence(x_packed,
                                   batch_first=True,
                                   total_length=x.size(1))

        x = self.projection_dropout(x)
        x = self.projection(x)

        x = common.embedding_masking(x=x, pad_mask=pad_mask)

        x = x + residual

        return x
