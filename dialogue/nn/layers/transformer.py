from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from dialogue.nn.layers import linear, attention, normalization


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 head_dim: Optional[int] = None,
                 normalization_type: str = 'rms',
                 dropout: float = 0.1,
                 activation_type: str = 'geglu',
                 use_attention_bias: bool = False,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 16,
                 use_bias_positions: bool = True,
                 feed_forward_normalization_type: Optional[str] = 'rms'):
        super().__init__()

        self.normalization_attention = normalization.get_normalization(
            normalized_shape=model_dim, normalization_type=normalization_type)

        self.self_attention = attention.MultiHeadSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_bias=use_attention_bias,
            use_relative_positions=use_relative_positions,
            max_relative_position=max_relative_position,
            use_bias_positions=use_bias_positions
        )

        self.dropout_attention = nn.Dropout(dropout)

        self.normalization_feed_forward = normalization.get_normalization(
            normalized_shape=model_dim, normalization_type=normalization_type)

        self.position_wise_feed_forward = linear.PositionWiseFeedForwardLayer(
            model_dim=model_dim,
            increased_dim=feed_forward_dim,
            normalization_type=feed_forward_normalization_type,
            dropout=dropout,
            activation_type=activation_type
        )

        self.dropout_feed_forward = nn.Dropout(dropout)

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :return: [batch_size, sequence_length, model_dim]
        """

        hidden = self.normalization_attention(x)

        attention_scores, hidden = self.self_attention(x=hidden,
                                                       pad_mask=pad_mask,
                                                       attention_mask=attention_mask)

        hidden = self.dropout_attention(hidden)

        x = x + hidden

        hidden = self.normalization_feed_forward(x)

        hidden = self.position_wise_feed_forward(hidden)

        x = x + self.dropout_feed_forward(hidden)

        return x, attention_scores


class FusionGate(nn.Module):

    def __init__(self, model_dim: int):
        super().__init__()

        self.raw_linear = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.hidden_linear = nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self, raw: Tensor, hidden: Tensor) -> Tensor:
        gate = torch.sigmoid(self.raw_linear(raw) + self.hidden_linear(hidden))

        x = gate * raw + (1 - gate) * hidden

        return x
