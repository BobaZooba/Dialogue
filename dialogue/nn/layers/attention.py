from abc import ABC
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from dialogue.nn.layers import activation


class RelativeAttentionPositions(nn.Module):

    def __init__(self,
                 head_dim: int,
                 max_relative_position: int,
                 num_heads: Optional[int] = None,
                 use_bias: bool = True):
        super().__init__()

        self.max_relative_position = max_relative_position
        self.num_heads = num_heads
        self.use_bias = use_bias

        self.relative_positions_keys = nn.Embedding(num_embeddings=self.max_relative_position * 2 + 1,
                                                    embedding_dim=head_dim)

        if self.use_bias and self.num_heads is not None:
            self.bias = nn.Parameter(torch.rand(1, self.num_heads, 1, 1))
            nn.init.xavier_uniform_(self.bias)

        self.relative_positions_values = nn.Embedding(num_embeddings=self.max_relative_position * 2 + 1,
                                                      embedding_dim=head_dim)

    def _generate_relative_positions_embeddings(self,
                                                length: int,
                                                is_key: bool = True) -> Tensor:
        """
        :param length: sequence_length
        :param is_key: use key positions
        :return relative_position_embeddings: [sequence_length, sequence_length, head_dim]
        """

        if is_key:
            embedding_layer = self.relative_positions_keys
        else:
            embedding_layer = self.relative_positions_values

        range_vector = torch.arange(length)
        distance_matrix = range_vector[None, :] - range_vector[:, None]

        distance_matrix_clipped = torch.clamp(distance_matrix,
                                              -self.max_relative_position,
                                              self.max_relative_position)

        final_matrix = (distance_matrix_clipped + self.max_relative_position)
        final_matrix = final_matrix.long().to(embedding_layer.weight.device)

        relative_positions_embeddings = embedding_layer(final_matrix)

        return relative_positions_embeddings

    def forward(self,
                tensor: Tensor,
                is_key: bool = True,
                transpose: bool = False) -> Tensor:
        """
        :param tensor: [batch_size, num_heads, sequence_length, head_dim or sequence_length]
        :param is_key: use key positions
        :param transpose:
        :return:
        """

        parameters_type = next(self.parameters()).dtype

        batch_size, num_heads, sequence_length, fourth_dim = tensor.size()

        relative_position_embeddings = self._generate_relative_positions_embeddings(length=sequence_length,
                                                                                    is_key=is_key)

        if transpose:
            relative_position_embeddings = relative_position_embeddings.transpose(-1, -2)

        tensor = tensor.permute(2, 0, 1, 3)
        tensor = tensor.reshape(sequence_length, num_heads * batch_size, -1)

        relative_attention_scores = torch.matmul(tensor,
                                                 relative_position_embeddings.type(tensor.dtype)).type(parameters_type)
        relative_attention_scores = relative_attention_scores.view(sequence_length, batch_size, num_heads, -1)
        relative_attention_scores = relative_attention_scores.permute(1, 2, 0, 3)

        if is_key and self.use_bias:
            relative_attention_scores += self.bias

        return relative_attention_scores

    def extra_repr(self) -> str:
        return f'(keys_bias={self.use_bias})'


class BaseAttention(nn.Module, ABC):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_bias: bool = False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias

        if head_dim is None:
            self.head_dim = model_dim // num_heads
            self.layer_dim = model_dim
        else:
            self.head_dim = head_dim
            self.layer_dim = self.head_dim * self.num_heads

        self.scaling = self.head_dim ** 0.5

    def _split_heads(self,
                     embeddings: Tensor,
                     batch_size: int,
                     sequence_length: int) -> Tensor:
        """
        From [batch_size * self.num_heads, sequence_length, sequence_length]
        To [batch_size, self.num_heads, sequence_length, sequence_length]
        """
        return embeddings.view(batch_size, self.num_heads, sequence_length, sequence_length)

    def _join_heads(self,
                    embeddings: Tensor,
                    batch_size: int,
                    sequence_length: int) -> Tensor:
        """
        From [batch_size, self.num_heads, sequence_len, sequence_len]
        To [batch_size * self.num_heads, sequence_len, sequence_len]
        """
        return embeddings.view(batch_size * self.num_heads, sequence_length, sequence_length)


class MultiHeadSelfAttention(BaseAttention):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 8,
                 use_bias_positions: bool = True):
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_bias=use_bias
        )

        self.in_projection = nn.Linear(self.model_dim, 3 * self.layer_dim, bias=self.use_bias)

        self.out_projection = nn.Linear(self.layer_dim, self.model_dim, bias=self.use_bias)

        self.dropout_layer = nn.Dropout(self.dropout)

        if use_relative_positions:
            self.relative_positions = RelativeAttentionPositions(head_dim=self.head_dim,
                                                                 max_relative_position=max_relative_position,
                                                                 num_heads=self.num_heads,
                                                                 use_bias=use_bias_positions)
        else:
            self.relative_positions = None

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :param need_weights: bool
        :return: [batch_size, sequence_length, model_dim]
        """

        x = x.transpose(0, 1)

        sequence_length, batch_size, model_dim = x.size()

        query, key, value = self.in_projection(x).chunk(3, dim=-1)

        # [batch_size * self.num_heads, sequence_len, self.head_dim]
        query = query.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)
        key = key.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)
        value = value.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        query /= self.scaling

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = torch.bmm(query, key.transpose(1, 2))

        if self.num_heads > 1:
            # [batch_size, self.num_heads, sequence_length, sequence_length]
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        if self.relative_positions is not None:
            query = query.transpose(0, 1).view(sequence_length, batch_size, self.num_heads, self.head_dim)
            query = query.permute(1, 2, 0, 3)
            relative_attention_scores_keys = self.relative_positions(query, is_key=True, transpose=True)
            attention_scores += relative_attention_scores_keys

        # fp16 compatibility
        parameters_type = next(self.parameters()).dtype

        if attention_mask is not None:
            if self.num_heads > 1:
                # [batch_size, sequence_length, sequence_length] -> [batch_size, 1, sequence_length, sequence_length]
                attention_mask = attention_mask.unsqueeze(1)

            attention_mask = attention_mask.to(dtype=parameters_type)
            attention_scores += attention_mask

        if pad_mask is not None:
            # [batch_size, sequence_length] -> [batch_size, 1, sequence_length]
            pad_mask = ~(pad_mask.bool()).unsqueeze(1)

            if self.num_heads > 1:
                # [batch_size, 1, sequence_length] -> [batch_size, 1, 1, sequence_length]
                pad_mask = pad_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(
                pad_mask,
                -float('inf'),
            )

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_length, sequence_length]
            attention_scores = self._join_heads(attention_scores, batch_size, sequence_length)

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = F.softmax(attention_scores.float(), dim=-1, dtype=tensor_type)

        attention_scores = self.dropout_layer(attention_scores)

        # attention_scores = [batch_size * self.num_heads, sequence_length, sequence_length]
        # value = [batch_size * self.num_heads, sequence_length, self.head_dim]
        # [batch_size * self.num_heads, sequence_length, self.head_dim]
        attention_output = torch.bmm(attention_scores, value.type(tensor_type)).type(parameters_type)

        if self.relative_positions is not None:
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)
            relative_attention_scores_values = self.relative_positions(attention_scores,
                                                                       is_key=False,
                                                                       transpose=False)
            attention_output = attention_output.view(batch_size, self.num_heads,
                                                     sequence_length, self.head_dim)
            attention_output += relative_attention_scores_values
            attention_output = attention_output.view(batch_size * self.num_heads,
                                                     sequence_length, self.head_dim)

        # [sequence_length, batch_size, model_dim]
        attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_length,
                                                                              batch_size,
                                                                              self.layer_dim)

        attention_output = self.out_projection(attention_output)

        # for visualize attention scores
        if need_weights:
            if self.num_heads > 1 and len(attention_scores) == 3:
                # [batch_size, self.num_heads, sequence_length, sequence_length]
                attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        attention_output = attention_output.transpose(0, 1)

        return attention_scores, attention_output


class RandomSynthesizedMultiHeadSelfAttention(BaseAttention):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 max_sequence_length: int = 32,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 8,
                 use_bias_positions: bool = True):
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_bias=use_bias
        )

        self.attention_scores = nn.Parameter(torch.rand(1,
                                                        self.num_heads,
                                                        max_sequence_length,
                                                        max_sequence_length))
        nn.init.xavier_uniform_(self.attention_scores)

        self.in_projection = nn.Linear(self.model_dim, self.layer_dim, bias=self.use_bias)

        self.out_projection = nn.Linear(self.layer_dim, self.model_dim, bias=self.use_bias)

        self.dropout_layer = nn.Dropout(dropout)

        if use_relative_positions:
            self.relative_positions = RelativeAttentionPositions(head_dim=self.head_dim,
                                                                 max_relative_position=max_relative_position,
                                                                 num_heads=self.num_heads,
                                                                 use_bias=use_bias_positions)
        else:
            self.relative_positions = None

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :param need_weights: bool
        :return: [batch_size, sequence_length, model_dim]
        """

        x = x.transpose(0, 1)

        sequence_length, batch_size, _ = x.size()

        value = self.in_projection(x)

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_length, self.head_dim]
            value = value.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)

        value = value.transpose(0, 1)

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = self.attention_scores.repeat(batch_size, 1, 1, 1)
        attention_scores = attention_scores[:, :, :sequence_length, :sequence_length]
        attention_scores = attention_scores.view(batch_size * self.num_heads,
                                                 sequence_length, sequence_length)

        if self.num_heads > 1:
            # [batch_size, self.num_heads, sequence_length, sequence_length]
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        # fp16 compatibility
        parameters_type = next(self.parameters()).dtype

        if attention_mask is not None:
            if self.num_heads > 1:
                # [batch_size, sequence_length, sequence_length] -> [batch_size, 1, sequence_length, sequence_length]
                attention_mask = attention_mask.unsqueeze(1)

            attention_mask = attention_mask.to(dtype=parameters_type)
            attention_scores += attention_mask

        if pad_mask is not None:
            # [batch_size, sequence_length] -> [batch_size, 1, sequence_length]
            pad_mask = ~(pad_mask.bool()).unsqueeze(1)

            if self.num_heads > 1:
                # [batch_size, 1, sequence_length] -> [batch_size, 1, 1, sequence_length]
                pad_mask = pad_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(
                pad_mask,
                -float('inf'),
            )

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_length, sequence_length]
            attention_scores = self._join_heads(attention_scores, batch_size, sequence_length)

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = F.softmax(attention_scores.float(), dim=-1, dtype=tensor_type)

        attention_scores = self.dropout_layer(attention_scores)

        # attention_scores = [batch_size * self.num_heads, sequence_length, sequence_length]
        # value = [batch_size * self.num_heads, sequence_length, self.head_dim]
        # [batch_size * self.num_heads, sequence_length, self.head_dim]
        attention_output = torch.bmm(attention_scores, value.type(tensor_type)).type(parameters_type)

        if self.relative_positions is not None:
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)
            relative_attention_scores_values = self.relative_positions(attention_scores,
                                                                       is_key=False)
            attention_output = attention_output.view(batch_size, self.num_heads,
                                                     sequence_length, self.head_dim)
            attention_output += relative_attention_scores_values
            attention_output = attention_output.view(batch_size * self.num_heads,
                                                     sequence_length, self.head_dim)

        # [sequence_length, batch_size, model_dim]
        attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_length,
                                                                              batch_size,
                                                                              self.layer_dim)

        attention_output = self.out_projection(attention_output)

        # for visualize attention scores
        if need_weights:
            if self.num_heads > 1 and len(attention_scores) == 3:
                # [batch_size, self.num_heads, sequence_length, sequence_length]
                attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        attention_output = attention_output.transpose(0, 1)

        return attention_output, attention_scores


class GatedMultiHeadAttention(nn.Module):

    def __init__(self,
                 model_dim: int,
                 first_attention: nn.Module,
                 second_attention: nn.Module,
                 gate_activation_type: str = 'sigmoid'):
        super().__init__()

        self.gate_projection = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.activation = activation.get_activation(activation_type=gate_activation_type)
        self.first_attention = first_attention
        self.second_attention = second_attention

        if self.first_attention.relative_positions is not None:
            self.relative_positions = self.first_attention.relative_positions

            if self.second_attention.relative_positions is not None:
                self.second_attention.relative_positions = self.relative_positions

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                need_weights: bool = False) -> Tuple[Tensor,
                                                     Tuple[Optional[Tensor],
                                                           Optional[Tensor]]]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :param need_weights: bool
        :return: [batch_size, sequence_length, model_dim]
        """

        gate = self.activation(self.gate_projection(x))

        first_attention_output, first_attention_scores = self.first_attention(x,
                                                                              pad_mask,
                                                                              attention_mask,
                                                                              need_weights)

        second_attention_output, second_attention_scores = self.second_attention(x,
                                                                                 pad_mask,
                                                                                 attention_mask,
                                                                                 need_weights)

        attention_output = gate * first_attention_output + (1. - gate) * second_attention_output

        return attention_output, (first_attention_scores, second_attention_scores)
