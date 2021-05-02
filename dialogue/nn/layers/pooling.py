from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from dialogue.nn.layers import common


class GlobalMaskedPooling(nn.Module):

    POOLING_TYPES = ('mean', 'max')

    def __init__(self,
                 pooling_type: str = 'mean',
                 dim: int = 1,
                 normalize: bool = False,
                 length_scaling: bool = True,
                 scaling_square_root: bool = True):
        super().__init__()

        self.pooling_type = pooling_type
        self.dim = dim

        self.normalize = normalize
        self.length_scaling = length_scaling
        self.scaling_square_root = scaling_square_root

        if self.pooling_type == 'max':
            self.mask_value = -float('inf')
        else:
            self.mask_value = 0.

        if self.pooling_type not in self.POOLING_TYPES:
            raise ValueError(f'Available types: {", ".join(self.POOLING_TYPES)}')

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        lengths = pad_mask.sum(self.dim).float()

        x = common.embedding_masking(x=x, pad_mask=pad_mask, value=self.mask_value)

        if self.pooling_type == 'mean':
            scaling = x.size(self.dim) / lengths
        else:
            scaling = torch.ones(x.size(0))

        if self.length_scaling:
            lengths_factor = lengths
            if self.scaling_square_root:
                lengths_factor = lengths_factor ** 0.5
            scaling /= lengths_factor

        scaling = scaling.masked_fill(lengths == 0, 1.).unsqueeze(-1)

        if self.pooling_type == 'mean':
            x = x.mean(self.dim)
        else:
            x, _ = x.max(self.dim)

        x *= scaling

        if self.normalize:
            x = F.normalize(x)

        return x

    def extra_repr(self) -> str:

        description = [
            f'pooling_type="{self.pooling_type}"',
            f'normalize={self.normalize}',
            f'length_scaling={self.length_scaling}',
            f'scaling_square_root={self.scaling_square_root}',
        ]

        description = ',\n'.join(description)

        return description


class AttentionPooling(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 inner_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        inner_dim = model_dim if inner_dim is None else inner_dim

        self.key_projection = nn.Linear(in_features=model_dim, out_features=inner_dim)
        self.value_projection = nn.Linear(in_features=model_dim, out_features=inner_dim)

        self.pooling_projection = nn.Linear(in_features=inner_dim, out_features=num_heads, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.scaling: float = inner_dim ** 0.5
        self.output_dim: int = inner_dim * num_heads

    def forward(self, x, pad_mask):

        key = self.key_projection(x)
        value = self.value_projection(x)

        key /= self.scaling

        attention_scores = self.pooling_projection(key).transpose(1, 2)

        attention_scores = attention_scores.masked_fill(~(pad_mask.bool()).unsqueeze(1), -float('inf'))

        parameters_type = next(self.parameters()).dtype

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        attention_scores = torch.softmax(attention_scores.float(), dim=-1, dtype=tensor_type)

        attention_scores = self.dropout(attention_scores)

        x = torch.bmm(attention_scores, value.type(tensor_type)).type(parameters_type)

        x = x.view(x.size(0), -1)

        return x
