from typing import Optional

import torch
from torch import nn

from dialogue import io
from dialogue.nn.layers import pooling, normalization


class GlobalPooling(nn.Module):

    def __init__(self,
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True):
        super().__init__()

        self.pooling = pooling.GlobalMaskedPooling(pooling_type=pooling_type,
                                                   length_scaling=length_scaling,
                                                   scaling_square_root=scaling_square_root)

    def forward(self, batch: io.Batch) -> io.Batch:

        x = self.pooling(batch[io.TYPES.backbone], batch[io.TYPES.pad_mask])

        batch.update({
            io.TYPES.aggregation: x
        })

        return batch


class AttentionAggregation(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 inner_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True):
        super().__init__()

        self.pooling = pooling.GlobalMaskedPooling(pooling_type=pooling_type,
                                                   length_scaling=length_scaling,
                                                   scaling_square_root=scaling_square_root)

        self.attention_pooling = pooling.AttentionPooling(model_dim=model_dim,
                                                          num_heads=num_heads,
                                                          inner_dim=inner_dim,
                                                          dropout=dropout)

    def forward(self, batch: io.Batch) -> io.Batch:

        mean_pooled = self.pooling(batch[io.TYPES.backbone], batch[io.TYPES.pad_mask])
        attention_pooled = self.attention_pooling(batch[io.TYPES.backbone], batch[io.TYPES.pad_mask]).squeeze(dim=1)

        x = torch.cat((mean_pooled, attention_pooled), dim=-1)

        batch.update({
            io.TYPES.aggregation: x
        })

        return batch


class ResidualAttentionAggregation(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 inner_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 normalization_type: str = 'rms',
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True):
        super().__init__()

        self.pooling = pooling.GlobalMaskedPooling(pooling_type=pooling_type,
                                                   length_scaling=length_scaling,
                                                   scaling_square_root=scaling_square_root)

        self.normalization_layer = normalization.get_normalization(normalized_shape=model_dim,
                                                                   normalization_type=normalization_type)

        self.attention_pooling = pooling.AttentionPooling(model_dim=model_dim,
                                                          num_heads=num_heads,
                                                          inner_dim=inner_dim,
                                                          dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        self.output_projection = nn.Linear(in_features=self.attention_pooling.output_dim,
                                           out_features=model_dim)

    def forward(self, batch: io.Batch) -> io.Batch:

        mean_pooled = self.pooling(batch[io.TYPES.backbone], batch[io.TYPES.pad_mask])

        attention_pooled = self.attention_pooling(self.normalization_layer(batch[io.TYPES.backbone]),
                                                  batch[io.TYPES.pad_mask])

        attention_pooled = self.output_projection(attention_pooled.squeeze(dim=1))

        attention_pooled = self.dropout(attention_pooled) + mean_pooled

        batch.update({
            io.TYPES.aggregation: attention_pooled
        })

        return batch
