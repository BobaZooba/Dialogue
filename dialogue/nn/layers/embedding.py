from typing import Optional

import torch
from torch import nn, Tensor

from dialogue.nn.layers import normalization, common


class FactorizedEmbedding(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 token_embedding_dim: int,
                 pad_index: int = 0):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=token_embedding_dim,
                                            padding_idx=pad_index)

        self.projection = nn.Linear(in_features=token_embedding_dim, out_features=embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding_layer(x)
        emb = self.projection(emb)
        return emb


class TransformerEmbedding(nn.Module):

    def __init__(self,
                 model_dim: int,
                 vocab_size: int,
                 n_positions: int = 0,
                 embedding_dim: Optional[int] = None,
                 n_segments: int = 3,
                 dropout: float = 0.1,
                 normalization_type: str = 'rms',
                 pad_index: int = 0):
        super().__init__()

        self.pad_index = pad_index

        self.scaling = model_dim ** 0.5

        if embedding_dim is not None and embedding_dim != model_dim:
            self.token_embedding = FactorizedEmbedding(num_embeddings=vocab_size,
                                                       embedding_dim=model_dim,
                                                       token_embedding_dim=embedding_dim,
                                                       pad_index=self.pad_index)
        else:
            self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=model_dim,
                                                padding_idx=self.pad_index)

        if n_segments > 1:
            self.segment_embedding = nn.Embedding(num_embeddings=n_segments + 1,
                                                  embedding_dim=model_dim,
                                                  padding_idx=self.pad_index)
        else:
            self.segment_embedding = None

        if n_positions > 1:
            self.positional_embedding = nn.Embedding(num_embeddings=n_positions + 1,
                                                     embedding_dim=model_dim,
                                                     padding_idx=self.pad_index)
        else:
            self.positional_embedding = None

        self.normalization = normalization.get_normalization(normalized_shape=model_dim,
                                                             normalization_type=normalization_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                sequence_indices: Tensor,
                pad_mask: Optional[Tensor] = None,
                position_indices: Optional[Tensor] = None,
                segment_indices: Optional[Tensor] = None) -> Tensor:
        """
        :param sequence_indices: [batch_size, sequence_length]
        :param pad_mask: [batch_size, sequence_length]
        :param position_indices: [batch_size, sequence_length]
        :param segment_indices: [batch_size, sequence_length]
        :return: [batch_size, sequence_length, model_dim]
        """

        embeddings = self.token_embedding(sequence_indices) * self.scaling

        if self.positional_embedding is not None:

            if position_indices is None:
                position_indices = torch.arange(1, sequence_indices.size(1) + 1)
                position_indices = position_indices.unsqueeze(0).repeat(sequence_indices.size(0), 1)
                position_indices = position_indices.to(sequence_indices.device)

            position_embeddings = self.positional_embedding(position_indices)
            embeddings += position_embeddings

        if self.segment_embedding is not None:

            if segment_indices is None:
                segment_indices = torch.ones_like(sequence_indices).to(sequence_indices.device)

            segment_embeddings = self.segment_embedding(segment_indices)
            embeddings += segment_embeddings

        embeddings = self.dropout(self.normalization(embeddings))

        embeddings = common.embedding_masking(embeddings, pad_mask)

        return embeddings
