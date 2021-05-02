import torch
from dialogue import io
from torch import nn, Tensor
from torch.nn.functional import normalize


class TextInput(nn.Module):

    def __init__(self, pad_index: int = 0):
        super().__init__()

        self.pad_index = pad_index

    def get_pad_mask(self, sequence_indices: Tensor) -> Tensor:
        return (sequence_indices != self.pad_index).long()

    def forward(self, sequence_indices: Tensor) -> io.Batch:

        pad_mask = self.get_pad_mask(sequence_indices).to(sequence_indices.device)

        position_indices = torch.arange(1, sequence_indices.size(1) + 1).to(sequence_indices.device)
        position_indices = position_indices.unsqueeze(0).repeat(sequence_indices.size(0), 1)
        position_indices *= pad_mask

        segment_indices = torch.zeros_like(sequence_indices).to(sequence_indices.device)
        segment_indices *= segment_indices

        batch = {
            io.TYPES.sequence_indices: sequence_indices,
            io.TYPES.pad_mask: pad_mask,
            io.TYPES.position_indices: position_indices,
            io.TYPES.segment_indices: segment_indices
        }

        return batch


class Normalization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, batch: io.Batch) -> Tensor:
        return normalize(batch[io.TYPES.head])


class KeyedNormalization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, batch: io.Batch) -> io.Batch:

        batch[io.TYPES.head] = normalize(batch[io.TYPES.head])

        return batch
