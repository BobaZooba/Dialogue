from abc import ABC, abstractmethod
from typing import Any, Sequence, List

from torch import Tensor
import torch

from dialogue import io
from dialogue.data import tokenization


class Preparer(ABC):

    def __init__(self, tokenizer: tokenization.Tokenizer):
        self.tokenizer = tokenizer

    def get_pad_mask(self, sequence_indices: Tensor) -> Tensor:
        return (sequence_indices != self.tokenizer.pad_index).long()

    def __call__(self, batch: Sequence[Any]) -> io.Batch:
        return self.collate(batch=batch)

    @abstractmethod
    def collate(self, batch: Sequence[Any]) -> io.Batch:
        ...


class PairsPreparer(Preparer):

    def __init__(self, tokenizer: tokenization.Tokenizer):
        super().__init__(tokenizer=tokenizer)

    def collect_batch(self, texts: Sequence[str]) -> io.Batch:

        sequence_indices = Tensor(self.tokenizer(texts)).long()
        pad_mask = self.get_pad_mask(sequence_indices)

        position_indices = torch.arange(1, sequence_indices.size(1) + 1)
        position_indices = position_indices.unsqueeze(0).repeat(sequence_indices.size(0), 1)
        position_indices *= pad_mask

        segment_indices = torch.zeros_like(sequence_indices)
        segment_indices *= segment_indices

        batch = {
            io.TYPES.sequence_indices: sequence_indices,
            io.TYPES.pad_mask: pad_mask,
            io.TYPES.position_indices: position_indices,
            io.TYPES.segment_indices: segment_indices
        }

        return batch

    def collate(self, batch: Sequence[io.TextBatch]) -> io.SequenceBatch:

        phrases = list()
        responses = list()

        for sample in batch:
            phrases.append(sample[io.TYPES.phrase])
            responses.append(sample[io.TYPES.response])

        phrase_batch = self.collect_batch(texts=phrases)
        response_batch = self.collect_batch(texts=responses)

        return phrase_batch, response_batch


class ContextPairsPreparer(PairsPreparer):

    def __init__(self, tokenizer: tokenization.HuggingFaceTokenizer,
                 context_separator: str = '@', max_context: int = 4):
        super().__init__(tokenizer=tokenizer)

        self.context_index = self.tokenizer.tokenizer.vocab[context_separator]
        self.context_separator = context_separator + ' '
        self.max_context = max_context

    def collate(self, batch: Sequence[io.TextBatch]) -> io.SequenceBatch:

        phrases = list()
        responses = list()

        for sample in batch:
            context = sample[io.TYPES.context] + [sample[io.TYPES.phrase]]
            context = context[-self.max_context:]
            phrases.append(self.context_separator.join(context))
            responses.append(sample[io.TYPES.response])

        phrase_batch = self.collect_batch(texts=phrases)
        response_batch = self.collect_batch(texts=responses)

        return phrase_batch, response_batch
