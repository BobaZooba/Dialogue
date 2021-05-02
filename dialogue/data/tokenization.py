import math
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Sequence

import sentencepiece as spm
import transformers
import youtokentome as yttm


class BaseTokenizer(ABC):

    def __call__(self,
                 texts: Union[str, Sequence[str], Sequence[Tuple[str, str]]],
                 batch_size: int = 512) -> Sequence[Sequence[int]]:

        if isinstance(texts, str):
            texts = [texts]

        tokenized = list()

        for i_batch in range(math.ceil(len(texts) / batch_size)):
            batch = texts[i_batch * batch_size:(i_batch + 1) * batch_size]
            tokenized.extend(self.encode(texts=batch))

        lengths = [len(sample) for sample in tokenized]

        if len(set(lengths)) > 1:
            batch_max_length = max(lengths)

            for i_sample in range(len(tokenized)):
                tokenized[i_sample] += [self.pad_index] * (batch_max_length - len(tokenized[i_sample]))

        return tokenized

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        ...

    @property
    @abstractmethod
    def pad_index(self) -> int:
        ...

    @property
    @abstractmethod
    def max_length(self) -> int:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def eval(self):
        ...


class Tokenizer(BaseTokenizer):

    def __init__(self,
                 tokenizer_path: str,
                 add_special_tokens: bool = True,
                 padding: bool = True,
                 max_length: int = 64,
                 pad_index: int = 0,
                 unk_index: int = 1,
                 bos_index: int = 2,
                 eos_index: int = 3):

        self.tokenizer_path = tokenizer_path

        self.add_special_tokens = add_special_tokens
        self.padding = padding

        self._max_length = max_length

        self._pad_index = pad_index
        self.unk_index = unk_index
        self.bos_index = bos_index
        self.eos_index = eos_index

        self.is_training = False

    @property
    def pad_index(self):
        return self._pad_index

    @property
    def max_length(self):
        return self._max_length

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def pad_sequence(self, sequence: Sequence[int], max_length: int) -> Sequence[int]:

        sequence += [self.pad_index] * (max_length - len(sequence))

        return sequence

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:

        if isinstance(texts, tuple):
            texts = list(texts)

        tokenized = self.tokenize(texts=texts)

        if self.add_special_tokens:
            max_length = self.max_length - 2
        else:
            max_length = self.max_length

        batch_max_length = 0

        for i_sample in range(len(tokenized)):

            sample = tokenized[i_sample][:max_length]

            if self.add_special_tokens:
                sample = [self.bos_index] + sample + [self.eos_index]

            batch_max_length = max(batch_max_length, len(sample))
            tokenized[i_sample] = sample

        if self.padding:
            tokenized = [self.pad_sequence(sequence=sample, max_length=batch_max_length)
                         for sample in tokenized]

        return tokenized

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def tokenize(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        ...

    @abstractmethod
    def decode(self, tokenized_texts: Sequence[Sequence[int]]) -> Sequence[str]:
        ...


class YTTMTokenizer(Tokenizer):

    def __init__(self,
                 tokenizer_path: str,
                 dropout: float = 0.,
                 add_special_tokens: bool = True,
                 skip_special_tokens_for_decode: bool = True,
                 padding: bool = True,
                 max_length: int = 64,
                 pad_index: int = 0,
                 unk_index: int = 1,
                 bos_index: int = 2,
                 eos_index: int = 3):

        super().__init__(
            tokenizer_path=tokenizer_path,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            pad_index=pad_index,
            unk_index=unk_index,
            bos_index=bos_index,
            eos_index=eos_index
        )

        self.tokenizer = yttm.BPE(model=self.tokenizer_path)
        self.dropout = dropout

        self.skip_special_tokens_for_decode = skip_special_tokens_for_decode

        self.decode_ignore_index = [
            self.pad_index,
            self.unk_index,
            self.bos_index,
            self.eos_index
        ]

    def __len__(self):
        return self.tokenizer.vocab_size()

    def tokenize(self, texts: List[str]) -> List[List[int]]:

        dropout = self.dropout if self.is_training else 0.

        tokenized = self.tokenizer.encode(sentences=texts,
                                          dropout_prob=dropout)

        return tokenized

    def decode(self, tokenized_texts: List[List[int]]) -> List[str]:

        if self.skip_special_tokens_for_decode:
            output = self.tokenizer.decode(ids=tokenized_texts, ignore_ids=self.decode_ignore_index)
        else:
            output = self.tokenizer.decode(ids=tokenized_texts)

        return output


class SentencePieceTokenizer(Tokenizer):

    def __init__(self,
                 tokenizer_path: str,
                 add_special_tokens: bool = True,
                 padding: bool = True,
                 max_length: int = 64,
                 pad_index: int = 0,
                 unk_index: int = 1,
                 bos_index: int = 2,
                 eos_index: int = 3):

        super().__init__(
            tokenizer_path=tokenizer_path,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            pad_index=pad_index,
            unk_index=unk_index,
            bos_index=bos_index,
            eos_index=eos_index
        )

        self.sentence_piece = spm.SentencePieceProcessor(model_file=self.tokenizer_path)

    def __len__(self):
        return self.sentence_piece.vocab_size()

    def tokenize(self, texts: List[str]) -> List[List[int]]:
        return self.sentence_piece.encode(texts)

    def decode(self, tokenized_texts: List[List[int]]) -> List[str]:
        return self.sentence_piece.decode(tokenized_texts)


class HuggingFaceTokenizer(BaseTokenizer):

    def __init__(self,
                 model_name: str,
                 add_special_tokens: bool = True,
                 padding: bool = True,
                 max_length: int = 64):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        self.add_special_tokens = add_special_tokens
        self.padding = padding

        self._max_length = max_length

    @property
    def pad_index(self):
        return self.tokenizer.pad_token_id

    @property
    def max_length(self):
        return self._max_length

    def __len__(self):
        return len(self.tokenizer)

    def train(self):
        ...

    def eval(self):
        ...

    def encode(self, texts: List[str]) -> List[List[int]]:

        tokenized = self.tokenizer.batch_encode_plus(texts,
                                                     add_special_tokens=self.add_special_tokens,
                                                     truncation=True,
                                                     max_length=self.max_length,
                                                     padding=self.padding)['input_ids']

        return tokenized

    def decode(self,
               tokenized_texts: List[List[int]],
               skip_special_tokens: bool = True) -> List[str]:

        output = [self.tokenizer.decode(token_ids=sample,
                                        skip_special_tokens=skip_special_tokens)
                  for sample in tokenized_texts]

        return output
