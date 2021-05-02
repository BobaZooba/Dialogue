from dataclasses import dataclass
from typing import Dict, Optional
from typing import List, Sequence, Union, Tuple

import torch
from torch import Tensor


@dataclass
class Types:

    phrase: str = 'phrase'
    response: str = 'response'
    context: str = 'context'
    candidates: str = 'candidates'
    target: str = 'target'
    sequence_indices: str = 'sequence_indices'
    pad_mask: str = 'pad_mask'
    position_indices: str = 'position_indices'
    segment_indices: str = 'segment_indices'
    backbone: str = 'backbone'
    aggregation: str = 'aggregation'
    head: str = 'head'
    encodings: str = 'encodings'
    embedding: str = 'embedding'
    pseudo_head: str = 'pseudo_head'
    loss: str = 'loss'
    context_embeddings: str = 'context_embeddings'
    response_embeddings: str = 'response_embeddings'
    similarity: str = 'similarity'
    similarity_matrix: str = 'similarity_matrix'
    positive_similarity_matrix: str = 'positive_similarity_matrix'
    negative_similarity_matrix: str = 'negative_similarity_matrix'
    train_loss: str = 'train_loss'
    valid_loss: str = 'valid_loss'
    metrics_from_loss: str = 'metrics_from_loss'
    train: str = 'train'
    valid: str = 'valid'
    relevance: str = 'relevance'
    relevance_target: str = 'relevance_target'
    relevance_score: str = 'relevance_score'
    dataset_key: str = 'dataset_key'
    pairs: str = 'pairs'
    context_similarity_matrix: str = 'context_similarity_matrix'
    response_similarity_matrix: str = 'response_similarity_matrix'
    not_duplicate: str = 'not_duplicate'


TYPES = Types()

RawTextData = List[Dict[str, Union[str, List[str]]]]
TextBatch = Dict[str, Union[str, int, float, Sequence[str]]]
Batch = Dict[str, Tensor]
SequenceBatch = Sequence[Batch]
