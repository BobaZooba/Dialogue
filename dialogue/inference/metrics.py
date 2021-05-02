from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Sequence

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score

from dialogue import io
from dialogue.utils import (parse_prefix,
                            parse_postftix,
                            get_non_eye_matrix,
                            convert_to_torch_tensor,
                            get_random_indices)


def parse_recall_parameters(parameters: Union[Sequence[int], str],
                            separator: str = ', ') -> Sequence[int]:

    correct_parameters = list()

    for parameter in parameters:
        if isinstance(parameter, str):
            k, c = parameter.split(separator)
            correct_parameters.append((int(k), int(c)))
        else:
            correct_parameters.append(parameter)

    correct_parameters = tuple(correct_parameters)

    return correct_parameters


def get_statistics(tensor: Tensor,
                   quantile: float = 0.95,
                   prefix: Optional[str] = None,
                   postfix: Optional[str] = None) -> Dict[str, float]:

    tensor = tensor.detach().float()

    prefix = parse_prefix(prefix)
    postfix = parse_postftix(postfix)

    statistics = {
        f'{prefix}max{postfix}': tensor.quantile(quantile).item(),
        f'{prefix}mean{postfix}': tensor.mean().item(),
        f'{prefix}min{postfix}': tensor.quantile(1. - quantile).item(),
    }

    return statistics


def metrics_from_similarity_matrix(similarity_matrix: Tensor,
                                   quantile: float = 0.95,
                                   prefix: Optional[str] = None,
                                   postfix: Optional[str] = None) -> Dict[str, float]:

    similarity_matrix = similarity_matrix.detach().float()

    prefix = parse_prefix(prefix)
    postfix = parse_postftix(postfix)

    positive_similarities = similarity_matrix.diag()
    negative_similarities = get_non_eye_matrix(similarity_matrix)

    reverse_quantile = 1. - quantile

    metrics = {
        f'{prefix}max_positive_similarity{postfix}': positive_similarities.quantile(quantile).item(),
        f'{prefix}mean_positive_similarity{postfix}': positive_similarities.mean().item(),
        f'{prefix}min_positive_similarity{postfix}': positive_similarities.quantile(reverse_quantile).item(),
        f'{prefix}max_negative_similarity{postfix}': negative_similarities.quantile(quantile).item(),
        f'{prefix}mean_negative_similarity{postfix}': negative_similarities.mean().item(),
        f'{prefix}min_negative_similarity{postfix}': negative_similarities.quantile(reverse_quantile).item()
    }

    return metrics


def get_similarities_by_embeddings(
        contexts_embeddings: Union[np.ndarray, Tensor],
        responses_embeddings: Union[np.ndarray, Tensor],
        additional_negative_responses_embeddings: Union[np.ndarray, Tensor, None] = None,
        n_negative_candidates: int = 9):

    contexts_embeddings = convert_to_torch_tensor(contexts_embeddings)
    responses_embeddings = convert_to_torch_tensor(responses_embeddings)

    if additional_negative_responses_embeddings is not None:
        responses_embeddings = torch.cat((
            responses_embeddings,
            convert_to_torch_tensor(additional_negative_responses_embeddings)
        ))

    negative_indices = get_random_indices(batch_size=responses_embeddings.size(0))[:, :n_negative_candidates]

    negative_candidates = responses_embeddings[negative_indices]

    candidates = torch.cat((responses_embeddings[:contexts_embeddings.size(0)].unsqueeze(dim=1),
                            negative_candidates), dim=1)

    similarities = torch.bmm(contexts_embeddings.unsqueeze(dim=1),
                             candidates.transpose(1, 2)).squeeze(dim=1)

    return similarities


def recall_k_at_c(similarities: Tensor, k: int = 1, c: int = 10, true_index: int = 0):
    assert similarities.size(-1) >= c

    sorted_candidates = similarities[:, :c].argsort(descending=True)

    metric_value = (sorted_candidates[:, :k] == true_index).sum().float() / sorted_candidates.size(0)

    return metric_value.item()


def recall_k_at_c_by_embeddings(
        contexts_embeddings: Union[np.ndarray, Tensor],
        responses_embeddings: Union[np.ndarray, Tensor],
        k: int = 1,
        c: int = 10):

    similarities = get_similarities_by_embeddings(contexts_embeddings=contexts_embeddings,
                                                  responses_embeddings=responses_embeddings,
                                                  n_negative_candidates=c - 1)

    metric_value = recall_k_at_c(similarities=similarities, k=k, c=c)

    return metric_value


def get_recalls_by_embeddings(
        contexts_embeddings: Union[np.ndarray, Tensor],
        responses_embeddings: Union[np.ndarray, Tensor],
        parameters: Sequence[Sequence[int]] = ((1, 2), (1, 5), (1, 10))) -> Dict[str, float]:

    max_c = 0

    for k, c in parameters:
        if c > max_c:
            max_c = c

    max_c = min(max_c, responses_embeddings.size(0) + 1)

    similarities = get_similarities_by_embeddings(contexts_embeddings=contexts_embeddings,
                                                  responses_embeddings=responses_embeddings,
                                                  n_negative_candidates=max_c - 1)

    metrics = {(k, c): recall_k_at_c(similarities=similarities, k=k, c=c)
               for k, c in parameters if c <= max_c}

    return metrics


def compute_recall(context_embeddings: Tensor,
                   response_embeddings: Tensor,
                   parameters: Union[Sequence[int], str] = ((1, 2), (1, 5)),
                   prefix: Optional[str] = None,
                   postfix: Optional[str] = None) -> Dict[str, float]:

    prefix = parse_prefix(prefix)
    postfix = parse_postftix(postfix)

    metrics = get_recalls_by_embeddings(context_embeddings,
                                        response_embeddings,
                                        parameters=parameters)

    metrics = {f'{prefix}recall{k}@{c}{postfix}': value for (k, c), value in metrics.items()}

    return metrics


def compute_metrics_from_loss(loss: io.Batch,
                              quantile: float = 0.95,
                              prefix: Optional[str] = None,
                              postfix: Optional[str] = None) -> Dict[str, float]:

    prefix = parse_prefix(prefix)
    postfix = parse_postftix(postfix)

    metrics = dict()

    with torch.no_grad():

        if io.TYPES.similarity_matrix in loss:
            metrics.update(metrics_from_similarity_matrix(
                similarity_matrix=loss[io.TYPES.similarity_matrix],
                quantile=quantile,
                prefix=f'{prefix}sm',
                postfix=postfix
            ))

        if io.TYPES.context_similarity_matrix in loss:
            metrics.update(get_statistics(
                tensor=loss[io.TYPES.context_similarity_matrix],
                quantile=quantile,
                prefix=f'{prefix}context',
                postfix=f'similarity{postfix}'
            ))

        if io.TYPES.response_similarity_matrix in loss:
            metrics.update(get_statistics(
                tensor=loss[io.TYPES.response_similarity_matrix],
                quantile=quantile,
                prefix=f'{prefix}response',
                postfix=f'similarity{postfix}'
            ))

        if io.TYPES.positive_similarity_matrix in loss:
            metrics.update(get_statistics(tensor=loss[io.TYPES.positive_similarity_matrix],
                                          quantile=quantile,
                                          prefix=f'{prefix}positive',
                                          postfix=f'similarity{postfix}'))

        if io.TYPES.negative_similarity_matrix in loss:
            metrics.update(get_statistics(tensor=loss[io.TYPES.negative_similarity_matrix],
                                          quantile=quantile,
                                          prefix=f'{prefix}negative',
                                          postfix=f'similarity{postfix}'))

    return metrics


def compute_roc_auc(predictions: Tensor,
                    targets: Tensor,
                    prefix: Optional[str] = None,
                    postfix: Optional[str] = None) -> Dict[str, Union[int, float]]:

    prefix = parse_prefix(prefix)
    postfix = parse_postftix(postfix)

    roc_auc = roc_auc_score(y_true=targets.detach().cpu().numpy(),
                            y_score=predictions.detach().cpu().numpy())

    output = {
        f'{prefix}roc_auc{postfix}': roc_auc
    }

    return output


def compute_metrics_from_targets(targets: io.Batch,
                                 predictions: Dict[str, Union[float, Tensor, io.Batch]],
                                 prefix: Optional[str] = None,
                                 postfix: Optional[str] = None) -> Dict[str, Union[int, float]]:

    prefix = parse_prefix(prefix)
    postfix = parse_postftix(postfix)

    metrics = {}

    if io.TYPES.relevance_target in targets:
        similarity = (predictions[io.TYPES.context_embeddings] * predictions[io.TYPES.response_embeddings]).sum(dim=1)
        metrics.update(compute_roc_auc(predictions=similarity,
                                       targets=targets[io.TYPES.relevance_target],
                                       prefix=prefix,
                                       postfix=postfix))

    return metrics


class BaseRecall(ABC):

    def __init__(self,
                 parameters: Sequence[Sequence[int]] = ((1, 2), (1, 5), (1, 10), (3, 10), (5, 10))):

        self.parameters = parameters
        self.max_negatives = max([c for k, c in self.parameters]) - 1

    def get_metrics_by_similarities(self, similarities: Union[np.ndarray, Tensor]):

        similarities = convert_to_torch_tensor(similarities)

        metrics = {(k, c): recall_k_at_c(similarities=similarities, k=k, c=c)
                   for k, c in self.parameters}

        return metrics

    @abstractmethod
    def calculate(self,
                  contexts: List[str],
                  responses: List[str],
                  additional_negative_responses: Optional[List[str]] = None):
        ...


class BiEncoderRecall(BaseRecall):

    def __init__(self,
                 context_encoder,
                 response_encoder,
                 parameters: Sequence[Sequence[int]] = ((1, 2), (1, 5), (1, 10), (3, 10), (5, 10))):
        super().__init__(parameters=parameters)

        self.context_encoder = context_encoder
        self.response_encoder = response_encoder

    def calculate(self,
                  contexts: List[str],
                  responses: List[str],
                  additional_negative_responses: Optional[List[str]] = None):

        contexts_embeddings = self.context_encoder(contexts)
        responses_embeddings = self.response_encoder(responses)

        if additional_negative_responses is not None:
            additional_responses_embeddings = self.response_encoder(additional_negative_responses)
        else:
            additional_responses_embeddings = None

        similarities = get_similarities_by_embeddings(
            contexts_embeddings=contexts_embeddings,
            responses_embeddings=responses_embeddings,
            additional_negative_responses_embeddings=additional_responses_embeddings,
            n_negative_candidates=self.max_negatives
        )

        metrics = self.get_metrics_by_similarities(similarities=similarities)

        return metrics


class CrossEncoderRecall(BaseRecall):

    def __init__(self,
                 encoder,
                 parameters: Sequence[Sequence[int]] = ((1, 2), (1, 5), (1, 10), (3, 10), (5, 10))):
        super().__init__(parameters=parameters)

        self.encoder = encoder

    def calculate(self,
                  contexts: List[str],
                  responses: List[str],
                  additional_negative_responses: Optional[List[str]] = None):

        if additional_negative_responses is not None:
            negative_texts = np.array(responses + additional_negative_responses)
        else:
            negative_texts = np.array(responses)

        negative_indices = np.random.randint(low=0, high=negative_texts.shape[0],
                                             size=(len(contexts), self.max_negatives))

        positive_indices = np.expand_dims(np.arange(len(responses)), 1)
        candidates = np.concatenate((positive_indices, negative_indices), axis=1)

        pairs = list()

        for n_sample in range(candidates.shape[0]):
            for n_candidate in range(candidates.shape[1]):
                pairs.append((contexts[n_sample], str(negative_texts[candidates[n_sample, n_candidate]])))

        similarities = self.encoder(pairs)

        similarities = similarities.reshape((len(contexts), self.max_negatives + 1))

        metrics = self.get_metrics_by_similarities(similarities=similarities)

        return metrics
