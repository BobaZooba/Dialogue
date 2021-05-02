from abc import ABC
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from torch import nn, Tensor
from abc import abstractmethod

from dialogue import utils, io


class CosineMiner(nn.Module, ABC):
    SAMPLING_TYPES = ('random', 'semi_hard', 'hard')

    def __init__(self,
                 n_negatives: int = 1,
                 sampling_type: str = 'semi_hard',
                 normalize: bool = False,
                 multinomial_sampling: bool = False,
                 semi_hard_epsilon: float = 0.,
                 margin: Optional[float] = None):
        super().__init__()

        self.n_negatives = n_negatives
        self.sampling_type = sampling_type
        self.normalize = normalize
        self.multinomial_sampling = multinomial_sampling
        self.margin = margin
        self.semi_hard_epsilon = semi_hard_epsilon

        if self.sampling_type not in self.SAMPLING_TYPES:
            raise ValueError(f'Not available sampling_type. Available: {", ".join(self.SAMPLING_TYPES)}')

        self.diagonal_mask_value = torch.tensor([-10000.])
        self.upper_bound_mask_value = torch.tensor([-100.])
        self.lower_bound_mask_value = torch.tensor([-10.])

    def get_indices(self, similarity_matrix):

        if self.multinomial_sampling:
            distribution_similarity_matrix = torch.softmax(similarity_matrix, dim=-1)
            negative_indices = torch.multinomial(distribution_similarity_matrix, num_samples=self.n_negatives)
        else:
            negative_indices = similarity_matrix.argsort(descending=True)[:, :self.n_negatives]

        negative_weights = similarity_matrix.gather(dim=-1, index=negative_indices)

        negative_weights = torch.softmax(negative_weights, dim=-1)

        negative_indices = negative_indices.to(similarity_matrix.device)
        negative_weights = negative_weights.to(similarity_matrix.device)

        return negative_indices, negative_weights

    def get_similarity_matrix(self,
                              context: Tensor,
                              response: Tensor,
                              semi_hard: bool = True) -> Tensor:

        with torch.no_grad():

            if self.normalize:
                context = F.normalize(context)
                response = F.normalize(response)

            similarity_matrix = context @ response.t()

            diagonal_mask = torch.eye(context.size(0)).bool().to(context.device)
            response_sim_matrix = similarity_matrix.masked_select(diagonal_mask)

            if semi_hard:
                difference = response_sim_matrix.detach().unsqueeze(-1).repeat(1, similarity_matrix.size(-1))
                difference = difference - similarity_matrix + self.semi_hard_epsilon

            similarity_matrix = similarity_matrix.where(~diagonal_mask.bool(),
                                                        self.diagonal_mask_value.to(context.device))

            if semi_hard:
                similarity_matrix = similarity_matrix.where(difference > 0.,
                                                            self.upper_bound_mask_value.to(context.device))

                if self.margin is not None:
                    similarity_matrix = similarity_matrix.where(difference <= self.margin,
                                                                self.lower_bound_mask_value.to(context.device))

        return similarity_matrix

    def random_sampling(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        possible_indices = torch.arange(batch_size).unsqueeze(dim=0).repeat(batch_size, 1)
        mask = ~torch.eye(batch_size).bool()
        possible_indices = possible_indices.masked_select(mask).view(batch_size, batch_size - 1)
        random_indices = torch.randint(batch_size - 1, (batch_size, self.n_negatives))
        negative_indices = torch.gather(possible_indices, 1, random_indices)

        weights = torch.ones_like(negative_indices).float() / batch_size

        return negative_indices, weights

    def semi_hard_sampling(self,
                           context: Tensor,
                           response: Tensor) -> Tuple[Tensor, Tensor]:

        similarity_matrix = self.get_similarity_matrix(context=context, response=response, semi_hard=True)

        negative_indices, weights = self.get_indices(similarity_matrix)

        return negative_indices, weights

    def hard_sampling(self,
                      context: Tensor,
                      response: Tensor) -> Tuple[Tensor, Tensor]:

        similarity_matrix = self.get_similarity_matrix(context=context, response=response, semi_hard=False)

        negative_indices, weights = self.get_indices(similarity_matrix)

        return negative_indices, weights

    def sampling(self, context: Tensor, response: Tensor) -> Tuple[Tensor, Tensor]:

        if self.sampling_type == 'hard':
            negative_indices, weights = self.hard_sampling(context=context, response=response)
        elif self.sampling_type == 'semi_hard':
            negative_indices, weights = self.semi_hard_sampling(context=context, response=response)
        else:
            negative_indices, weights = self.random_sampling(batch_size=context.size(0))
            negative_indices = negative_indices.to(context.device)
            weights = weights.to(context.device)

        return negative_indices, weights


class CosineTripletLoss(nn.Module):

    def __init__(self,
                 margin: float = 0.05,
                 normalize: bool = True,
                 use_margin_for_sampling: bool = False,
                 sampling_type: str = 'semi_hard',
                 multinomial_sampling: bool = False,
                 semi_hard_epsilon: float = 0.):
        super().__init__()

        self.margin = margin
        self.normalize = normalize

        self.miner = CosineMiner(n_negatives=1,
                                 sampling_type=sampling_type,
                                 normalize=False,
                                 multinomial_sampling=multinomial_sampling,
                                 margin=self.margin if use_margin_for_sampling else None,
                                 semi_hard_epsilon=semi_hard_epsilon)

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        if self.normalize:
            context = F.normalize(context)
            response = F.normalize(response)

        response_sim_matrix = (context * response).sum(dim=1)

        negative_indices, _ = self.miner.sampling(context=context, response=response)
        negative_indices = negative_indices.squeeze(dim=1)

        negative = response[negative_indices]

        negative_sim_matrix = (context * negative).sum(dim=1)

        loss = torch.relu(self.margin - response_sim_matrix + negative_sim_matrix).mean()

        loss_output = {
            io.TYPES.loss: loss,
            io.Keys.positive_similarity_matrix: response_sim_matrix,
            io.Keys.negative_similarity_matrix: negative_sim_matrix,
        }

        return loss_output


class MultipleNegativeCosineTripletLoss(nn.Module):

    def __init__(self,
                 margin: float = 0.05,
                 normalize: bool = True,
                 use_centroid: bool = True,
                 weighted: bool = False,
                 n_negatives: int = 5,
                 use_margin_for_sampling: bool = False,
                 sampling_type: str = 'semi_hard',
                 multinomial_sampling: bool = False,
                 semi_hard_epsilon: float = 0.):
        super().__init__()

        self.margin = margin
        self.use_centroid = use_centroid
        self.weighted = weighted
        self.normalize = normalize

        self.miner = CosineMiner(n_negatives=n_negatives,
                                 sampling_type=sampling_type,
                                 normalize=False,
                                 multinomial_sampling=multinomial_sampling,
                                 margin=self.margin if use_margin_for_sampling else None,
                                 semi_hard_epsilon=semi_hard_epsilon)

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        if self.normalize:
            context = F.normalize(context)
            response = F.normalize(response)

        response_sim_matrix = (context * response).sum(dim=1)

        negative_indices, _ = self.miner.sampling(context=context, response=response)

        negative = response[negative_indices]

        negative_sim_matrix = torch.bmm(context.unsqueeze(dim=1), negative.transpose(1, 2)).squeeze(dim=1)

        response_sim_matrix = response_sim_matrix.unsqueeze(dim=1)

        difference = response_sim_matrix + negative_sim_matrix

        loss = torch.relu(self.margin - difference.view(-1)).mean()

        loss_output = {
            io.TYPES.loss: loss,
            io.Keys.positive_similarity_matrix: response_sim_matrix,
            io.Keys.negative_similarity_matrix: negative_sim_matrix,
        }

        return loss_output


class TwoStreamMultipleNegativeCosineTripletLoss(nn.Module):

    def __init__(self,
                 margin: float = 0.05,
                 normalize: bool = True,
                 use_centroid: bool = True,
                 weighted: bool = False,
                 n_negatives: int = 5,
                 use_margin_for_sampling: bool = False,
                 sampling_type: str = 'semi_hard',
                 multinomial_sampling: bool = False,
                 semi_hard_epsilon: float = 0.):
        super().__init__()

        self.margin = margin
        self.use_centroid = use_centroid
        self.weighted = weighted
        self.normalize = normalize

        self.miner = CosineMiner(n_negatives=n_negatives,
                                 sampling_type=sampling_type,
                                 normalize=False,
                                 multinomial_sampling=multinomial_sampling,
                                 margin=self.margin if use_margin_for_sampling else None,
                                 semi_hard_epsilon=semi_hard_epsilon)

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        if self.normalize:
            context = F.normalize(context)
            response = F.normalize(response)

        response_sim_matrix = (context * response).sum(dim=1)

        similarity_matrix = self.miner.get_similarity_matrix(context=context,
                                                             response=response,
                                                             semi_hard=self.miner.sampling_type == 'semi_hard')

        response_negative_indices, response_weights = self.miner.get_indices(similarity_matrix=similarity_matrix)
        context_negative_indices, context_weights = self.miner.get_indices(similarity_matrix=similarity_matrix.t())

        response_negative = response[response_negative_indices]
        response_negative_sim_matrix = torch.bmm(context.unsqueeze(dim=1),
                                                 response_negative.transpose(1, 2)).squeeze(dim=1)

        context_negative = context[context_negative_indices]
        context_negative_sim_matrix = torch.bmm(response.unsqueeze(dim=1),
                                               context_negative.transpose(1, 2)).squeeze(dim=1)

        response_sim_matrix = response_sim_matrix.unsqueeze(dim=1)

        response_negative_loss_part = response_sim_matrix + response_negative_sim_matrix
        context_negative_loss_part = response_sim_matrix + context_negative_sim_matrix

        difference = torch.cat((response_negative_loss_part.view(-1),
                                context_negative_loss_part.view(-1)))

        loss = torch.relu(self.margin - difference).mean()

        loss_output = {
            io.TYPES.loss: loss,
            io.Keys.positive_similarity_matrix: response_sim_matrix
        }

        return loss_output


class CentroidMultipleNegativeCosineTripletLoss(nn.Module):

    def __init__(self,
                 margin: float = 0.05,
                 normalize: bool = True,
                 use_centroid: bool = True,
                 weighted: bool = False,
                 n_negatives: int = 5,
                 use_margin_for_sampling: bool = False,
                 sampling_type: str = 'semi_hard',
                 multinomial_sampling: bool = False,
                 semi_hard_epsilon: float = 0.):
        super().__init__()

        self.margin = margin
        self.use_centroid = use_centroid
        self.weighted = weighted
        self.normalize = normalize

        self.miner = CosineMiner(n_negatives=n_negatives,
                                 sampling_type=sampling_type,
                                 normalize=False,
                                 multinomial_sampling=multinomial_sampling,
                                 margin=self.margin if use_margin_for_sampling else None,
                                 semi_hard_epsilon=semi_hard_epsilon)

    def get_negative_sim_matrix(self,
                                context: Tensor,
                                response: Tensor,
                                negative_indices: Tensor,
                                weights: Tensor):

        negative = response[negative_indices]

        if self.use_centroid:
            if self.weighted:
                negative = negative * weights.unsqueeze(dim=-1)
                negative = negative.sum(dim=1)
            else:
                negative = negative.mean(dim=1)
            negative_sim_matrix = (context * negative).sum(dim=1)
        else:
            context = context.unsqueeze(dim=1)
            negative_sim_matrix = torch.bmm(context, negative.transpose(1, 2)).squeeze(dim=1)
            if self.weighted:
                negative_sim_matrix = (negative_sim_matrix * weights).sum(dim=-1)
            else:
                negative_sim_matrix = negative_sim_matrix.mean(dim=-1)

        return negative_sim_matrix

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        if self.normalize:
            context = F.normalize(context)
            response = F.normalize(response)

        response_sim_matrix = (context * response).sum(dim=1)

        negative_indices, weights = self.miner.sampling(context=context, response=response)

        negative_sim_matrix = self.get_negative_sim_matrix(context=context,
                                                           response=response,
                                                           negative_indices=negative_indices,
                                                           weights=weights)

        loss = torch.relu(self.margin - response_sim_matrix + negative_sim_matrix).mean()

        loss_output = {
            io.TYPES.loss: loss,
            io.Keys.positive_similarity_matrix: response_sim_matrix,
            io.Keys.negative_similarity_matrix: negative_sim_matrix,
        }

        return loss_output


class TwoStreamCentroidMultipleNegativeCosineTripletLoss(CentroidMultipleNegativeCosineTripletLoss):

    def __init__(self,
                 margin: float = 0.05,
                 normalize: bool = True,
                 use_centroid: bool = True,
                 weighted: bool = False,
                 n_negatives: int = 5,
                 use_margin_for_sampling: bool = False,
                 sampling_type: str = 'semi_hard',
                 multinomial_sampling: bool = False,
                 semi_hard_epsilon: float = 0.):
        super().__init__(
            margin=margin,
            normalize=normalize,
            use_centroid=use_centroid,
            weighted=weighted,
            n_negatives=n_negatives,
            use_margin_for_sampling=use_margin_for_sampling,
            sampling_type=sampling_type,
            multinomial_sampling=multinomial_sampling,
            semi_hard_epsilon=semi_hard_epsilon
        )

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        if self.normalize:
            context = F.normalize(context)
            response = F.normalize(response)

        response_sim_matrix = (context * response).sum(dim=1)

        similarity_matrix = self.miner.get_similarity_matrix(context=context,
                                                             response=response,
                                                             semi_hard=self.miner.sampling_type == 'semi_hard')

        response_negative_indices, response_weights = self.miner.get_indices(similarity_matrix=similarity_matrix)
        context_negative_indices, context_weights = self.miner.get_indices(similarity_matrix=similarity_matrix.t())

        negative_sim_matrix_from_response = self.get_negative_sim_matrix(context=context,
                                                                         response=response,
                                                                         negative_indices=response_negative_indices,
                                                                         weights=response_weights)

        negative_sim_matrix_from_context = self.get_negative_sim_matrix(context=response,
                                                                        response=context,
                                                                        negative_indices=context_negative_indices,
                                                                        weights=context_weights)

        response_loss = self.margin - response_sim_matrix

        loss_context_response = torch.relu(response_loss + negative_sim_matrix_from_response).mean()
        loss_response_context = torch.relu(response_loss + negative_sim_matrix_from_context).mean()

        loss = 0.5 * loss_context_response + 0.5 * loss_response_context

        loss_output = {
            io.TYPES.loss: loss,
            io.Keys.positive_similarity_matrix: response_sim_matrix
        }

        return loss_output


class LabelSmoothingLoss(nn.Module):

    def __init__(self,
                 smoothing: float = 0.1,
                 use_kl: bool = False,
                 ignore_index: int = -100):
        super().__init__()

        assert 0 <= smoothing < 1

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.use_kl = use_kl

    def smooth_one_hot(self, true_labels: Tensor, classes: int) -> Tensor:

        confidence = 1.0 - self.smoothing

        with torch.no_grad():
            true_dist = torch.empty(size=(true_labels.size(0), classes), device=true_labels.device)
            true_dist.fill_(self.smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

        return true_dist

    def forward(self,
                prediction: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        :param prediction: [batch_size, num_classes]
        :param target: [batch_size]
        :param mask: [batch_size, num_classes] True if need
        :return: scalar
        """

        # TODO solve LabelSmoothingLoss masking
        # final_mask = target != self.ignore_index
        #
        # if mask is not None:
        #     final_mask = torch.logical_and(final_mask, mask)
        #
        # prediction = prediction.masked_select(final_mask)
        # target = target.masked_select(final_mask)

        prediction = F.log_softmax(prediction, dim=-1)

        target_smoothed_dist = self.smooth_one_hot(target, classes=prediction.size(-1))

        if self.use_kl:
            loss = F.kl_div(prediction, target_smoothed_dist, reduction='batchmean')
        else:
            loss = torch.mean(torch.sum(-target_smoothed_dist * prediction, dim=-1))

        return loss


class BaseSoftmax(nn.Module, ABC):

    def __init__(self,
                 smoothing: float = 0.1,
                 normalize: bool = True,
                 scaling_factor: Optional[float] = 10.,
                 margin: float = 0.,
                 warm_up_steps: int = 0,
                 start_step: int = 0):
        super().__init__()

        self.normalize = normalize
        self.scaling_factor = scaling_factor
        self.max_margin = margin
        self.warm_up_steps = warm_up_steps
        self.step = start_step

        if smoothing == 0.:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = LabelSmoothingLoss(smoothing=smoothing)

    @property
    def margin(self) -> float:

        if self.warm_up_steps > 0 and self.step < self.warm_up_steps:
            margin = min(1., self.step / self.warm_up_steps) * self.max_margin
        else:
            margin = self.max_margin

        return margin

    @property
    def semi_hard_factor(self):

        if self.warm_up_steps > 0 and self.step < self.warm_up_steps:
            limit = (self.semi_hard_added_fraction_max - self.semi_hard_added_fraction_min)
            alpha = (min(1., (self.step / self.warm_up_steps)) * limit + self.semi_hard_added_fraction_min)
        else:
            alpha = self.semi_hard_added_fraction_max

        alpha += 1.

        return alpha

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context_embeddings, response_embeddings = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        response_as_context_embeddings, context_as_response_embeddings = None, None

        if io.TYPES.pseudo_head in response_batch:
            response_as_context_embeddings = response_batch[io.TYPES.pseudo_head]

        if io.TYPES.pseudo_head in context_batch:
            context_as_response_embeddings = context_batch[io.TYPES.pseudo_head]

        if self.normalize:

            context = F.normalize(context_embeddings)
            response = F.normalize(response_embeddings)

            if response_as_context_embeddings is not None:
                response_as_context_embeddings = F.normalize(response_as_context_embeddings)

            if context_as_response_embeddings is not None:
                context_as_response_embeddings = F.normalize(context_as_response_embeddings)

            if self.scaling_factor is not None:

                context *= self.scaling_factor
                response *= self.scaling_factor

                if response_as_context_embeddings is not None:
                    response_as_context_embeddings *= self.scaling_factor

                if context_as_response_embeddings is not None:
                    context_as_response_embeddings *= self.scaling_factor

        loss = self.compute_loss(context_embeddings, response_embeddings,
                                 response_as_context_embeddings, context_as_response_embeddings)

        if self.training:
            self.step += 1

        return loss

    @abstractmethod
    def compute_loss(self,
                     context_embeddings: Tensor,
                     response_embeddings: Tensor,
                     response_as_context_embeddings: Optional[Tensor] = None,
                     context_as_response_embeddings: Optional[Tensor] = None) -> io.Batch:
        ...


class SoftmaxLoss(BaseSoftmax):

    def __init__(self,
                 inner: bool = True,
                 pseudo_context_response: bool = True,
                 pseudo_duplicates: bool = True,
                 pseudo_inner: bool = True,
                 dual: bool = True,
                 smoothing: float = 0.1,
                 normalize: bool = True,
                 scaling_factor: Optional[float] = 10.,
                 margin: float = 0.,
                 warm_up_steps: int = 0,
                 start_step: int = 0):
        super().__init__(
            smoothing=smoothing,
            normalize=normalize,
            scaling_factor=scaling_factor,
            margin=margin,
            warm_up_steps=warm_up_steps,
            start_step=start_step
        )

        self.inner = inner
        self.pseudo_context_response = pseudo_context_response
        self.pseudo_duplicates = pseudo_duplicates
        self.pseudo_inner = pseudo_inner
        self.dual = dual

    def compute_loss(self,
                     context_embeddings: Tensor,
                     response_embeddings: Tensor,
                     response_as_context_embeddings: Optional[Tensor] = None,
                     context_as_response_embeddings: Optional[Tensor] = None) -> io.Batch:

        tensor_type = context_embeddings.dtype

        context_similarity_matrix, response_similarity_matrix = None, None

        # similarity_matrix
        similarity_matrix = context_embeddings @ response_embeddings.t()

        context_norm = context_embeddings.norm(dim=-1).type(tensor_type)
        response_norm = response_embeddings.norm(dim=-1).type(tensor_type)

        margin = (torch.eye(similarity_matrix.size(0), dtype=tensor_type) * self.margin).to(similarity_matrix.device)

        normalized_margin = margin * context_norm * response_norm

        similarity_matrix = similarity_matrix - normalized_margin.to(similarity_matrix.device)

        targets = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

        row_similarity_matrix = similarity_matrix

        if self.inner:
            context_similarity_matrix = context_embeddings @ context_embeddings.t()
            context_similarity_matrix = utils.get_non_eye_matrix(context_similarity_matrix)
            row_similarity_matrix = torch.cat((row_similarity_matrix, context_similarity_matrix), dim=1)

        pseudo_similarity_matrix = None

        if self.pseudo_duplicates:
            pseudo_context_response_similarity_matrix = context_embeddings @ context_as_response_embeddings.t()
            pseudo_context_response_similarity_matrix = utils.get_non_eye_matrix(
                pseudo_context_response_similarity_matrix)
            row_similarity_matrix = torch.cat((row_similarity_matrix,
                                               pseudo_context_response_similarity_matrix), dim=1)

        if self.pseudo_inner:
            pseudo_context_similarity_matrix = context_embeddings @ response_as_context_embeddings.t()
            row_similarity_matrix = torch.cat((row_similarity_matrix,
                                               pseudo_context_similarity_matrix), dim=1)

        if self.pseudo_context_response:
            pseudo_similarity_matrix = response_as_context_embeddings @ context_as_response_embeddings.t()
            row_similarity_matrix = torch.cat((row_similarity_matrix, pseudo_similarity_matrix), dim=1)

        if self.inner:
            pseudo_inner_context_embeddings = response_as_context_embeddings @ response_as_context_embeddings.t()
            pseudo_inner_context_embeddings = utils.get_non_eye_matrix(pseudo_inner_context_embeddings)
            row_similarity_matrix = torch.cat((row_similarity_matrix, pseudo_inner_context_embeddings), dim=1)

        loss = self.criterion(row_similarity_matrix, targets)

        if self.dual:

            column_similarity_matrix = similarity_matrix.t()

            if self.inner:
                response_similarity_matrix = response_embeddings @ response_embeddings.t()
                response_similarity_matrix = utils.get_non_eye_matrix(response_similarity_matrix)

                column_similarity_matrix = torch.cat((column_similarity_matrix, response_similarity_matrix), dim=1)

            if self.pseudo_duplicates:
                pseudo_response_context_similarity_matrix = response_embeddings @ response_as_context_embeddings.t()
                pseudo_response_context_similarity_matrix = utils.get_non_eye_matrix(
                    pseudo_response_context_similarity_matrix)
                column_similarity_matrix = torch.cat((column_similarity_matrix,
                                                      pseudo_response_context_similarity_matrix), dim=1)

            if self.pseudo_inner:
                pseudo_response_similarity_matrix = response_embeddings @ context_as_response_embeddings.t()
                column_similarity_matrix = torch.cat((column_similarity_matrix,
                                                      pseudo_response_similarity_matrix), dim=1)

            if self.pseudo_context_response:
                if pseudo_similarity_matrix is None:
                    pseudo_similarity_matrix = response_as_context_embeddings @ context_as_response_embeddings.t()

                pseudo_similarity_matrix = pseudo_similarity_matrix.t()
                column_similarity_matrix = torch.cat((column_similarity_matrix,
                                                      pseudo_similarity_matrix), dim=1)

            if self.inner:
                pseudo_inner_response_embeddings = context_as_response_embeddings \
                                                   @ context_as_response_embeddings.t()
                pseudo_inner_response_embeddings = utils.get_non_eye_matrix(pseudo_inner_response_embeddings)
                column_similarity_matrix = torch.cat((column_similarity_matrix,
                                                      pseudo_inner_response_embeddings), dim=1)

            loss = loss + self.criterion(column_similarity_matrix, targets)

        context_norm = context_norm.unsqueeze(1).repeat(1, context_embeddings.size(0))
        response_norm = response_norm.unsqueeze(1).repeat(1, response_embeddings.size(0))

        similarity_matrix = similarity_matrix / (context_norm * response_norm.t())
        similarity_matrix += margin

        loss_output = {
            io.TYPES.loss: loss,
            io.TYPES.positive_similarity_matrix: similarity_matrix.diag(),
            io.TYPES.negative_similarity_matrix: utils.get_non_eye_matrix(similarity_matrix)
        }

        if context_similarity_matrix is not None:
            context_norms = utils.get_non_eye_matrix(context_norm * context_norm.t())
            context_similarity_matrix = context_similarity_matrix / context_norms
            loss_output[io.TYPES.context_similarity_matrix] = context_similarity_matrix

        if response_similarity_matrix is not None:
            response_norms = utils.get_non_eye_matrix(response_norm * response_norm.t())
            response_similarity_matrix = response_similarity_matrix / response_norms
            loss_output[io.TYPES.response_similarity_matrix] = response_similarity_matrix

        return loss_output


class CzarSoftmax(nn.Module):

    def __init__(self,
                 smoothing: float = 0.1,
                 inner: bool = False,
                 normalize: bool = True,
                 scaling_factor: Optional[float] = 10.,
                 semi_hard: bool = True,
                 semi_hard_factor_min: float = -0.1,
                 semi_hard_factor_max: float = 0.1,
                 margin: float = 0.,
                 sheduling_semi_hard_factor: bool = True,
                 sheduling_margin: bool = True,
                 warm_up_steps: int = 0,
                 start_step: int = 0):
        super().__init__()

        self.inner = inner
        self.normalize = normalize
        self.scaling_factor = scaling_factor
        self.semi_hard = semi_hard
        self.semi_hard_factor_min = semi_hard_factor_min
        self.semi_hard_factor_max = semi_hard_factor_max
        self.max_margin = margin
        self.sheduling_semi_hard_factor = sheduling_semi_hard_factor
        self.sheduling_margin = sheduling_margin
        self.warm_up_steps = warm_up_steps
        self.step = start_step

        if smoothing == 0.:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = LabelSmoothingLoss(smoothing=smoothing)

    @property
    def margin(self) -> float:

        if self.warm_up_steps > 0 and self.step < self.warm_up_steps:
            margin = min(1., self.step / self.warm_up_steps) * self.max_margin
        else:
            margin = self.max_margin

        return margin

    @property
    def semi_hard_factor(self):

        if self.sheduling_semi_hard_factor and self.warm_up_steps > 0 and self.step < self.warm_up_steps:
            limit = (self.semi_hard_factor_max - self.semi_hard_factor_min)
            alpha = (min(1., (self.step / self.warm_up_steps)) * limit + self.semi_hard_factor_min)
        else:
            alpha = self.semi_hard_factor_max

        alpha += 1.

        return alpha

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        tensor_type = context.dtype

        if self.normalize:

            context = F.normalize(context)
            response = F.normalize(response)

            if self.scaling_factor is not None:
                context *= self.scaling_factor
                response *= self.scaling_factor

        similarity_matrix = context @ response.t()

        positive_similarity = similarity_matrix.diag().unsqueeze(1)

        negative_similarity = utils.get_triu_matrix(similarity_matrix)
        negative_similarity = negative_similarity.unsqueeze(0).repeat(positive_similarity.size(0), 1)

        if self.inner:
            context_similarity_matrix = utils.get_triu_matrix(context @ context.t()).unsqueeze(0)
            context_similarity_matrix = context_similarity_matrix.repeat(positive_similarity.size(0), 1)

            response_similarity_matrix = utils.get_triu_matrix(response @ response.t()).unsqueeze(0)
            response_similarity_matrix = response_similarity_matrix.repeat(positive_similarity.size(0), 1)

            result_negative_similarity = torch.cat((negative_similarity,
                                                    context_similarity_matrix,
                                                    response_similarity_matrix), dim=-1)
        else:
            context_similarity_matrix, response_similarity_matrix = None, None
            result_negative_similarity = negative_similarity

        if self.semi_hard:
            semi_hard_mask = result_negative_similarity > positive_similarity * self.semi_hard_factor
            result_negative_similarity[semi_hard_mask] = float('-inf')

        context_norm = context.norm(dim=-1).type(tensor_type)
        response_norm = response.norm(dim=-1).type(tensor_type)

        margin = (torch.ones(similarity_matrix.size(0),
                             dtype=tensor_type) * self.margin).to(similarity_matrix.device)

        normalized_margin = margin * context_norm * response_norm

        positive_similarity = positive_similarity - normalized_margin.unsqueeze(1).to(similarity_matrix.device)

        result_similarity = torch.cat((positive_similarity, result_negative_similarity), dim=-1)

        targets = torch.zeros(similarity_matrix.size(0)).long().to(similarity_matrix.device)

        loss = self.criterion(result_similarity, targets)

        if self.training:
            self.step += 1

        loss_output = {
            io.TYPES.loss: loss,
            io.TYPES.similarity_matrix: similarity_matrix
        }

        return loss_output


class MiningSoftmaxLoss(SoftmaxLoss):

    def __init__(self,
                 smoothing: float = 0.2,
                 dual: bool = False,
                 normalize: bool = False,
                 margin: float = 0.,
                 miner_type: str = 'cosine',
                 n_negatives: int = 4,
                 sampling_type: str = 'semi_hard',
                 multinomial_sampling: bool = False,
                 semi_hard_epsilon: float = 0.,
                 miner_normalize: bool = True,
                 warm_up_steps: int = 0,):
        super().__init__(
            smoothing=smoothing,
            dual=dual,
            normalize=normalize,
            margin=margin,
            warm_up_steps=warm_up_steps
        )

        if miner_type == 'cosine':
            self.miner = CosineMiner(n_negatives=n_negatives,
                                     sampling_type=sampling_type,
                                     normalize=miner_normalize,
                                     multinomial_sampling=multinomial_sampling,
                                     semi_hard_epsilon=semi_hard_epsilon)
        else:
            raise ValueError('Not available miner_type')

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        if self.normalize:
            context = F.normalize(context)
            response = F.normalize(response)

        # TODO add weighted loss
        negative_indices, _ = self.miner.sampling(context=context, response=response)

        negative = response[negative_indices]

        candidates = torch.cat((response.unsqueeze(dim=1), negative), dim=1)

        context = context.unsqueeze(dim=1)

        similarity_matrix = torch.bmm(context, candidates.transpose(1, 2)).squeeze(dim=1)

        margin = torch.eye(similarity_matrix.size(0)) * self.margin
        similarity_matrix = similarity_matrix - margin.to(similarity_matrix.device)

        targets = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

        loss = self.criterion(similarity_matrix, targets)

        if self.dual:
            loss = loss + self.criterion(similarity_matrix.t(), targets)

        if self.training:
            self.step += 1

        loss_output = {
            io.TYPES.loss: loss,
            io.TYPES.similarity_matrix: similarity_matrix
        }

        return loss_output


class BasePTMLLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.miner = ...
        self.criterion = ...

    def forward(self, context_batch: io.Batch, response_batch: io.Batch) -> io.Batch:

        context, response = context_batch[io.TYPES.head], response_batch[io.TYPES.head]

        embeddings = torch.cat((context, response))
        targets = torch.cat((torch.arange(context.size(0)), torch.arange(response.size(0))))

        mined = self.miner(embeddings, targets)

        loss = self.criterion(embeddings, targets, mined)

        loss_output = {
            io.TYPES.loss: loss
        }

        return loss_output


class TripletLoss(BasePTMLLoss):

    def __init__(self,
                 miner_margin: float = 0.2,
                 loss_margin: float = 0.05):
        super().__init__()

        self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets='semihard')
        self.criterion = losses.TripletMarginLoss(margin=loss_margin)


class BaseNTXentLoss(losses.generic_pair_loss.GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(use_similarity=True, mat_based_loss=False, **kwargs)
        self.temperature = temperature

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float().to(neg_pairs.device)
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')

            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0])
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return {"loss": {"losses": -log_exp, "indices": (a1, p), "reduction_type": "pos_pair"}}
        return self.zero_losses()


class NTXentLoss(BasePTMLLoss):

    def __init__(self,
                 cutoff: float = 0.5,
                 nonzero_loss_cutoff: float = 1.4,
                 temperature: float = 0.07):
        super().__init__()

        self.miner = miners.DistanceWeightedMiner(cutoff=cutoff, nonzero_loss_cutoff=nonzero_loss_cutoff)
        self.criterion = BaseNTXentLoss(temperature=temperature)
