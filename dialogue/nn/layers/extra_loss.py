import torch
from torch import nn, Tensor
from typing import Optional
from dialogue.nn.losses import LabelSmoothingLoss, BaseSoftmax
from dialogue import io, utils


def get_negative_similarity_matrix(similarity_matrix):
    expanded_similarity_matrix = torch.cat((similarity_matrix, similarity_matrix.T), dim=-1)

    mask = ~torch.cat((torch.eye(similarity_matrix.size(0)), torch.eye(similarity_matrix.size(0))), dim=-1).bool()

    negative_similarity_matrix = expanded_similarity_matrix[mask].view(similarity_matrix.size(0),
                                                                       similarity_matrix.size(0) * 2 - 2)

    return negative_similarity_matrix


def get_inner_similarities(embeddings, mask):
    inner_similarity_matrix = embeddings @ embeddings.T
    inner_similarities = inner_similarity_matrix[mask]

    return inner_similarities


class ExtraSoftmaxLoss(BaseSoftmax):

    def __init__(self,
                 negative_loss_alpha: float = 0.1,
                 full_negatives: bool = True,
                 inner_quantile: float = 0.95,
                 inner_smoothing: float = 0.1,
                 pseudo_positive_quantile: float = 0.9,
                 pseudo_positive_smoothing: float = 0.3,
                 pseudo_negative_quantile: float = 1.,
                 pseudo_negative_smoothing: Optional[float] = None,
                 duplicates_quantile: float = 0.85,
                 duplicates_smoothing: float = 0.3,
                 pseudo_inner_quantile: float = 0.9,
                 pseudo_inner_smoothing: float = 0.1,
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

        self.negative_loss_alpha = negative_loss_alpha

        self.full_negatives = full_negatives

        self.inner_quantile = inner_quantile
        self.inner_smoothing = inner_smoothing

        self.pseudo_positive_quantile = pseudo_positive_quantile
        self.pseudo_positive_smoothing = pseudo_positive_smoothing

        self.pseudo_negative_quantile = pseudo_negative_quantile
        self.pseudo_negative_smoothing = pseudo_negative_smoothing if \
            pseudo_negative_smoothing is not None else self.smoothing

        self.duplicates_quantile = duplicates_quantile
        self.duplicates_smoothing = duplicates_smoothing

        self.pseudo_inner_quantile = pseudo_inner_quantile
        self.pseudo_inner_smoothing = pseudo_inner_smoothing

        self.negative_criterion = nn.BCEWithLogitsLoss()

    def get_negative_similarities_loss(self, similarities, smoothing, quantile):
        if quantile < 1.:
            similarities_mask = similarities <= torch.quantile(similarities, quantile)
            similarities = similarities[similarities_mask]

        targets = (torch.zeros_like(similarities) + smoothing).to(similarities.device)

        loss = self.negative_criterion(similarities, targets)

        return loss

    def compute_loss(self,
                     context_embeddings: Tensor,
                     response_embeddings: Tensor,
                     response_as_context_embeddings: Optional[Tensor] = None,
                     context_as_response_embeddings: Optional[Tensor] = None) -> io.Batch:

        use_pseudo = context_as_response_embeddings is not None and response_as_context_embeddings is not None

        negative_losses = list()

        similarity_matrix = context_embeddings @ response_embeddings.T

        positive_similarity = similarity_matrix.diag()

        positive_similarity -= self.margin * context_embeddings.norm(dim=-1) * response_embeddings.norm(dim=-1)

        positive_similarity = positive_similarity.unsqueeze(dim=1)

        negative_similarity_matrix = get_negative_similarity_matrix(similarity_matrix)

        inner_mask = torch.triu(torch.ones_like(similarity_matrix)).bool()
        inner_mask = inner_mask * ~torch.eye(similarity_matrix.size(0)).bool()

        if self.inner_quantile > 0.:
            inner_context_loss = self.get_negative_similarities_loss(
                get_inner_similarities(context_embeddings, inner_mask),
                smoothing=self.inner_smoothing,
                quantile=self.inner_quantile
            )
            negative_losses.append(inner_context_loss)

            inner_response_loss = self.get_negative_similarities_loss(
                get_inner_similarities(response_embeddings, inner_mask),
                smoothing=self.inner_smoothing,
                quantile=self.inner_quantile
            )
            negative_losses.append(inner_response_loss)

            if context_as_response_embeddings is not None:
                inner_context_as_response_loss = self.get_negative_similarities_loss(
                    get_inner_similarities(context_as_response_embeddings, inner_mask),
                    smoothing=self.inner_smoothing,
                    quantile=self.inner_quantile
                )
                negative_losses.append(inner_context_as_response_loss)

            if response_as_context_embeddings is not None:
                inner_response_as_context_loss = self.get_negative_similarities_loss(
                    get_inner_similarities(response_as_context_embeddings, inner_mask),
                    smoothing=self.inner_smoothing,
                    quantile=self.inner_quantile
                )
                negative_losses.append(inner_response_as_context_loss)

        if use_pseudo:
            pseudo_similarity_matrix = context_as_response_embeddings @ response_as_context_embeddings.T

            pseudo_positive_similarities = pseudo_similarity_matrix.diag()
            pseudo_negative_similarities = utils.get_non_eye_matrix(pseudo_similarity_matrix).view(-1)

            if self.pseudo_positive_quantile > 0.:
                pseudo_positive_loss = self.get_negative_similarities_loss(
                    pseudo_positive_similarities,
                    smoothing=self.pseudo_positive_smoothing,
                    quantile=self.pseudo_positive_quantile
                )
                negative_losses.append(pseudo_positive_loss)

            if self.pseudo_negative_quantile > 0.:
                pseudo_negative_loss = self.get_negative_similarities_loss(
                    pseudo_negative_similarities,
                    smoothing=self.pseudo_negative_smoothing,
                    quantile=self.pseudo_negative_quantile
                )
                negative_losses.append(pseudo_negative_loss)

            context_pseudo_context_similarity_matrix = context_embeddings @ context_as_response_embeddings.T
            context_pseudo_response_similarity_matrix = context_embeddings @ response_as_context_embeddings.T

            response_pseudo_response_similarity_matrix = response_embeddings @ response_as_context_embeddings.T
            response_pseudo_context_similarity_matrix = response_embeddings @ context_as_response_embeddings.T

            context_pseudo_context_positive_similarity = context_pseudo_context_similarity_matrix.diag()
            context_pseudo_context_negative_similarity_matrix = get_negative_similarity_matrix(
                context_pseudo_context_similarity_matrix
            )

            response_pseudo_response_positive_similarity = response_pseudo_response_similarity_matrix.diag()
            response_pseudo_response_negative_similarity_matrix = get_negative_similarity_matrix(
                response_pseudo_response_similarity_matrix
            )

            negative_similarity_matrix = torch.cat((
                negative_similarity_matrix,
                context_pseudo_context_negative_similarity_matrix,
                response_pseudo_response_negative_similarity_matrix),
                dim=1
            )

            if self.duplicates_quantile > 0.:
                context_duplicates_negative_loss = self.get_negative_similarities_loss(
                    context_pseudo_context_positive_similarity,
                    smoothing=self.duplicates_smoothing,
                    quantile=self.duplicates_quantile
                )
                negative_losses.append(context_duplicates_negative_loss)

                response_duplicates_negative_loss = self.get_negative_similarities_loss(
                    response_pseudo_response_positive_similarity,
                    smoothing=self.duplicates_smoothing,
                    quantile=self.duplicates_quantile
                )
                negative_losses.append(response_duplicates_negative_loss)

                if self.inner_quantile > 0.:
                    pseudo_inner_context_negative_loss = self.get_negative_similarities_loss(
                        context_pseudo_response_similarity_matrix.view(-1),
                        smoothing=self.inner_smoothing,
                        quantile=self.inner_quantile
                    )
                    negative_losses.append(pseudo_inner_context_negative_loss)

                    pseudo_inner_response_negative_loss = self.get_negative_similarities_loss(
                        response_pseudo_context_similarity_matrix.view(-1),
                        smoothing=self.inner_smoothing,
                        quantile=self.inner_quantile
                    )
                    negative_losses.append(pseudo_inner_response_negative_loss)

        if self.full_negatives:
            negative_similarity_matrix = negative_similarity_matrix.view(-1).unsqueeze(dim=0)
            negative_similarity_matrix = negative_similarity_matrix.repeat(positive_similarity.size(0), 1)

        result_similarity_matrix = torch.cat((positive_similarity, negative_similarity_matrix), dim=1)

        targets = torch.zeros(positive_similarity.size(0)).long().to(positive_similarity.device)

        loss = self.criterion(result_similarity_matrix, targets)

        negative_loss = torch.mean(Tensor(negative_losses))

        loss += self.negative_loss_alpha * negative_loss

        loss_output = {
            io.TYPES.loss: loss,
            io.TYPES.positive_similarity_matrix: similarity_matrix.diag(),
            io.TYPES.negative_similarity_matrix: utils.get_non_eye_matrix(similarity_matrix)
        }

        return loss_output
