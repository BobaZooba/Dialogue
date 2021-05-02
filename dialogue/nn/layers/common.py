from torch import nn, Tensor


def compute_n_parameters(module: nn.Module):
    return sum((p.numel() for p in module.parameters()))


def embedding_masking(x: Tensor,
                      pad_mask: Tensor,
                      value: float = 0.) -> Tensor:
    x = x.masked_fill((~(pad_mask.bool())).unsqueeze(-1), value)
    return x


class SpatialDropout(nn.Dropout2d):

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)

        return x
