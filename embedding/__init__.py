from math import sqrt

from torch import FloatTensor, empty, nn
from torch.nn.functional import linear

from embedding.functional import (
    channel_first_to_last,
    concat_class_token,
    image_to_patch,
)

__all__ = ["PatchEmbedding", "ClassToken"]


class PatchEmbedding(nn.Module):
    def __init__(self, c: int, p: int, d: int) -> None:
        super().__init__()
        self.c = c
        self.p = p
        self.d = d
        self.E = nn.Parameter(empty(d, c * p**2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.E, std=1 / sqrt(self.c * self.p**2))

    def forward(self, x: FloatTensor) -> FloatTensor:
        x_hat = channel_first_to_last(x)
        x_hat, _ = image_to_patch(x_hat, self.p)
        return linear(x_hat, self.E)

    def extra_repr(self) -> str:
        return "c={}, p={}, d={}".format(self.c, self.p, self.d)


class ClassToken(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

        self.x_class = nn.Parameter(empty(d))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.x_class, std=1 / sqrt(self.d))

    def forward(self, x: FloatTensor) -> FloatTensor:
        return concat_class_token(x, self.x_class)

    def extra_repr(self) -> str:
        return "d={}".format(self.d)
