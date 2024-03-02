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
    def __init__(self, channel_dim: int, patch_size: int, emb_dim: int) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.E = nn.Parameter(empty(emb_dim, channel_dim * patch_size**2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.E, std=1 / sqrt(self.channel_dim * self.patch_size**2))

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = channel_first_to_last(x)
        x, _ = image_to_patch(x, self.patch_size)
        return linear(x, self.E)

    def extra_repr(self) -> str:
        return "channel_dim={}, patch_size={}, emb_dim={}".format(
            self.channel_dim, self.patch_size, self.emb_dim
        )


class ClassToken(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        self.x_class = nn.Parameter(empty(emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.x_class, std=1 / sqrt(self.emb_dim))

    def forward(self, x: FloatTensor) -> FloatTensor:
        return concat_class_token(x, self.x_class)

    def extra_repr(self) -> str:
        return "emb_dim={}".format(self.emb_dim)
