from torch import FloatTensor, empty, nn

__all__ = ["PositionalEncoding"]


class PositionalEncoding(nn.Module):
    def __init__(self, num_patch: int, emb_dim: int) -> None:
        super().__init__()
        self.num_patch = num_patch
        self.emb_dim = emb_dim

        self.E_pos = nn.Parameter(empty(num_patch, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.E_pos)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return x + self.E_pos

    def extra_repr(self) -> str:
        return "num_patch={}, emb_dim={}".format(self.num_patch, self.emb_dim)
