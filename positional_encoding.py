from torch import FloatTensor, empty, nn

__all__ = ["PositionalEncoding"]


class PositionalEncoding(nn.Module):
    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        self.n = n
        self.d = d

        self.E_pos = nn.Parameter(empty(n, d))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.E_pos)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return x + self.E_pos

    def extra_repr(self) -> str:
        return "n={}, d={}".format(self.n, self.d)
