from torch import FloatTensor, nn

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(self, d: int, d_mlp: int):
        super().__init__()
        self.linear1 = nn.Linear(d, d_mlp)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_mlp, d)

    def forward(self, z: FloatTensor) -> FloatTensor:
        return self.linear2(self.gelu(self.linear1(z)))
