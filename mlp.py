from torch import FloatTensor, nn

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(self, emb_dim: int, mlp_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, emb_dim)

    def forward(self, z: FloatTensor) -> FloatTensor:
        return self.linear2(self.gelu(self.linear1(z)))
