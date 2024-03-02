from torch import FloatTensor, nn

from attention import MultiheadSelfAttention
from mlp import MLP

__all__ = ["TransformerEncoder"]


class TransformerEncoder(nn.Module):
    def __init__(self, d: int, d_h: int, k: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.msa = MultiheadSelfAttention(d, d_h, k)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d, d * 4)

    def forward(self, z: FloatTensor) -> FloatTensor:
        z_hat = self.msa(self.ln1(z)) + z
        z_hat = self.mlp(self.ln2(z_hat)) + z_hat
        return z_hat
