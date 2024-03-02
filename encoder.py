from torch import FloatTensor, nn

from attention import MultiheadSelfAttention
from mlp import MLP

__all__ = ["TransformerEncoder"]


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim: int, qkv_dim: int, num_head: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.msa = MultiheadSelfAttention(emb_dim, qkv_dim, num_head)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, emb_dim * 4)

    def forward(self, z: FloatTensor) -> FloatTensor:
        zl = self.msa(self.ln1(z)) + z
        zl = self.mlp(self.ln2(zl)) + zl
        return zl
