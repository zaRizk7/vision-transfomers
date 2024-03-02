from math import sqrt

from torch import FloatTensor, empty, nn

from attention.functional import multihead_self_attention

__all__ = ["MultiheadSelfAttention"]


class MultiheadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, qkv_dim: int, num_head: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.qkv_dim = qkv_dim
        self.num_head = num_head

        self.U_qkv = nn.Parameter(empty(num_head, 3, emb_dim, qkv_dim))
        self.U_msa = nn.Parameter(empty(num_head * qkv_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.U_qkv, std=1 / sqrt(self.emb_dim))
        nn.init.normal_(self.U_msa, std=1 / sqrt(self.num_head * self.qkv_dim))

    def forward(self, z: FloatTensor) -> FloatTensor:
        return multihead_self_attention(z, self.U_qkv, self.U_msa)

    def extra_repr(self) -> str:
        return "emb_dim={}, qkv_dim={}, num_head={}".format(
            self.emb_dim, self.qkv_dim, self.num_head
        )
