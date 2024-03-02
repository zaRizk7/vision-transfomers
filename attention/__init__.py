from math import sqrt

from torch import FloatTensor, empty, nn

from attention.functional import multihead_self_attention

__all__ = ["MultiheadSelfAttention"]


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d: int, d_h: int, k: int) -> None:
        super().__init__()
        self.d = d
        self.d_h = d_h
        self.k = k

        self.U_qkv = nn.Parameter(empty(k, 3, d, d_h))
        self.U_msa = nn.Parameter(empty(k * d_h, d))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.U_qkv, std=1 / sqrt(self.d))
        nn.init.normal_(self.U_msa, std=1 / sqrt(self.k * self.d_h))

    def forward(self, z: FloatTensor) -> FloatTensor:
        return multihead_self_attention(z, self.U_qkv, self.U_msa)

    def extra_repr(self) -> str:
        return "d={}, d_h={}, k={}".format(self.d, self.d_h, self.k)
