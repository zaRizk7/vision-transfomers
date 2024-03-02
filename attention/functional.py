from math import sqrt
from typing import Iterator, Tuple

from torch import FloatTensor, concat, einsum

__all__ = [
    "attention",
    "qkv_self_project",
    "self_attention",
    "multihead_self_attention",
]


def attention(q: FloatTensor, k: FloatTensor, v: FloatTensor) -> FloatTensor:
    A = einsum("bmd,bnd->bmn", q, k)
    A = A / sqrt(q.size(-1))
    A = A.softmax(-1)
    return einsum("bmn,bnd->bmd", A, v)


def qkv_self_project(
    z: FloatTensor, U_qkv: FloatTensor
) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
    return einsum("bnk,fkd->fbnd", z, U_qkv).unbind()


def self_attention(z: FloatTensor, U_qkv: FloatTensor) -> Iterator[FloatTensor]:
    for U_qkv_i in U_qkv.unbind():
        q_i, k_i, v_i = qkv_self_project(z, U_qkv_i)
        yield attention(q_i, k_i, v_i)


def multihead_self_attention(
    z: FloatTensor, U_qkv: FloatTensor, U_msa: FloatTensor
) -> FloatTensor:
    sa = concat(tuple(self_attention(z, U_qkv)), -1)
    return einsum("bno,od->bnd", sa, U_msa)
