from typing import Tuple

from torch import FloatTensor, concat, einsum

__all__ = ["channel_first_to_last", "image_to_patch", "concat_class_token"]


def channel_first_to_last(x: FloatTensor) -> FloatTensor:
    return einsum("bchw->bhwc", x)


def image_to_patch(x: FloatTensor, p: int) -> Tuple[FloatTensor, Tuple[int, int]]:
    b, h, w, c = x.size()
    h_p, w_p = h // p, w // p
    return x.reshape(b, h_p * w_p, p**2 * c), (h_p, w_p)


def concat_class_token(x: FloatTensor, x_cls: FloatTensor) -> FloatTensor:
    return concat((x_cls.expand(x.size(0), 1, x_cls.size(0)), x), 1)
