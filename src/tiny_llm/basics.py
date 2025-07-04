import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    if bias is not None:
        return mx.matmul(x, w.T) + bias
    else:
        return mx.matmul(x, w.T)


def silu(x: mx.array) -> mx.array:
    """
    SiLU(x)=x/(1+mx.exp(-x))
    """

    return x / (1+mx.exp(-x))

    