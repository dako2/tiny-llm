import mlx.core as mx

"""

y= x/mx.sqrt(mean(x^2)+epsilon)*weight

Where:

x is the input tensor.
weight is a learnable scaling parameter.
epsilon (eps) is a small constant added for numerical stability (e.g., 1e-5 or 1e-6).
mean(x^2) is the sum of squares and then division by the number of elements.

D is the embedding dimension.
x: N.. x D
weight: D
output: N.. x D
"""

class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.weight = weight.astype(mx.float32)
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        origin_type = x.dtype
        x = x.astype(mx.float32)
        y= x *self.weight * mx.rsqrt(mx.mean(mx.square(x), axis=self.dim, keepdims=True)+self.eps)
        return y.astype(origin_type)
