import mlx.core as mx

"""
You only need to consider offset being None or a single slice. The list[slice] case will be implemented when we start implementing the continuous batching feature. Assume all batches provided use the same offset.

x: (N, L, H, D)
cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
In the traditional form of RoPE, each head on the dimension of D is viewed as consequtive complex pairs. That is to say, if D = 8, then, x[0] and x[1] are a pair, x[2] and x[3] are another pair, and so on. A pair gets the same frequency from cos/sin_freqs.

Note that, practically, D can be even or odd. In the case of D being odd, the last dimension of x doesn’t have a matching pair, and is typically left untouched in most implementations. For simplicity, we just assume that D is always even.

output[0] = x[0] * cos_freqs[0] + x[1] * -sin_freqs[0]
output[1] = x[0] * sin_freqs[0] + x[1] * cos_freqs[0]
output[2] = x[2] * cos_freqs[1] + x[3] * -sin_freqs[1]
output[3] = x[2] * sin_freqs[1] + x[3] * cos_freqs[1]
...and so on
You can do this by reshaping x to (N, L, H, D // 2, 2) and then applying the above formula to each pair.
"""

class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        pass

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        pass
