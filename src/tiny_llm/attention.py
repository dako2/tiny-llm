import mlx.core as mx
from .basics import softmax, linear
import torch

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """
    https://skyzh.github.io/tiny-llm/

    Implement scaled_dot_product_attention following the below attention function. The function takes key, value, and query of the same dimensions, and an optional mask matrix M.

    Attention=softmax(QK/sqrt(d)+M)V
        
    1/sqrt(dk) is the scale factor. The user might specify their own scale factor or use the default one.
    L is seq_len, in PyTorch API it's S (source len)
    D is head_dim
    the number of attention heads (n_heads)

    head_dim = hidden_size // num_heads


    embedding dimension (d)
    N.. can be zero or more dimensions for batches

    key: N.. x L x D
    value: N.. x L x D
    query: N.. x L x D
    output: N.. x L x D
    scale = 1/sqrt(D) if not specified


    """
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    #NxLxD
    scores = mx.matmul(query, key.swapaxes(-1, -2)) * factor
    if mask is not None:
        scores = scores + mask #LxL
    return mx.matmul(softmax(scores, axis=-1), value) #LxL, LxD ==> NxLxD
    
class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads #

        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads #heads
        self.scale = mx.rsqrt(self.head_dim)

        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:

        # E is hidden_size or embed_dim or dims or model_dim
        # H is num_heads
        # D is head_dim
        # L is seq_len, in PyTorch API it's S (source len)

        # w_q/w_k/w_v: E x (H x D)
        # output/input: N x L x E
        # w_o: (H x D) x E

        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        x = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    """
    mask = torch.tril(torch.ones(L, S), (S - L))
    mask = torch.where(mask.bool(), torch.tensor(0), torch.tensor(-torch.inf)).to(dtype)

    Mask: L = 3, S = 5
    0   0   0   -inf -inf
    0   0   0   0    -inf
    0   0   0   0    0
    """
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None, #causal or some array 
) -> mx.array:
    """
    the case where the number of query heads is a multiple of the number of key/value heads.
    H_q // Group = H_key, H_value

    N.. is zero or more dimensions for batches
    H_q is the number of query heads
    H is the number of key/value heads (H_q must be divisible by H)
    L is the query sequence length
    S is the key/value sequence length
    D is the head dimension

    query: N.. x H_q x L x D
    key: N.. x H x S x D
    value: N.. x H x S x D
    mask: N.. x H_q x L x S
    output: N.. x H_q x L x D

    """
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale

    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]

    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(-1, H, n_repeats, L, D) # -1 takes 2x3 to 6, e.g. H_q ==> H x n_repeats
    key = key.reshape(-1, H, 1, S, D) #QK Hx1xLxS
    value = value.reshape(-1, H, 1, S, D) #Hx1xSxD

    #N..xH x1x LxS ==> S<L
    scores = mx.matmul(query, key.swapaxes(-1, -2)) * factor 
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype) # H x n_repeats x L x S
            scores = scores + mask  # H x n_repeats x L x S + L x S == H x n_repeats x L x S
        else:
            mask = mask.reshape(-1, H, n_repeats, mask.shape[-2], mask.shape[-1])
            scores = scores + mask
    result = mx.matmul(softmax(scores, axis=-1), value)
    return result.reshape(expected_shape)

def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
