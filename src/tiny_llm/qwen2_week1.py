import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        """
        hidden_size: D
        num_kv_heads: H
        num_heads: H_q
        max_seq_len: L

        x: B, L, E
        q = linear(x, wq, bq) -> B, L, H_q, D
        k = linear(x, wk, bk) -> B, L, H, D
        v = linear(x, wv, bv) -> B, L, H, D
        q = rope(q, offset=slice(offset, offset + L))
        k = rope(k, offset=slice(offset, offset + L))
        (transpose as needed)
        x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
        (transpose as needed)
        x = linear(x, wo) -> B, L, E
        """
        self.hidden_size = hidden_size #D
        self.num_heads = num_heads #H_q
        self.num_kv_heads = num_kv_heads #H
        
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )

        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:

        B, L, _ = x.shape

        projection_q = linear(x, wq, bq).reshape(B, L, self.num_heads, self.head_dim)
        projection_k = linear(x, wk, bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        projection_v = linear(x, wv, bv).reshape(B, L, self.num_kv_heads, self.head_dim)
        
        #scaled_dot_product_attention_grouped
        projection_q = self.rope(projection_q, offset=slice(offset, offset + L))
        projection_k = self.rope(projection_k, offset=slice(offset, offset + L))
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        x = scaled_dot_product_attention_grouped(
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)

class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        """
        MLP(x)=(SiLU(W gate(x)) ⊙ Wup (x)) W_down
        
        """
        self.w_gate = w_gate
        self.w_down = w_down
        self.w_up = w_up
        self.dim = dim 
        self.hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)

class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        
        self.intermediate_size = intermediate_size
        self.w_gate = w_gate 
        self.w_up = w_up 
        self.w_down = w_down 
        self.w_post_attention_layernorm = w_post_attention_layernorm
        
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )

        self.self_atten = Qwen2MultiHeadAttention(
                        hidden_size=hidden_size,
                        num_heads=num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        wq=wq,
                        wk=wk,
                        wv=wv,
                        wo=wo,
                        bq=bq,
                        bk=bk,
                        bv=bv,
                        max_seq_len=max_seq_len,
                        theta=theta
                    )
        self.mlp = Qwen2MLP(-1, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
    
        x1 = self.input_layernorm(x)
        r = self.self_atten(x1, offset, mask)
        h = x + r
        x2 = self.post_attention_layernorm(h)
        out = self.mlp(x2) + h
        return out


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
