import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x, :]

    def as_linear(self, x: mx.array) -> mx.array:
        """
        In the Qwen2 model, the embedding layer can also be used as a linear layer to map the embeddings back to the token space.
        
        Embedding::as_linear
            weight: vocab_size x embedding_dim
            Input: N.. x embedding_dim
            Output: N.. x vocab_size
        """
        return linear(x, self.weight)
