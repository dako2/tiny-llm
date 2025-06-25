
# Notes:
https://blog.kkkzoz.top/posts/learning/deep-learning-basic/#saving-and-loading-models



# tiny-llm - LLM Serving in a Week

[![CI (main)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml/badge.svg)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml)

A course on LLM serving using MLX for system engineers. The codebase
is solely (almost!) based on MLX array/matrix APIs without any high-level neural network APIs, so that we
can build the model serving infrastructure from scratch and dig into the optimizations.

The goal is to learn the techniques behind efficiently serving a large language model (e.g., Qwen2 models).

Why MLX: nowadays it's easier to get a macOS-based local development environment than setting up an NVIDIA GPU.

Why Qwen2: this was the first LLM I've interacted with -- it's the go-to example in the vllm documentation. I spent some time looking at the vllm source code and built some knowledge around it.

## Book

The tiny-llm book is available at [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/). You can follow the guide and start building.

## Community

You may join skyzh's Discord server and study with the tiny-llm community.

[![Join skyzh's Discord Server](book/src/discord-badge.svg)](https://skyzh.dev/join/discord)

## Roadmap

Week 1 is complete. Week 2 is in progress.

| Week + Chapter | Topic                                                       | Code | Test | Doc |
| -------------- | ----------------------------------------------------------- | ---- | ---- | --- |
| 1.1            | Attention                                                   | ✅    | ✅   | ✅  |
| 1.2            | RoPE                                                        | ✅    | ✅   | ✅  |
| 1.3            | Grouped Query Attention                                     | ✅    | ✅   | ✅  |
| 1.4            | RMSNorm and MLP                                             | ✅    | ✅   | ✅  |
| 1.5            | Load the Model                                              | ✅    | ✅   | ✅  |
| 1.6            | Generate Responses (aka Decoding)                           | ✅    | ✅   | ✅  |
| 1.7            | Sampling                                                    | ✅    | ✅   | ✅  |
| 2.1            | Key-Value Cache                                             | ✅    | 🚧   | 🚧  |
| 2.2            | Quantized Matmul and Linear - CPU                           | ✅    | 🚧   | 🚧  |
| 2.3            | Quantized Matmul and Linear - GPU                           | ✅    | 🚧   | 🚧  |
| 2.4            | Flash Attention 2 - CPU                                     | ✅    | 🚧   | 🚧  |
| 2.5            | Flash Attention 2 - GPU                                     | ✅    | 🚧   | 🚧  |
| 2.6            | Continuous Batching                                         | ✅    | 🚧   | 🚧  |
| 2.7            | Chunked Prefill                                             | ✅    | 🚧   | 🚧  |
| 3.1            | Paged Attention - Part 1                                    | 🚧    | 🚧   | 🚧  |
| 3.2            | Paged Attention - Part 2                                    | 🚧    | 🚧   | 🚧  |
| 3.3            | MoE (Mixture of Experts)                                    | 🚧    | 🚧   | 🚧  |
| 3.4            | Speculative Decoding                                        | 🚧    | 🚧   | 🚧  |
| 3.5            | Prefill-Decode Separation (requires two Macintosh devices)  | 🚧    | 🚧   | 🚧  |
| 3.6            | Parallelism                                                 | 🚧    | 🚧   | 🚧  |
| 3.7            | AI Agent     / Tool Calling                                 | 🚧    | 🚧   | 🚧  |

Other topics not covered: quantized/compressed kv cache, prefix/prompt cache; sampling, fine tuning; smaller kernels (softmax, silu, etc)
