import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from src.scaled_dot_product_attention import SDPAttention
from src.rope_einx import RoPe

class MultiHeadSelfAttentionRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        """Causal multi-head self-attention

            Args:
                d_model (int): Dimensionality of the Transformer block inputs
                num_heads (int): Number of heads to use in multi-head self-attention
                max_seq_len (int): Maximum sequence length to pre-cache
                theta (float): RoPE parameter
        """
        super().__init__()

        self.Q = nn.Parameter(torch.randn(d_model, d_model))
        self.K = nn.Parameter(torch.randn(d_model, d_model))
        self.V = nn.Parameter(torch.randn(d_model, d_model))
        self.O = nn.Parameter(torch.randn(d_model, d_model))

        self.num_heads = num_heads

        self.rope_layer = RoPe(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)

    def forward(self, x: Float[Tensor, " ... sequence_length d_in"], token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_out"]:
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        q_x = x @ self.Q.T
        k_x = x @ self.K.T
        v_x = x @ self.V.T

        q_x_heads = q_x.reshape(*batch_shape, seq_len, self.num_heads, -1)
        k_x_heads = k_x.reshape(*batch_shape, seq_len, self.num_heads, -1)
        v_x_heads = v_x.reshape(*batch_shape, seq_len, self.num_heads, -1)

        q_x_heads = q_x_heads.transpose(-3, 2)
        k_x_heads = k_x_heads.transpose(-3, 2)
        v_x_heads = v_x_heads.transpose(-3, 2)

        q_x_heads = self.rope_layer.forward(q_x_heads, token_positions)
        k_x_heads = self.rope_layer.forward(k_x_heads, token_positions)

        causal_mask = ~torch.triu(torch.ones(x.shape[-2], x.shape[-2], dtype=bool), diagonal=1)

        attention_layer = SDPAttention(q_x_heads, k_x_heads, v_x_heads, causal_mask)
        embedding_cmhsa = attention_layer.forward()
        embedding_cmhsa_trans = embedding_cmhsa.transpose(-3, -2)
        embedding_cmhsa_combined = embedding_cmhsa_trans.contiguous().reshape(*batch_shape, seq_len, -1)
        result = embedding_cmhsa_combined @ self.O.T

        return result


        


