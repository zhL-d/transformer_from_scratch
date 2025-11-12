import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
import einx

class RoPe(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """Constructthe RoPE module and create buffers if needed.

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None): Device to store the buffer on
        """
        super().__init__()

        block_num = d_k // 2

        angle_i = torch.arange(max_seq_len)
        angle_k = torch.arange(1, block_num + 1)
        angle = einx.multiply("max_seq_len 1, 1 block_num -> max_seq_len block_num", angle_i[:, None], torch.reciprocal(theta ** ((2*angle_k[None, :] - 2) / d_k)))

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        
    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        *batch, seq_len, d_k = x.shape
        block = d_k // 2

        x_blocked = x.reshape(*batch, seq_len, block, -1)

        x_even = x_blocked[..., 0]
        x_odd = x_blocked[..., 1]

        sin_pos = self.sin[token_positions]
        cos_pos = self.cos[token_positions]

        x_even_rot = x_even * cos_pos - x_odd * sin_pos
        x_odd_rot = x_even * sin_pos + x_odd * cos_pos

        result_blocked = torch.stack((x_even_rot, x_odd_rot), dim=-1)
        result = result_blocked.reshape(*batch, seq_len, -1)

        return result
