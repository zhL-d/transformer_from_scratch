import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

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

        angle_i, angle_k = torch.meshgrid(
            torch.arange(max_seq_len, dtype=torch.float32), 
            torch.arange(1, d_k // 2 + 1, dtype=torch.float32), 
            indexing="ij"
        )
        angle = angle_i / theta ** ((2*angle_k - 2) / d_k)

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

        self.R = torch.zeros(max_seq_len, d_k // 2, 2, 2)

        self.R[:, :, 0, 0] = self.cos
        self.R[:, :, 0, 1] = -self.sin
        self.R[:, :, 1, 0] = self.sin
        self.R[:, :, 1, 1] = self.cos
        
    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        batch, seq_len, d_k = x.shape
        block = d_k // 2

        # TODO: batch ...
        x_blocked = x.reshape(batch, seq_len, block, -1)
        x_blocked_reshaped = x_blocked.unsqueeze(-1)


        r_selected_batched = self.R[token_positions]

        x_roped = r_selected_batched @ x_blocked_reshaped

        result = x_roped.reshape(batch, seq_len, -1)

        return result
