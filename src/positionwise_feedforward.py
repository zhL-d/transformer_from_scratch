import torch
import torch.nn as nn

class PWFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        # d_ff = 8/3 * d_model

        self.W1 = nn.Parameter(torch.randn(d_ff, d_model, dtype=dtype, device=device))
        self.W3 = nn.Parameter(torch.randn(d_ff, d_model, dtype=dtype, device=device))

        self.W2 = nn.Parameter(torch.randn(d_model, d_ff, dtype=dtype, device=device))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_term = x @ self.W1.T
        w1_gate_term = PWFFN.silu(w1_term)

        w3_term = x @ self.W3.T

        l1 = w1_gate_term * w3_term
        result = l1 @ self.W2.T

        return result
   
    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)