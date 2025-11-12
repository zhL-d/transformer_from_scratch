import torch
import torch.nn as nn
import einx
from jaxtyping import Float
from torch import Tensor

class PWFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        # d_ff = 8/3 * d_model

        self.W1 = nn.Parameter(torch.randn(d_ff, d_model, dtype=dtype, device=device))
        self.W3 = nn.Parameter(torch.randn(d_ff, d_model, dtype=dtype, device=device))

        self.W2 = nn.Parameter(torch.randn(d_model, d_ff, dtype=dtype, device=device))
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, "... d_model"]:
        w1_item = einx.dot("... [d_model], [d_model] d_ff -> ... d_ff", x, self.W1.T)
        w1_gate_item = PWFFN.silu(w1_item)

        w3_item = einx.dot("... [d_model], [d_model] d_ff -> ... d_ff", x, self.W3.T)

        l1 = einx.multiply("... d_ff, ... d_ff -> ... d_ff", w1_gate_item, w3_item)
        result = einx.dot("... [d_ff], [d_ff] d_model -> ... d_model", l1, self.W2.T)

        return result
   
    @staticmethod
    def silu(x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return x * torch.sigmoid(x)