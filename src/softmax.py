import torch
from jaxtyping import Float
from torch import Tensor

def Softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    logit_stable = x - x.max(dim=dim, keepdim=True).values
    logit_stable_exp = torch.exp(logit_stable)
    result = logit_stable_exp / logit_stable_exp.sum(dim=dim, keepdim=True)

    return result
