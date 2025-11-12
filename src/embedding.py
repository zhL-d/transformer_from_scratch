import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Construct an embedding module.

        Args:
            num_embeddings(int): Size of the vocabulary
            embedding_dim(int): Dimension of the embedding vectors
            device(torch.device | None): Device to store the parameters on
            dtype(torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        temp_W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        temp_W = torch.nn.init.trunc_normal_(temp_W, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(temp_W)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids(torch.Tensor): token ids with shape (batch_size, sequence_length)
        """
        return self.W[token_ids]


