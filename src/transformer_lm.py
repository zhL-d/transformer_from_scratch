import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from src.embedding import Embedding
from src.transformer_block import TransformerBlock
from src.rmsnorm_einx import RMSNorm
from src.linear_module import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float):
        """Implement the Transformer language model

            Args:
                vocab_size (int): The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix
                context_length (int): The maximum context length, necessary for determining the dimensionality of the position embedding matrix
                num_layers (int): The number of Transformer blocks to use
                d_model (int): Dimensionality of the Transformer block inputs
                num_heads (int): Number of heads to use in multi-head self-attention
                d_ff (int): Dimensionality of the position-wise feed-forward inner layer
                rope_theta (float): RoPE parameter               
        """
        super().__init__()

        self.embedding_layer = Embedding(vocab_size, d_model)
        self.transformerblock_layer = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        # self.transformerblock_layer = [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        self.rmsnorm_layer = RMSNorm(d_model)
        self.linear_layer = Linear(d_model, vocab_size)
    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        """Implement the Transformer language model

            Args:
                in_indices (Int[Tensor, " batch_size sequence_length"]): Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`
            
            Returns:
                Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized next-word distribution for each token.
        """
        input_embedding = self.embedding_layer.forward(in_indices)

        for block in self.transformerblock_layer:
            output_embedding = block.forward(input_embedding)
            input_embedding = output_embedding
        
        output_embedding = self.rmsnorm_layer.forward(output_embedding)
        
        result = self.linear_layer.forward(output_embedding)

        return result