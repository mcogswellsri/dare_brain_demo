import torch
import torch.nn as nn


# from https://github.com/pytorch/fairseq/blob/89a2a0ccdebd0943b2878ff2150f8a5f836cc4aa/fairseq/utils.py
def make_positions(tensor):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    ones = torch.ones_like(tensor).int()
    return torch.cumsum(ones, dim=1).long()


# from https://github.com/pytorch/fairseq/blob/89a2a0ccdebd0943b2878ff2150f8a5f836cc4aa/fairseq/modules/learned_positional_embedding.py
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        self.max_positions = self.num_embeddings

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        positions = make_positions(input)
        return super().forward(positions)

