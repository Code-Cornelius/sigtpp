from typing import Optional

import torch
import torch.nn as nn

from src.nn.embeddings.time import TrigoTimeEmbedding


class PositionEmbedding(nn.Module):
    """
    A module that computes position embeddings for a sequence using a trigonometric embedding.
    *Use it when you pass a whole sequence to a module.*

    Args:
        embed_size (int): The dimensionality of the embeddings.
        max_length (int): The maximum length of the sequence that the module can handle.

    Attributes:
        max_length (int): Maximum length for sequences.
        trigo_embedding (TrigoTimeEmbedding): Instance of trigonometric time embedding to compute position embeddings.
    """

    def __init__(self, embed_size: int, max_length: int):
        super().__init__()
        self.max_length = max_length
        self.trigo_embedding = TrigoTimeEmbedding(embed_size, 0, max_length)

    def forward(
        self, sequence: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute position embeddings based on either a sequence or directly provided positions.

        Args:
            sequence (Optional[torch.Tensor]): Input sequence of shape (batch_size, sequence_length). Either this or
                                               `positions` must be provided.
            positions (Optional[torch.Tensor]): Precomputed positions of shape (batch_size, sequence_length, 1).
                                                Either this or `sequence` must be provided.

        Returns:
            torch.Tensor: The resulting position embeddings of shape (batch_size, sequence_length, embed_size).

        Raises:
            AssertionError: If both `sequence` and `positions` are provided, or if neither is provided.
        """
        assert (sequence is not None) ^ (positions is not None), (
            f"Invalid input: "
            f"{'Both sequence and positions are None' if sequence is None and positions is None else 'Both sequence and positions are provided'}."
            f" Got sequence={sequence} and positions={positions}. Provide one or the other, but not both or neither."
        )

        if sequence is not None:
            # Compute positions from sequence
            positions = torch.arange(sequence.shape[1], device=sequence.device, dtype=torch.float32).unsqueeze(0)
            positions = positions.expand(sequence.shape[0], -1).unsqueeze(-1)  # Expand to match the sequence batch size

        embeddings = self.trigo_embedding(positions)

        return embeddings

    @property
    def embed_size(self) -> int:
        return self.trigo_embedding.embed_size


# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    embed_size = 6
    max_length = 10
    sequence = torch.randn(2, 5)  # Example sequence of shape (batch_size, sequence_length)

    position_embedding = PositionEmbedding(embed_size, max_length)
    embeddings_from_sequence = position_embedding(sequence=sequence)
    print("Generated embeddings from sequence:", embeddings_from_sequence)

    positions = torch.tensor([[0.0], [1.0], [2.0]])  # Example positions provided directly
    embeddings_from_positions = position_embedding(positions=positions)
    print("Generated embeddings from positions:", embeddings_from_positions)
