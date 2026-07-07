"""
    Transformer - Input Layer

    Word Embedding: converts word indices to dense vectors
    Positional Encoding: adds position information to embeddings
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


class Embedding(nn.Module):
    """
    Word Embedding layer
    Converts word indices to dense vectors and scales by sqrt(d_model)
    """
    def __init__(self, vocab_size, d_model):
        """
        Initialize Embedding layer
        Args:
            vocab_size: vocabulary size (number of unique words)
            d_model: embedding dimension (feature size per word)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layer: maps word indices to dense vectors
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor (word indices) [batch, seq_len]
        Returns:
            embedded: scaled embeddings [batch, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) to balance gradients
        # Prevents gradient vanishing or explosion
        return self.embed(x) * math.sqrt(self.d_model)


def dm_test():
    """Test Embedding layer"""
    vocab_size, d_model = 1000, 512
    my_embed = Embedding(vocab_size, d_model)

    # Sample input: 2 sentences, each with 4 words
    x = torch.tensor([
        # ['我', '爱', '吃', '汤圆'],
        # ['我', '爱', '吃', '蛋酒']
        [100, 2, 421, 300],
        [500, 888, 306, 509]
    ])

    result = my_embed(x)

    print(f'result: {result}, result.shape: {result.shape}')


if __name__ == '__main__':
    dm_test()

