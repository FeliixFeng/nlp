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

class PositionalEncoding(nn.Module):
    """
    Positional Encoding layer
    Adds position information to embeddings using sin/cos functions
    """
    def __init__(self, d_model, dropout, max_len=60):
        """
        Initialize Positional Encoding
        Args:
            d_model: embedding dimension
            dropout: dropout probability
            max_len: maximum sequence length
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)

        # Divisor term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Calculate sin/cos for even/odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: embedded input [batch, seq_len, d_model]
        Returns:
            x: input with positional encoding added
        """
        # Add positional encoding (truncate to match input length)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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

def use_position():
    """Test Embedding + Positional Encoding"""
    vocab_size, d_model = 1000, 512

    my_embed = Embedding(vocab_size, d_model)

    x = torch.tensor([
        # ['我', '爱', '吃', '汤圆'],
        # ['我', '爱', '吃', '蛋酒']
        [100, 2, 421, 300],
        [500, 888, 306, 509]
    ])

    result = my_embed(x)
    my_position = PositionalEncoding(d_model, 0.1)

    result = my_position(result)

    return result


if __name__ == '__main__':
    # dm_test()
    use_position()
