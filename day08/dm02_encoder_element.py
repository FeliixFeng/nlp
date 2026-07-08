"""
    Transformer Encoder Components

    Components:
        1. Masked Self-Attention (Multi-Head Attention)
        2. Feed-Forward Network
        3. Residual Connection + Layer Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from dm01_input import *

# todo 1. Test upper triangular matrix
def dm01_test_triu():
    """Test np.triu with different k values"""
    arr = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
    ]

    print("k=0 (main diagonal):")
    print(np.triu(arr, k=0))
    print("k=1 (above main diagonal):")
    print(np.triu(arr, k=1))
    print("k=-1 (below main diagonal):")
    print(np.triu(arr, k=-1))

# todo 2. Create lower triangular mask
def dm02_test_triu(size):
    """
    Create lower triangular mask for self-attention
    Args:
        size: matrix size
    Returns:
        mask: lower triangular tensor [1, size, size]
    """
    # Create upper triangular matrix with k=1 (above diagonal)
    temp = np.triu(m=np.ones((1, size, size)), k=1).astype('uint8')

    # Convert to lower triangular: 1 - upper_triangular
    return torch.from_numpy(1 - temp)


# todo 3. Visualize attention mask
def dm03_test_mask():
    """Visualize the lower triangular mask"""
    plt.figure(figsize=(5, 5))
    plt.imshow(dm02_test_triu(size=20)[0])
    plt.title("Attention Mask (Lower Triangular)")
    plt.colorbar()
    plt.show()



# todo 4. Scaled Dot-Product Attention
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot-Product Attention
    Args:
        query: query tensor [batch, heads, seq, d_k]
        key: key tensor [batch, heads, seq, d_k]
        value: value tensor [batch, heads, seq, d_k]
        mask: optional mask tensor
        dropout: optional dropout layer
    Returns:
        output: attention output [batch, heads, seq, d_k]
        p_attn: attention weights [batch, heads, seq, seq]
    """
    d_k = query.size(-1)

    # Compute attention scores: Q * K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (set masked positions to -infinity)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    p_attn = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Compute weighted sum: attention_weights * V
    return torch.matmul(p_attn, value), p_attn

# todo 5. Test attention mechanism
def use_attention():
    """Test attention with and without mask"""
    position_x = use_position()

    query = key = value = position_x

    # Without mask
    result1, p_attn = attention(query, key, value)
    print(f'Without mask: {result1.shape}, attn: {p_attn.shape}')

    # With mask
    mask = torch.zeros(2, 4, 4)
    result2, p_attn = attention(query, key, value, mask=mask)
    print(f'With mask: {result2.shape}, attn: {p_attn.shape}')




# todo 6. Clone module N times
def clones(module, N):
    """
    Clone a module N times
    Args:
        module: nn.Module to clone
        N: number of clones
    Returns:
        ModuleList of N cloned modules
    """
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))

# todo 7. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    Splits embedding into multiple heads for parallel attention computation
    """
    def __init__(self, embed_dim, head, dropout_p=0.1):
        """
        Initialize Multi-Head Attention
        Args:
            embed_dim: embedding dimension (must be divisible by head)
            head: number of attention heads
            dropout_p: dropout probability
        """
        super().__init__()

        # Check if embed_dim is divisible by head
        assert embed_dim % head == 0

        self.d_k = embed_dim // head  # Dimension per head
        self.head = head  # Number of heads

        # 4 linear layers: Q, K, V, and output
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(p=dropout_p)
        self.attn = None  # Store attention weights for visualization

    def forward(self, query, key, value, mask=None):
        """
        Forward pass
        Args:
            query: query tensor [batch, seq, embed_dim]
            key: key tensor [batch, seq, embed_dim]
            value: value tensor [batch, seq, embed_dim]
            mask: optional mask tensor
        Returns:
            output: attention output [batch, seq, embed_dim]
        """
        if mask is not None:
            mask = mask.unsqueeze(0)

        self.batch = query.size(0)

        # Step 1: Linear projection and reshape to multi-head
        # [batch, seq, embed_dim] → [batch, seq, head, d_k] → [batch, head, seq, d_k]
        query, key, value = [
            model(x).view(self.batch, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))
        ]

        # Step 2: Compute attention
        x, self.attn = attention(query, key, value, mask, self.dropout)

        # Step 3: Reshape back and apply output linear
        # [batch, head, seq, d_k] → [batch, seq, head, d_k] → [batch, seq, embed_dim]
        attn_x = x.transpose(1, 2).contiguous().view(self.batch, -1, self.head * self.d_k)

        return self.linears[-1](attn_x)


# todo 8. Test Multi-Head Attention
def use_multihead():
    """Test Multi-Head Attention with mask"""
    my_attention = MultiHeadAttention(512, 8)
    position_x = use_position()
    query = key = value = position_x
    mask = torch.zeros(8, 4, 4)

    result = my_attention(query, key, value, mask)
    print(f'Multi-Head Attention: {result.shape}')

    return result


# todo 9. Feed-Forward Network
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    Two linear layers with ReLU activation
    """
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        """
        Initialize Feed-Forward Network
        Args:
            d_model: input/output dimension
            d_ff: hidden dimension (typically 4 * d_model)
            dropout_p: dropout probability
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)  # Expand
        self.linear2 = nn.Linear(d_ff, d_model)  # Contract
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor [batch, seq, d_model]
        Returns:
            output tensor [batch, seq, d_model]
        """
        x = self.linear1(x)
        x = self.dropout(F.relu(x))
        x = self.linear2(x)
        return x


# todo 10. Test Feed-Forward Network
def use_ff():
    """Test Feed-Forward Network with Multi-Head Attention output"""
    attn_x = use_multihead()
    my_ff = FeedForward(512, 2048)

    result = my_ff(attn_x)

    print(f'Feed-Forward: {result.shape}')
    return result









if __name__ == '__main__':
    # dm01_test_triu()
    # print(dm02_test_triu(size=5))
    # dm03_test_mask()
    # use_attention()
    # use_multihead()
    use_ff()