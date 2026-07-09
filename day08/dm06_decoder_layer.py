"""
    Transformer Decoder Layer

    Structure:
        1. Masked Multi-Head Self-Attention + Residual + Norm
        2. Multi-Head Cross-Attention + Residual + Norm
        3. Feed-Forward + Residual + Norm
"""

import torch
import copy
import math
from dm05_encoder import *


# todo 1. Decoder Layer
class DecoderLayer(nn.Module):
    """
    Single Decoder Layer
    Contains masked self-attention, cross-attention, and feed-forward sublayers
    """
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.1):
        """
        Initialize Decoder Layer
        Args:
            d_model: embedding dimension
            self_attn: Masked Multi-Head Attention for self-attention
            src_attn: Multi-Head Attention for cross-attention (encoder-decoder)
            feed_forward: Feed-Forward Network
            dropout: dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        # Three sublayers
        self.layers = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, encoder_output, source_mask, target_mask):
        """
        Forward pass
        Args:
            x: decoder input [batch, seq, d_model]
            encoder_output: encoder output [batch, seq, d_model]
            source_mask: mask for encoder output
            target_mask: mask for decoder self-attention (causal mask)
        Returns:
            output tensor [batch, seq, d_model]
        """
        # Sublayer 1: Masked Multi-Head Self-Attention
        # Prevents attending to future positions
        x = self.layers[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # Sublayer 2: Multi-Head Cross-Attention
        # Query from decoder, Key/Value from encoder
        x = self.layers[1](x, lambda x: self.src_attn(x, encoder_output, encoder_output, source_mask))

        # Sublayer 3: Feed-Forward Network
        x = self.layers[2](x, lambda x: self.feed_forward(x))

        return x


# todo 2. Test Decoder Layer
def use_decoder_layer():
    """Test Decoder Layer"""
    # Create sample target input
    y = torch.LongTensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ])

    # Embedding + Positional Encoding
    my_embed = Embedding(1000, 512)
    embed_y = my_embed(y)
    print(f'Embedding shape: {embed_y.shape}')

    my_position = PositionalEncoding(512, 0.1)
    position_y = my_position(embed_y)

    # Create attention layers
    multi_attn = MultiHeadAttention(512, 8)
    self_attn = copy.deepcopy(multi_attn)
    src_attn = copy.deepcopy(multi_attn)
    ff = FeedForward(512, 2048)

    # Get encoder output
    encoder_output = use_encoder()

    # Create masks
    source_mask = torch.zeros(8, 4, 4)
    target_mask = torch.ones(8, 4, 4)

    # Create and test decoder layer
    my_decoder_layer = DecoderLayer(512, self_attn, src_attn, ff)
    result = my_decoder_layer(position_y, encoder_output, source_mask, target_mask)

    print(f'Decoder Layer output shape: {result.shape}')
    print(f'Decoder Layer:\n{my_decoder_layer}')


if __name__ == '__main__':
    use_decoder_layer()
