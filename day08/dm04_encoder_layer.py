"""
    Transformer Encoder Layer

    Structure:
        1. Self-Attention + Residual + LayerNorm
        2. Feed-Forward + Residual + LayerNorm
"""

from dm01_input import *
from dm02_encoder_element import *
from dm03_encoder_sublayer import *


# todo 1. Encoder Layer
class EncoderLayer(nn.Module):
    """
    Single Encoder Layer
    Contains self-attention and feed-forward sublayers
    """
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        """
        Initialize Encoder Layer
        Args:
            d_model: embedding dimension
            self_attn: Multi-Head Attention module
            feed_forward: Feed-Forward Network module
            dropout: dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # Two sublayers: self-attention + feed-forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        """
        Forward pass
        Args:
            x: input tensor [batch, seq, d_model]
            mask: attention mask
        Returns:
            output tensor [batch, seq, d_model]
        """
        # Sublayer 1: Self-Attention + Residual + Norm
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # Sublayer 2: Feed-Forward + Residual + Norm
        x = self.sublayer[1](x, lambda x: self.feed_forward(x))

        return x


# todo 2. Test Encoder Layer
def use_encoder_layer():
    """Test Encoder Layer"""
    x = use_position()

    multi_head = MultiHeadAttention(512, 8)
    feed_forward = FeedForward(512, 2048)

    encoder_layer = EncoderLayer(512, multi_head, feed_forward)

    mask = torch.zeros(8, 4, 4)
    output = encoder_layer(x, mask)

    print(f'Encoder Layer output shape: {output.shape}')
    print(f'Encoder Layer:\n{encoder_layer}')


if __name__ == '__main__':
    use_encoder_layer()
