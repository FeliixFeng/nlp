"""
    Transformer Encoder

    Structure:
        N × EncoderLayer + Final LayerNorm
"""

from dm04_encoder_layer import *


# todo 1. Encoder
class Encoder(nn.Module):
    """
    Transformer Encoder
    Stacks N encoder layers and applies final layer normalization
    """
    def __init__(self, layer, N):
        """
        Initialize Encoder
        Args:
            layer: EncoderLayer module to clone
            N: number of layers (typically 6)
        """
        super().__init__()

        # Clone N encoder layers
        self.layers = clones(layer, N)

        # Final layer normalization
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        """
        Forward pass
        Args:
            x: input tensor [batch, seq, d_model]
            mask: attention mask
        Returns:
            output tensor [batch, seq, d_model]
        """
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)

        # Apply final layer normalization
        return self.norm(x)


# todo 2. Test Encoder
def use_encoder():
    """Test Encoder"""
    x = use_position()

    multi_head = MultiHeadAttention(512, 8)
    feed_forward = FeedForward(512, 2048)

    encoder_layer = EncoderLayer(512, multi_head, feed_forward)

    # Create encoder with 3 layers
    encoder = Encoder(encoder_layer, 3)

    mask = torch.zeros(8, 4, 4)
    encoder_output = encoder(x, mask)

    print(f'Encoder output shape: {encoder_output.shape}')

    return encoder_output


if __name__ == '__main__':
    use_encoder()
