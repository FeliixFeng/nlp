"""
    Transformer Decoder

    Structure:
        N × DecoderLayer + Final LayerNorm
"""

from dm06_decoder_layer import *


# todo 1. Decoder
class Decoder(nn.Module):
    """
    Transformer Decoder
    Stacks N decoder layers and applies final layer normalization
    """
    def __init__(self, layer, N):
        """
        Initialize Decoder
        Args:
            layer: DecoderLayer module to clone
            N: number of layers (typically 6)
        """
        super().__init__()

        # Clone N decoder layers
        self.layers = clones(layer, N)

        # Final layer normalization
        self.norm = LayerNorm(layer.d_model)

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
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        # Apply final layer normalization
        return self.norm(x)


# todo 2. Test Decoder
def use_decoder():
    """Test Decoder"""
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

    # Create decoder layer and decoder
    my_decoder_layer = DecoderLayer(512, self_attn, src_attn, ff)
    my_decoder = Decoder(my_decoder_layer, 6)

    # Test decoder
    result = my_decoder(position_y, encoder_output, source_mask, target_mask)
    print(f'Decoder output shape: {result.shape}')
    print(f'Decoder:\n{my_decoder}')


if __name__ == '__main__':
    use_decoder()
