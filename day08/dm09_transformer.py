"""
    Complete Transformer Model

    Structure:
        Source Embedding → Encoder →
        Target Embedding → Decoder →
        Generator → Output Probabilities
"""

from dm08_output import *


# todo 1. Complete Transformer
class MyTransformer(nn.Module):
    """
    Complete Transformer Model
    Combines Encoder, Decoder, Embeddings, and Generator
    """
    def __init__(self, source_embed, encoder, target_embed, decoder, generator):
        """
        Initialize Transformer
        Args:
            source_embed: source language embedding layer
            encoder: Transformer Encoder
            target_embed: target language embedding layer
            decoder: Transformer Decoder
            generator: output layer (Linear + Softmax)
        """
        super().__init__()

        self.source_embed = source_embed
        self.encoder = encoder
        self.target_embed = target_embed
        self.decoder = decoder
        self.generator = generator

    def forward(self, source_x, target_y, source_mask, target_mask):
        """
        Forward pass (for training)
        Args:
            source_x: source sentence indices [batch, src_seq]
            target_y: target sentence indices [batch, tgt_seq]
            source_mask: mask for source
            target_mask: mask for target (causal mask)
        Returns:
            output: log probabilities [batch, tgt_seq, vocab_size]
        """
        # Encode source
        encoder_result = self.encoder(source_x, source_mask)

        # Decode with encoder output
        decoder_result = self.decoder(target_y, encoder_result, source_mask, target_mask)

        # Generate output probabilities
        output = self.generator(decoder_result)

        return output

    def encode(self, source_x, source_mask):
        """
        Encode source sentence (for inference)
        Args:
            source_x: source sentence indices [batch, src_seq]
            source_mask: mask for source
        Returns:
            encoder output [batch, src_seq, d_model]
        """
        embed_x = self.source_embed(source_x)
        return self.encoder(embed_x, source_mask)

    def decode(self, target_y, encoder_result, source_mask, target_mask):
        """
        Decode target sentence (for inference)
        Args:
            target_y: target sentence indices [batch, tgt_seq]
            encoder_result: encoder output
            source_mask: mask for source
            target_mask: mask for target
        Returns:
            decoder output [batch, tgt_seq, d_model]
        """
        embed_y = self.target_embed(target_y)
        return self.decoder(embed_y, encoder_result, source_mask, target_mask)


# todo 2. Build Transformer model
def make_model():
    """Build complete Transformer model"""
    # TODO: Instantiate all components and return MyTransformer
    pass


if __name__ == '__main__':
    pass
