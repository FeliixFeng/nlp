"""
    Complete Transformer Model

    Structure:
        Source Embedding → Encoder →
        Target Embedding → Decoder →
        Generator → Output Probabilities
"""

import copy
import torch
from dm08_output import *


# todo 1. Complete Transformer
class EncoderDecoder(nn.Module):
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
        encoder_result = self.encode(source_x, source_mask)

        # Decode with encoder output
        decoder_result = self.decode(target_y, encoder_result, source_mask, target_mask)

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
    """
    Build complete Transformer model
    Returns:
        model: EncoderDecoder instance
    """
    c = copy.deepcopy  # Deep copy function

    # Hyperparameters
    vocab_size = 1000
    d_model = 512
    d_ff = 2048
    n_heads = 8
    n_layers = 6
    dropout_p = 0.2

    # Source embedding
    source_embed = Embedding(vocab_size=vocab_size, d_model=d_model)
    source_position = PositionalEncoding(d_model=d_model, dropout=dropout_p)

    # Multi-Head Attention
    self_attn = MultiHeadAttention(embed_dim=d_model, head=n_heads)

    # Feed-Forward Network
    ff = FeedForward(d_model=d_model, d_ff=d_ff)

    # Encoder
    my_encoder_layer = EncoderLayer(d_model=d_model, self_attn=self_attn, feed_forward=ff, dropout=dropout_p)
    my_encoder = Encoder(my_encoder_layer, N=n_layers)

    # Target embedding (same architecture, different weights)
    target_embed = c(source_embed)
    target_position = c(source_position)

    # Decoder attention layers
    self_attn1 = c(self_attn)
    source_attn1 = c(self_attn)
    ff1 = c(ff)

    # Decoder
    my_decoder_layer = DecoderLayer(d_model=d_model, self_attn=self_attn1, src_attn=source_attn1, feed_forward=ff1, dropout=dropout_p)
    my_decoder = Decoder(my_decoder_layer, N=n_layers)

    # Generator (output layer)
    generator = Generator(d_model=d_model, vocab_size=vocab_size)

    # Assemble complete Transformer
    my_transformer = EncoderDecoder(
        nn.Sequential(source_embed, source_position),
        my_encoder,
        nn.Sequential(target_embed, target_position),
        my_decoder,
        generator
    )

    print(my_transformer)

    # Test forward pass
    source_x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target_y = torch.LongTensor([[3, 8, 6, 4], [9, 6, 2, 6]])
    source_mask = torch.zeros(8, 4, 4)
    target_mask = c(source_mask)

    print('=' * 20)
    result = my_transformer(source_x, target_y, source_mask, target_mask)

    print(f'Result shape: {result.shape}')

    return my_transformer




if __name__ == '__main__':
    make_model()