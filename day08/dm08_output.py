"""
    Transformer Output Layer (Generator)

    Structure:
        Linear Layer + Log Softmax
"""

from dm07_decoder import *


# todo 1. Generator (Output Layer)
class Generator(nn.Module):
    """
    Output Layer
    Maps decoder output to vocabulary probabilities
    """
    def __init__(self, d_model, vocab_size):
        """
        Initialize Generator
        Args:
            d_model: input dimension (from decoder)
            vocab_size: output vocabulary size
        """
        super().__init__()

        # Linear layer: d_model → vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: decoder output [batch, seq, d_model]
        Returns:
            log_probs: log probabilities [batch, seq, vocab_size]
        """
        x = self.linear(x)

        # Log Softmax for probability distribution
        return F.log_softmax(x, dim=-1)


# todo 2. Test Generator
def use_generator():
    """Test Generator"""
    # Get decoder output
    result = use_decoder()

    # Create generator
    generator = Generator(512, 1000)

    # Generate output
    output = generator(result)

    print(f'Generator output shape: {output.shape}')

    # Check probability distribution
    log_probs = output[0, 0]
    probs = torch.exp(log_probs)
    print(f'First word probability sum: {torch.sum(probs):.4f} (should be ~1.0)')


if __name__ == '__main__':
    use_generator()
