"""
    seq2seq, RNN, N vs M
    英译法案例：使用 GPU 加速
"""


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


# 设置 GPU 设备（Mac M3 直接使用 MPS）
device = 'mps'
print(f'当前设备: {device}')

# ========================================
# Special tokens and constants
# ========================================
# Start of Sentence token
SOS_token = 0
# End of Sentence token
EOS_token = 1
# Maximum sentence length (including punctuation)
MAX_LENGTH = 10
# Data file path
data_path = './data/eng-fra-v2.txt'


# ========================================
# Utility functions
# ========================================
def normalizeString(s):
    """
    String normalization function
    Args:
        s: input string
    Returns:
        normalized string
    """
    # Convert to lowercase and strip whitespace
    s = s.lower().strip()
    # Add space before punctuation (. ! ?)
    # \1 represents the first captured group
    s = re.sub(r"([.!?])", r" \1", s)
    # Replace characters that are not letters or punctuation with space
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# ========================================
# Read data and build vocabulary
# ========================================
def my_getdata():
    """
    Read data from file and process it
    Returns:
        my_pairs: list of [english, french] pairs
    """
    # 1. Read file line by line
    # open().read().strip().split('\n')
    my_lines = open(data_path, encoding='utf-8').read().strip().split('\n')

    # 2. Process each line: split by tab and normalize
    my_pairs = [[normalizeString(s) for s in l.split('\t')] for l in my_lines]

    # 3. Build vocabulary for English and French
    # 3-1 Initialize word2index dictionaries
    english_word2index = {"SOS" : 0 , "EOS": 1}
    english_word_n = 2

    french_word2index = {"SOS" : 0, "EOS" : 1}
    french_word_n = 2

    # 3-2 Build word2index dictionaries
    # Traverse all data pairs and add words to dictionaries
    for pair in my_pairs:
        # Process English sentence (pair[0])
        # Split sentence by space: "i m ." → ["i", "m", "."]
        for word in pair[0].split(' '):
            # If word not in dictionary, add it
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1

        # Process French sentence (pair[1])
        # Split sentence by space: "j ai ans ." → ["j", "ai", "ans", "."]
        for word in pair[1].split(' '):
            # If word not in dictionary, add it
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # 3-3 Create reverse dictionaries: index → word
    # Using dictionary comprehension to reverse key-value pairs
    # {0: "SOS", 1: "EOS", ...} → {"SOS": 0, "EOS": 1, ...}
    english_index2word = {v : k for k, v in english_word2index.items()}
    french_index2word = {v : k for k, v in french_word2index.items()}

    # Return all dictionaries and vocabulary sizes
    return (english_index2word, english_word2index, english_word_n,
            french_word2index, french_index2word, french_word_n, my_pairs)


english_index2word, english_word2index, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()


# ========================================
# Dataset class for PyTorch DataLoader
# ========================================
class MyPairsDataset(Dataset):
    def __init__(self, my_pairs):
        """
        Initialize dataset
        Args:
            my_pairs: list of [english, french] pairs
        """
        # Store sample data
        self.my_pairs = my_pairs
        # Number of samples
        self.sample_len = len(my_pairs)

    def __len__(self):
        """Return number of samples"""
        return self.sample_len

    def __getitem__(self, index):
        """
        Get sample by index
        Args:
            index: sample index
        Returns:
            tensor_x: English tensor
            tensor_y: French tensor
        """
        # Fix index out of range
        index = min(max(index, 0), self.sample_len - 1)

        # Get sample x (English) and y (French)
        x = self.my_pairs[index][0]
        y = self.my_pairs[index][1]

        # Convert English text to numbers
        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        # Convert French text to numbers
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        # Note: tensor_x and tensor_y are 1D arrays
        # DataLoader will batch them into 2D arrays

        return tensor_x, tensor_y


def get_dataloader():
    """Create DataLoader for training"""
    my_dataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
    return my_dataloader


# ========================================
# GRU Encoder
# ========================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Initialize GRU Encoder
        Args:
            input_size: vocabulary size (number of unique words)
            hidden_size: embedding dimension (feature size per word)
        """
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Embedding layer: converts word indices to dense vectors
        # input_size: vocabulary size, hidden_size: embedding dimension
        self.embedding = nn.Embedding(input_size, hidden_size)

        # GRU layer: processes sequences
        # batch_first=True: input shape (batch, seq, feature)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        """
        Forward pass
        Args:
            input: input tensor (word indices)
            hidden: initial hidden state
        Returns:
            output: GRU output for all time steps
            hidden: final hidden state (context vector)
        """
        # Embedding: [batch, seq_len] → [batch, seq_len, hidden_size]
        output = self.embedding(input)

        # GRU: process sequence
        # output: all hidden states [batch, seq_len, hidden_size]
        # hidden: last hidden state [1, batch, hidden_size]
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def init_hidden(self):
        """Initialize hidden state with zeros"""
        # Shape: [1, 1, hidden_size]
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ========================================
# GRU Decoder (without Attention)
# ========================================
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        Initialize GRU Decoder
        Args:
            hidden_size: embedding dimension
            output_size: output vocabulary size (for prediction)
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Embedding layer: converts word indices to dense vectors
        self.embedding = nn.Embedding(output_size, hidden_size)

        # GRU layer: processes decoder input sequence
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Fully connected layer: maps hidden state to vocabulary
        self.out = nn.Linear(hidden_size, output_size)

        # LogSoftmax: outputs log probabilities for each word
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """
        Forward pass
        Args:
            input: decoder input (previous word indices)
            hidden: hidden state from encoder or previous step
        Returns:
            output: log probabilities for next word
            hidden: updated hidden state
        """
        # Embedding: [batch, 1] → [batch, 1, hidden_size]
        output = self.embedding(input)

        # ReLU activation
        output = F.relu(output)

        # GRU: [batch, 1, hidden_size] → [batch, 1, hidden_size]
        output, hidden = self.gru(output, hidden)

        # Linear + Softmax: [1, hidden_size] → [output_size]
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def init_hidden(self):
        """Initialize hidden state with zeros"""
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ========================================
# GRU Decoder (with Attention)
# ========================================
class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """
        Initialize Attention Decoder
        Args:
            output_size: target vocabulary size (French)
            hidden_size: embedding dimension
            dropout_p: dropout probability (default 0.1)
            max_length: maximum sequence length (default 10)
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Embedding layer: converts word indices to dense vectors
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # Attention layer: Q and K concatenated → attention scores
        # input: hidden_size * 2 (embedded + hidden), output: max_length
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # Combine layer: merge embedded + attention applied
        # input: hidden_size * 2, output: hidden_size
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Dropout layer: prevent overfitting
        self.dropout = nn.Dropout(self.dropout_p)

        # GRU layer: process decoder input
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # Output layer: map to vocabulary
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # LogSoftmax: output log probabilities
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):
        """
        Forward pass with attention
        Args:
            input: decoder input (word indices)
            hidden: hidden state from previous step
            encoder_outputs: all encoder outputs [seq_len, hidden_size]
        Returns:
            output: log probabilities for next word
            hidden: updated hidden state
            attn_weights: attention weights (for visualization)
        """
        # Embedding: [1, 1] → [1, 1, hidden_size]
        embedded = self.embedding(input)
        embedded = F.relu(embedded)

        # Attention calculation:
        # Concatenate embedded and hidden → [1, hidden_size * 2]
        # → Linear → [1, max_length] → softmax → attention weights [1, max_length]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )

        # Apply attention to encoder outputs:
        # attn_weights [1, 1, max_length] × encoder_outputs [1, max_length, hidden_size]
        # → attn_applied [1, 1, hidden_size]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # Combine: concatenate embedded and attention applied
        # [1, hidden_size] + [1, hidden_size] → [1, hidden_size * 2]
        output = torch.cat((embedded[0], attn_applied[0]), 1)

        # Linear: [1, hidden_size * 2] → [1, hidden_size]
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        # GRU: [1, 1, hidden_size] → [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)

        # Output: [1, hidden_size] → [output_size]
        output = self.softmax(self.out(output[0]))

        return output, hidden, attn_weights

    def init_hidden(self):
        """Initialize hidden state with zeros"""
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ========================================
# Training parameters
# ========================================
epochs = 5
learning_rate = 1e-4
teacher_forcing_ratio = 0.5
print_interval = 1000
plot_interval = 100


# ========================================
# Training function (single batch)
# ========================================
def train_one_batch(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    """
    Train one batch of data
    Args:
        x: input tensor (English)
        y: target tensor (French)
        encoder: EncoderRNN model
        decoder: AttnDecoderRNN model
        encoder_optimizer: optimizer for encoder
        decoder_optimizer: optimizer for decoder
        criterion: loss function
    Returns:
        loss: average loss for this batch
    """
    # 1. Encode: get encoder outputs and hidden state
    encode_hidden = encoder.init_hidden()
    encode_output, encode_hidden = encoder(x, encode_hidden)

    # 2. Prepare decoder inputs
    # Pad encoder output to MAX_LENGTH for attention
    encode_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for idx in range(x.shape[1]):
        encode_output_c[idx] = encode_output[0, idx]

    # Use encoder's last hidden as decoder's initial hidden
    decode_hidden = encode_hidden

    # Start with SOS token as first decoder input
    input_y = torch.tensor([[SOS_token]], device=device)

    # Initialize loss
    myloss = 0.0
    y_len = y.shape[1]

    # Decide whether to use teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher Forcing: use real target as next input
        for idx in range(y_len):
            output_y, decode_hidden, attn_weight = decoder(input_y, decode_hidden, encode_output_c)
            target_y = y[0][idx].view(1)
            myloss = myloss + criterion(output_y, target_y)
            # Use real target as next input
            input_y = y[0][idx].view(1, -1)
    else:
        # Without Teacher Forcing: use prediction as next input
        for idx in range(y_len):
            output_y, decode_hidden, attn_weight = decoder(input_y, decode_hidden, encode_output_c)
            target_y = y[0][idx].view(1)
            myloss = myloss + criterion(output_y, target_y)
            # Get prediction and use as next input
            topv, topi = output_y.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            input_y = topi.detach()

    # 3. Backpropagation
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    myloss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return myloss.item() / y_len


# ========================================
# Training function (full training loop)
# ========================================
def train_seq2seq():
    """
    Train the Seq2Seq model
    Returns:
        plot_loss_list: list of losses for plotting
    """
    # 1. Create dataset and dataloader
    mypairsdataset = MyPairsDataset(my_pairs)
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 2. Create models
    my_encoderrnn = EncoderRNN(english_word_n, 256).to(device)
    my_attndecoderrnn = AttnDecoderRNN(
        output_size=french_word_n,
        hidden_size=256,
        dropout_p=0.1,
        max_length=MAX_LENGTH
    ).to(device)

    # 3. Create optimizers
    myadam_encode = optim.Adam(my_encoderrnn.parameters(), lr=learning_rate)
    myadam_decode = optim.Adam(my_attndecoderrnn.parameters(), lr=learning_rate)

    # 4. Create loss function
    mycriterion = nn.NLLLoss()

    # 5. Training loop
    plot_loss_list = []
    smoothed_loss = 0.0
    smooth_factor = 0.95  # EMA smoothing factor

    for epoch_idx in range(1, epochs + 1):
        print_loss_total = 0.0
        plot_loss_total = 0.0
        starttime = time.time()

        for item, (x, y) in enumerate(tqdm(mydataloader, desc=f'Epoch {epoch_idx}'), start=1):
            # Train one batch
            myloss = train_one_batch(
                x, y, my_encoderrnn, my_attndecoderrnn,
                myadam_encode, myadam_decode, mycriterion
            )
            print_loss_total += myloss
            plot_loss_total += myloss

            # Print loss every print_interval batches
            if item % print_interval == 0:
                print_loss_avg = print_loss_total / print_interval
                print_loss_total = 0
                tqdm.write('轮次%d 损失%.6f 时间:%d' % (epoch_idx, print_loss_avg, time.time() - starttime))

            # Record loss for plotting every plot_interval batches (with EMA smoothing)
            if item % plot_interval == 0:
                plot_loss_avg = plot_loss_total / plot_interval
                # Exponential moving average
                smoothed_loss = smooth_factor * smoothed_loss + (1 - smooth_factor) * plot_loss_avg
                plot_loss_list.append(smoothed_loss)
                plot_loss_total = 0

        # Save model after each epoch
        torch.save(my_encoderrnn.state_dict(), './model/my_encoderrnn_%d.pth' % epoch_idx)
        torch.save(my_attndecoderrnn.state_dict(), './model/my_attndecoderrnn_%d.pth' % epoch_idx)

    # Plot loss curve
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/s2s_loss.png')
    plt.show()

    return plot_loss_list


# ========================================
# Main entry point
# ========================================
if __name__ == '__main__':
    train_seq2seq()
