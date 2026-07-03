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
# Todo 1: Define special tokens
# ========================================
# Start of Sentence token
SOS_token = 0
# End of Sentence token
EOS_token = 1
# Maximum sentence length (including punctuation)
MAX_LENGTH = 10
# Data file path
data_path = './data/eng-fra-v2.txt'

# Todo 2: ...


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
    # print(f'my_lines--->', len(my_lines))

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
    my_dataset = MyPairsDataset(my_pairs)

    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)

    for i, (x, y) in enumerate(my_dataloader):
        print(f'x: {x.shape}, {x}')
        print(f'y: {y.shape}, {y}')

        break

    return my_dataloader



# Test the function
if __name__ == '__main__':
    get_dataloader()
    pass
