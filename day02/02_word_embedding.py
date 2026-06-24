
"""
    Word Embedding Demo
    - jieba: Chinese word segmentation
    - Keras Tokenizer: word to index
    - nn.Embedding: learnable embedding matrix
    - TensorBoard: visualize embeddings
"""

import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn


def dm01_embedding_show():
    # Step 1: Prepare sentences
    sentence1 = "传智教育是一家上市公司,旗下有黑马程序员品牌.我实在黑马这里学习人工智能"
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    # Step 2: Segment with jieba
    word_list = []
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))
    print(word_list)

    # Step 3: Build vocab, word -> index (1-based)
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(word_list)
    print(f'word_index: {my_tokenizer.word_index}')

    # Step 4: Convert words to ids
    my_token_list = my_tokenizer.word_index.values()
    print(my_token_list)

    seq2id = my_tokenizer.texts_to_sequences(word_list)
    print(seq2id)

    # Step 5: Create embedding layer (vocab_size x dim)
    # Random init, will be learned during training
    embed = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    print(f'embed: {embed.weight.data}')
    print(f'embed.shape: {embed.weight.shape}')

    # Step 6: Save to TensorBoard for visualization
    my_summary = SummaryWriter(log_dir='./runs')
    my_summary.add_embedding(embed.weight.data, my_token_list)
    my_summary.close()

    # Step 7: Get embedding vector for each word
    for idx in range(len(my_tokenizer.word_index)):
        temp_vector = embed(torch.tensor(idx))
        print(f'temp_vector: {temp_vector}')

        word = my_tokenizer.index_word[idx + 1]
        print(f'word: {word}')


if __name__ == '__main__':
    dm01_embedding_show()
