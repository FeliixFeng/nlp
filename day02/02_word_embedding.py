
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn


def dm01_embedding_show():
    # Two example sentences
    sentence1 = "传智教育是一家上市公司,旗下有黑马程序员品牌.我实在黑马这里学习人工智能"
    sentence2 = "我爱自然语言处理"

    sentences = [sentence1, sentence2]

    # Tokenize with jieba
    word_list = []
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))
    print(word_list)

    # Build word index with Keras Tokenizer
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(word_list)

    print(f'word_index: {my_tokenizer.word_index}')

    # Convert words to sequences (word -> id)
    my_token_list = my_tokenizer.word_index.values()
    print(my_token_list)

    seq2id = my_tokenizer.texts_to_sequences(word_list)
    print(seq2id)

    # Create embedding layer: vocab_size x embedding_dim
    embed = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    print(f'embed: {embed.weight.data}')
    print(f'embed.shape: {embed.weight.shape}')


if  __name__ == '__main__':
    dm01_embedding_show()
