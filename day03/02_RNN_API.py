"""
    Case: Demo traditional RNN implementation (演示传统RNN代码实现)

    RNN Introduction (RNN介绍):
        Overview (概述):
            Recurrent Neural Network 循环神经网络, mainly handles sequential data 主要处理序列数据.
            Sequential data: later data depends on previous data (后面的数据对前面的数据有依赖).

        Classification by Input & Output (按输入输出划分):
            N vs N: input n, output n - poetry, couplets (对联，合辙诗词)
            N vs 1: sentiment analysis, text classification (情感分析，文本分类，意图识别)
            1 vs N: generation tasks, image captioning (生成类任务，看图说话)
            N vs M: Seq2Seq, machine translation (机器翻译) # Most used 用的最多

        Classification by Internal Structure (按内部结构划分):
            Traditional RNN (传统RNN):
                Input layer, Hidden layer(Embedding, RNN layer), Output layer
                (输入层，隐藏层(词嵌入层，循环网络层)，输出层)
            LSTM:
                Forget gate, Input gate, Cell state, Output gate
                (遗忘门，输入门，细胞状态，输出门)
            Bi-LSTM:
                Forward LSTM + Backward LSTM, then concatenate
                (从前往后做一次LSTM，从后往前做一次LSTM，然后拼接)
            GRU:
                Reset gate, Update gate (重置门，更新门)
            Bi-GRU:
                Forward GRU + Backward GRU, then concatenate
                (从前往后做一次GRU，从后往前做一次GRU，然后拼接)
"""
import torch
import torch.nn as nn



def dm_rnn_for_base():
    # 1. Create RNN model
    # para1: input_size (embedding_dim)
    # para2: hidden_size (hidden_dim)
    # para3: num_layers (number of hidden layers)
    rnn = nn.RNN(5, 6, 1)

    # 2. Prepare input data
    # Shape: (seq_len, batch_size, input_size)
    input = torch.randn(1, 3, 5)

    # 3. Initialize hidden state
    # Shape: (num_layers, batch_size, hidden_size)
    h0 = torch.randn(1, 3, 6)

    # 4. Run RNN
    output, hn = rnn(input, h0)

    print(f'output: {output}, output.shape: {output.shape}')
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')
    print(f'rnn: {rnn}')

def dm_rnn_for_sequence():
    # 1. Create RNN model
    # para1: input_size (embedding_dim)
    # para2: hidden_size (hidden_dim)
    # para3: num_layers (number of hidden layers)
    rnn = nn.RNN(5, 6, 1)

    # 2. Prepare input data (longer sequence: 20 steps)
    # Shape: (seq_len, batch_size, input_size)
    input = torch.randn(20, 3, 5)

    # 3. Initialize hidden state
    # Shape: (num_layers, batch_size, hidden_size)
    h0 = torch.randn(1, 3, 6)

    # 4. Run RNN
    output, hn = rnn(input, h0)

    print(f'output: {output}, output.shape: {output.shape}')
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')
    print(f'rnn: {rnn}')



def dm_lstm_base():
    # LSTM: adds cell state to remember long-term dependencies
    # (LSTM增加了细胞状态，用于记忆长期依赖)
    lstm = nn.LSTM(5, 6, 1)

    # Input shape: (seq_len, batch_size, input_size)
    input = torch.randn(1, 3, 5)

    # LSTM needs both hidden state (h0) and cell state (c0)
    # (LSTM需要同时初始化隐藏状态和细胞状态)
    h0 = torch.randn(1, 3, 6)
    c0 = torch.randn(1, 3, 6)

    # Run LSTM: returns output, (hn, cn)
    output, (hn, cn) = lstm(input, (h0, c0))

    print(f'output: {output.shape}')
    print(f'hidden: {hn.shape}, cell: {cn.shape}')
    print(f'lstm: {lstm}')


if __name__ == '__main__':
    # dm_rnn_for_base()
    # dm_rnn_for_sequence()
    dm_lstm_base()








