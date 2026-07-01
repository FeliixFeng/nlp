"""
RNN Global Name Classifier (RNN全球人名分类)
- Task: Predict which country a name belongs to (18 countries)
- Method: Letter-level one-hot encoding + RNN/LSTM/GRU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Set font for Chinese characters (设置中文字体)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Characters: 26*2 letters + punctuation = 57 (字符表：57个字符)
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 18 countries (18个国家)
categories = [
    'Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French',
    'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean',
    'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese'
]


def read_data(file_path):
    # Read data file: name\tcountry per line (读取数据：每行 名字\t国家)
    my_list_x, my_list_y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) <= 5:
                continue
            x, y = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x, my_list_y


class NameClassDataset(Dataset):
    # Custom Dataset for PyTorch DataLoader (自定义数据集)
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y

    def __len__(self):
        return len(self.my_list_x)

    def __getitem__(self, index):
        index = min(index, len(self.my_list_x) - 1)  # Prevent out of bounds
        x = self.my_list_x[index]
        y = self.my_list_y[index]

        # One-hot encode: "zhang" → (5, 57) tensor
        tensor_x = torch.zeros(len(x), n_letters)
        for li, letter in enumerate(x):
            tensor_x[li][all_letters.find(letter)] = 1

        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)
        return tensor_x, tensor_y


class My_RNN(nn.Module):
    # RNN model: input(57) → RNN(128) → Linear(18) → Softmax
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(My_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)  # Add batch dim: (seq, 57) → (seq, 1, 57)
        output, hn = self.rnn(input, hidden)
        output = self.linear(output[-1])  # Take last time step
        return self.softmax(output), hn

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


class My_LSTM(nn.Module):
    # LSTM model: same as RNN but with cell state (多一个细胞状态)
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(My_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        output, (hn, cn) = self.lstm(input, (hidden, cell))
        output = self.linear(output[-1])
        return self.softmax(output), hn, cn

    def init_hidden(self):
        # Return (hidden, cell) (返回隐藏状态和细胞状态)
        return (torch.zeros(self.n_layers, 1, self.hidden_size),
                torch.zeros(self.n_layers, 1, self.hidden_size))


class My_GRU(nn.Module):
    # GRU model: simplified LSTM, no cell state (简化版LSTM，无细胞状态)
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(My_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        output, hn = self.gru(input, hidden)
        output = self.linear(output[-1])
        return self.softmax(output), hn

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


# Training parameters (训练参数)
mylr = 1e-3
epochs = 1


def train(model, model_name):
    # Read and prepare data (读取并准备数据)
    my_list_x, my_list_y = read_data('./data/name_classification.txt')
    mydataset = NameClassDataset(my_list_x, my_list_y)
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)

    # Loss and optimizer (损失函数和优化器)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=mylr)

    # Metrics tracking (记录指标)
    starttime = time.time()
    total_iter = 0
    total_loss_list = []
    total_acc_list = []
    all_loss = 0.0
    all_acc = 0

    # Training loop (训练循环)
    for epoch in range(epochs):
        pbar = tqdm(mydataloader, desc=f'[{model_name}] Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            # Forward pass (前向传播)
            hidden = model.init_hidden()
            if model_name == 'LSTM':
                hidden, cell = hidden
                output, hidden, cell = model(x[0], hidden, cell)
            else:
                output, hidden = model(x[0], hidden)

            loss = criterion(output, y)

            # Backward pass (反向传播)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics (记录指标)
            total_iter += 1
            all_loss += loss.item()
            all_acc += 1 if output.argmax(1).item() == y.item() else 0
            if total_iter % 100 == 0:
                avg_loss = all_loss / total_iter
                avg_acc = all_acc / total_iter
                total_loss_list.append(avg_loss)
                total_acc_list.append(avg_acc)
                pbar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{avg_acc:.4f}')

        # Save model (保存模型)
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), f'./model/{model_name.lower()}_model.bin')

    total_time = int(time.time() - starttime)
    print(f'{model_name} done! time: {total_time}s, acc: {total_acc_list[-1]:.4f}')
    return total_loss_list, total_time, total_acc_list


def lineToTensor(line):
    # Convert string to one-hot tensor: "zhang" → (5, 57)
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        idx = all_letters.find(letter)
        if idx != -1:
            tensor[li][idx] = 1
    return tensor


def predict(model, model_name, x):
    # Load trained model (加载训练好的模型)
    model.load_state_dict(torch.load(f'./model/{model_name.lower()}_model.bin'))

    # Predict without gradient (预测不需要梯度)
    with torch.no_grad():
        hidden = model.init_hidden()
        if model_name == 'LSTM':
            hidden, cell = hidden
            output, _, _ = model(lineToTensor(x), hidden, cell)
        else:
            output, _ = model(lineToTensor(x), hidden)

    # Get top 3 predictions (取前3个预测结果)
    topv, topi = output.topk(3, 1, True)
    print(f'x===> {x}')
    for i in range(3):
        print(f'\t\t {categories[topi[0][i]]}')


def plot_results(loss_rnn, loss_lstm, loss_gru,
                 acc_rnn, acc_lstm, acc_gru,
                 time_rnn, time_lstm, time_gru):
    # Plot loss comparison (损失对比)
    plt.figure(0)
    plt.plot(loss_rnn, label="RNN")
    plt.plot(loss_lstm, color="red", label="LSTM")
    plt.plot(loss_gru, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.title('Loss Comparison')
    plt.savefig('./img/RNN_LSTM_GRU_loss.png')
    plt.show()

    # Plot accuracy comparison (准确率对比)
    plt.figure(1)
    plt.plot(acc_rnn, label="RNN")
    plt.plot(acc_lstm, color="red", label="LSTM")
    plt.plot(acc_gru, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.title('Accuracy Comparison')
    plt.savefig('./img/RNN_LSTM_GRU_acc.png')
    plt.show()

    # Plot training time (训练时间对比)
    plt.figure(2)
    plt.bar(["RNN", "LSTM", "GRU"], [time_rnn, time_lstm, time_gru])
    plt.title('Training Time')
    plt.savefig('./img/RNN_LSTM_GRU_time.png')
    plt.show()


if __name__ == '__main__':
    os.makedirs('./img', exist_ok=True)

    # Train all models (训练所有模型)
    # loss_rnn, time_rnn, acc_rnn = train(My_RNN(57, 128, 18), 'RNN')
    # loss_lstm, time_lstm, acc_lstm = train(My_LSTM(57, 128, 18), 'LSTM')
    # loss_gru, time_gru, acc_gru = train(My_GRU(57, 128, 18), 'GRU')
    # plot_results(loss_rnn, loss_lstm, loss_gru, acc_rnn, acc_lstm, acc_gru, time_rnn, time_lstm, time_gru)

    # Predict (预测)
    predict(My_RNN(57, 128, 18), 'RNN', 'zhang')
    predict(My_LSTM(57, 128, 18), 'LSTM', 'zhang')
    predict(My_GRU(57, 128, 18), 'GRU', 'zhang')
