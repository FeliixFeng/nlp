"""
    RNN Global Name Classifier (RNN全球人名分类)
    - Predict which country a name belongs to (18 countries)
    - Use letter-level one-hot encoding
    - Compare RNN, LSTM, GRU models
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

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# All possible characters: 26*2 letters + punctuation
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 18 countries in dataset
categories = [
    'Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French',
    'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean',
    'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese'
]
category_num = len(categories)


def read_data(file_path):
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
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        self.sample_len = len(my_list_x)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        index = min(index, self.sample_len - 1)
        x = self.my_list_x[index]
        y = self.my_list_y[index]

        # One-hot encode
        tensor_x = torch.zeros(len(x), n_letters)
        for li, letter in enumerate(x):
            letter_index = all_letters.find(letter)
            tensor_x[li][letter_index] = 1

        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)
        return tensor_x, tensor_y


class My_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(My_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        output, hn = self.rnn(input, hidden)
        tmp_output = output[-1]
        tmp_output = self.linear(tmp_output)
        return self.softmax(tmp_output), hn

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


class My_LSTM(nn.Module):
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
        tmp_output = output[-1]
        tmp_output = self.linear(tmp_output)
        return self.softmax(tmp_output), hn, cn

    def init_hidden(self):
        h0 = torch.zeros(self.n_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.n_layers, 1, self.hidden_size)
        return h0, c0


class My_GRU(nn.Module):
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
        tmp_output = output[-1]
        tmp_output = self.linear(tmp_output)
        return self.softmax(tmp_output), hn

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


# Training parameters
mylr = 1e-3
epochs = 1


def train(model, model_name):
    # Read data
    my_list_x, my_list_y = read_data('./data/name_classification.txt')
    print(f'{model_name} - data count: {len(my_list_x)}')

    # Create dataset and dataloader
    my_nameclassdataset = NameClassDataset(my_list_x, my_list_y)
    mydataloader = DataLoader(dataset=my_nameclassdataset, batch_size=1, shuffle=True)

    # Loss function and optimizer
    mycrossentropyloss = nn.NLLLoss()
    myadam = optim.Adam(model.parameters(), lr=mylr)

    # Training metrics
    starttime = time.time()
    total_iter_num = 0
    total_loss_list = []
    total_acc_list = []
    all_loss = 0.0
    all_acc = 0

    # Training loop
    for epoch_idx in range(epochs):
        pbar = tqdm(mydataloader, desc=f'[{model_name}] Epoch {epoch_idx+1}/{epochs}')
        for i, (x, y) in enumerate(pbar):
            # Initialize hidden state (LSTM needs cell state too)
            hidden = model.init_hidden()
            if model_name == 'LSTM':
                hidden, cell = hidden
                output_y, hidden, cell = model(x[0], hidden, cell)
            else:
                output_y, hidden = model(x[0], hidden)

            myloss = mycrossentropyloss(output_y, y)

            myadam.zero_grad()
            myloss.backward()
            myadam.step()

            total_iter_num += 1
            pred = output_y.argmax(dim=1)

            # Record every 100 samples (累计平均)
            all_loss += myloss.item()
            all_acc += 1 if pred.item() == y.item() else 0
            if total_iter_num % 100 == 0:
                avg_loss = all_loss / total_iter_num
                avg_acc = all_acc / total_iter_num
                total_loss_list.append(avg_loss)
                total_acc_list.append(avg_acc)
                pbar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{avg_acc:.4f}')

        # Save model every epoch
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), f'./model/{model_name.lower()}_model.bin')

    endtime = time.time()
    total_time = int(endtime - starttime)
    print(f'{model_name} training done! time: {total_time}s, final acc: {total_acc_list[-1]:.4f}')

    return total_loss_list, total_time, total_acc_list


def lineToTensor(line):
    # Convert name string to one-hot tensor
    # (名字字符串转one-hot张量)
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        if letter_index != -1:
            tensor[li][letter_index] = 1
    return tensor


def predict(model, model_name, x):
    # Load trained model
    model_path = f'./model/{model_name.lower()}_model.bin'
    model.load_state_dict(torch.load(model_path))

    # Convert input to tensor
    tensor_x = lineToTensor(x)

    # Predict (no gradient needed)
    with torch.no_grad():
        hidden = model.init_hidden()
        if model_name == 'LSTM':
            hidden, cell = hidden
            output, hidden, cell = model(tensor_x, hidden, cell)
        else:
            output, hidden = model(tensor_x, hidden)

    # Get top 3 predictions
    topv, topi = output.topk(3, 1, True)

    print(f'x===> {x}')
    for i in range(3):
        category_idx = topi[0][i]
        category = categories[category_idx]
        print(f'\t\t {category}')


def dm08_predict_rnn():
    predict(My_RNN(57, 128, 18), 'RNN', 'zhang')


def dm09_predict_lstm():
    predict(My_LSTM(57, 128, 18), 'LSTM', 'zhang')


def dm10_predict_gru():
    predict(My_GRU(57, 128, 18), 'GRU', 'zhang')


def plot_results(loss_list_rnn, loss_list_lstm, loss_list_gru,
                 acc_list_rnn, acc_list_lstm, acc_list_gru,
                 time_rnn, time_lstm, time_gru):
    # Plot loss comparison
    plt.figure(0)
    plt.plot(loss_list_rnn, label="RNN")
    plt.plot(loss_list_lstm, color="red", label="LSTM")
    plt.plot(loss_list_gru, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.title('Loss Comparison')
    plt.xlabel('iterations (x100)')
    plt.ylabel('loss')
    plt.savefig('./img/RNN_LSTM_GRU_loss.png')
    plt.show()

    # Plot accuracy comparison
    plt.figure(1)
    plt.plot(acc_list_rnn, label="RNN")
    plt.plot(acc_list_lstm, color="red", label="LSTM")
    plt.plot(acc_list_gru, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.title('Accuracy Comparison')
    plt.xlabel('iterations (x100)')
    plt.ylabel('accuracy')
    plt.savefig('./img/RNN_LSTM_GRU_acc.png')
    plt.show()

    # Plot training time comparison
    plt.figure(2)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [time_rnn, time_lstm, time_gru]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.title('Training Time Comparison')
    plt.ylabel('seconds')
    plt.savefig('./img/RNN_LSTM_GRU_time.png')
    plt.show()


if __name__ == '__main__':
    os.makedirs('./img', exist_ok=True)

    # Train (训练)
    # print("=" * 50)
    # print("Training RNN...")
    # loss_rnn, time_rnn, acc_rnn = train(My_RNN(57, 128, 18), 'RNN')

    # print("=" * 50)
    # print("Training LSTM...")
    # loss_lstm, time_lstm, acc_lstm = train(My_LSTM(57, 128, 18), 'LSTM')

    # print("=" * 50)
    # print("Training GRU...")
    # loss_gru, time_gru, acc_gru = train(My_GRU(57, 128, 18), 'GRU')

    # print("=" * 50)
    # print("Plotting results...")
    # plot_results(loss_rnn, loss_lstm, loss_gru,
    #              acc_rnn, acc_lstm, acc_gru,
    #              time_rnn, time_lstm, time_gru)

    # Predict (预测)
    dm08_predict_rnn()
    dm09_predict_lstm()
    dm10_predict_gru()
