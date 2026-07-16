import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import time
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from rich import print


device = torch.device('mps')
MODEL_PATH = '/Users/haifeng/Documents/develop/AI/models/bert-base-chinese'


def load_data():
    """Load train/valid/test datasets from CSV files."""
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    valid_dataset = load_dataset('csv', data_files='./data/validation.csv', split='train')
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')

    print(f'train: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}')
    print(f'sample: {train_dataset[:3]}')

    return train_dataset, valid_dataset, test_dataset


class CollateFn:
    """Picklable collate function for DataLoader with num_workers > 0."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        sentences = [item['text'] for item in data]
        labels = [item['label'] for item in data]

        inputs = self.tokenizer(
            sentences,
            truncation=True,
            max_length=300,
            padding='max_length',
            return_tensors='pt',
        )

        return inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], torch.LongTensor(labels)


# Custom classifier: BERT (frozen) + linear head
class TextClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=2):
        super().__init__()
        self.bert = pretrained_model
        self.linear = nn.Linear(768, num_classes)

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        return self.linear(bert_output.pooler_output)


def train():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    bert = BertModel.from_pretrained(MODEL_PATH).to(device)

    model = TextClassifier(bert).to(device)

    train_dataset, valid_dataset, _ = load_data()
    collate_fn = CollateFn(tokenizer)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.linear.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        start_time = time.time()

        for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(train_loader), start=1):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            output = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                preds = torch.argmax(output, dim=-1)
                acc = (preds == labels).float().mean().item()
                elapsed = time.time() - start_time
                print(f'epoch {epoch+1} | step {i} | loss {loss:.2f} | acc {acc:.2f} | {elapsed:.1f}s')

        torch.save(model.state_dict(), f'./model/classification_{epoch+1}.pth')


def evaluate():
    # Load test set
    print('\n加载测试集')
    _, _, test_dataset = load_data()

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    bert = BertModel.from_pretrained(MODEL_PATH).to(device)

    model = TextClassifier(bert).to(device)
    model.load_state_dict(torch.load('./model/classification_1.pth', map_location=device))
    model.eval()

    collate_fn = CollateFn(tokenizer)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    correct = 0
    total = 0

    for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(test_loader, total=20)):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(input_ids, token_type_ids, attention_mask)

        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        if i % 5 == 0:
            print(f'acc: {correct / total:.4f}', end=' ')
            print(tokenizer.decode(input_ids[0], skip_special_tokens=True), end=' ')
            print(f'预测值: {preds[0].item()}, 真实值: {labels[0].item()}')

        if i >= 20:
            break

    print(f'\n准确率(前20 batch): {correct / total:.4f}')


if __name__ == '__main__':
    # train()
    evaluate()
