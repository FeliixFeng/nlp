import torch
from transformers import BertTokenizer, BertForMaskedLM
from rich import print
import random


device = torch.device('mps')
MODEL_PATH = '/Users/haifeng/Documents/develop/AI/models/bert-base-chinese'


def load_test_sentences():
    """加载测试句子，使用现有的训练数据"""
    sentences = []
    with open('./data/train.csv', 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            # 处理CSV格式：跳过标签，只取文本
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                sentences.append(parts[1])
    return sentences


def mask_random_word(sentence, tokenizer):
    """随机mask句子中的一个词"""
    # 分词
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) < 2:
        return None, None, None

    # 随机选择一个位置（排除[CLS]和[SEP]）
    mask_idx = random.randint(0, len(tokens) - 1)
    original_token = tokens[mask_idx]

    # 替换为[MASK]
    tokens[mask_idx] = '[MASK]'

    # 转换为ID
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]

    # 记录MASK的位置（+1是因为加了[CLS]）
    mask_pos = mask_idx + 1

    return input_ids, mask_pos, original_token


def predict_masked_word(model, tokenizer, input_ids, mask_pos, top_k=5):
    """预测被MASK的词"""
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs.logits

    # 获取MASK位置的预测
    mask_predictions = predictions[0, mask_pos]
    top_k_ids = torch.topk(mask_predictions, top_k).indices.tolist()
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_ids]
    top_k_probs = torch.softmax(mask_predictions, dim=0)[top_k_ids].tolist()

    return list(zip(top_k_tokens, top_k_probs))


def main():
    # 加载模型和分词器
    print("加载BERT模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForMaskedLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # 加载测试句子
    print("加载测试数据...")
    sentences = load_test_sentences()
    print(f"共加载 {len(sentences)} 个句子")

    # 进行填空测试
    print("\n" + "="*60)
    print("中文填空测试")
    print("="*60)

    correct = 0
    total = 0

    # 测试10个句子
    for i in range(10):
        sentence = random.choice(sentences)
        input_ids, mask_pos, original_token = mask_random_word(sentence, tokenizer)

        if input_ids is None:
            continue

        # 预测
        predictions = predict_masked_word(model, tokenizer, input_ids, mask_pos)

        # 显示结果
        masked_sentence = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"\n句子 {i+1}: {masked_sentence}")
        print(f"原始词: {original_token}")
        print(f"预测结果:")
        for token, prob in predictions:
            print(f"  {token}: {prob:.4f}")

        # 检查是否预测正确
        if predictions[0][0] == original_token:
            correct += 1
        total += 1

    print("\n" + "="*60)
    print(f"准确率: {correct}/{total} = {correct/total:.2%}")
    print("="*60)


if __name__ == '__main__':
    main()
