"""
    Transformers Pipeline Examples

    Usage:
        sentiment-analysis: 文本分类/情感分析
        feature-extraction: 特征抽取
        fill-mask: 完形填空
        question-answering: 问答（需要直接加载模型）
"""

import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np


# todo 1. Sentiment Analysis
def dm01_test_classification():
    """Test sentiment analysis"""
    task = 'sentiment-analysis'
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese_sentiment'
    my_model = pipeline(task, model=model_path)

    output = my_model('我爱北京天安门, 天安门上太阳升!')
    print(f'Sentiment: {output}')


# todo 2. Feature Extraction
def dm02_test_feature_extraction():
    """Test feature extraction"""
    task = 'feature-extraction'
    model_path = '/Users/haifeng/Documents/develop/AI/models/bert-base-chinese'
    my_model = pipeline(task, model=model_path)

    output = my_model('人生改如何起头')
    print(f'Feature shape: {np.array(output).shape}')


# todo 3. Fill Mask
def dm03_test_fill_mask():
    """Test fill mask (cloze task)"""
    task = 'fill-mask'
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese-bert-wwm'
    my_model = pipeline(task, model=model_path)

    output = my_model('我[MASK]你')
    print(f'Fill mask: {output}')


# todo 4. Question Answering (New API)
def dm04_test_question_answering():
    """
    Test question answering
    Note: transformers 5.x removed 'question-answering' task
    Need to load model directly
    """
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese_pretrain_mrc_roberta_wwm'

    # Load model and tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()

    context = '我是王哥, 我是一个工人, 我的喜好是钓鱼和喝茶'
    questions = ['我叫啥?', '我是做什么的?', '我的爱好是什么?']

    # Process each question separately
    for question in questions:
        # Tokenize question and context together
        inputs = tokenizer(
            question,
            context,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Use token_type_ids to find context part (1 = context, 0 = question)
        token_type_ids = inputs['token_type_ids'][0]
        context_mask = (token_type_ids == 1).float()

        # Apply mask to only consider context tokens
        start_logits_masked = start_logits * context_mask + (1 - context_mask) * (-10000)
        end_logits_masked = end_logits * context_mask + (1 - context_mask) * (-10000)

        # Find best start and end positions
        start_idx = torch.argmax(start_logits_masked)
        end_idx = torch.argmax(end_logits_masked) + 1

        # Decode answer
        answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx])

        print(f'Q: {question}')
        print(f'A: {answer}')
        print()


if __name__ == '__main__':
    # dm01_test_classification()
    # dm02_test_feature_extraction()
    # dm03_test_fill_mask()
    dm04_test_question_answering()
