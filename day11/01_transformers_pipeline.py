"""
    Transformers Pipeline Examples

    Usage:
        sentiment-analysis: 文本分类/情感分析
        feature-extraction: 特征抽取
        fill-mask: 完形填空
        question-answering: 问答（需要直接加载模型）
        summarization: 摘要（需要直接加载模型）
"""

import torch
from transformers import (
    pipeline,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
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


# todo 4. Question Answering (Raw Output)
def dm04_test_question_answering():
    """
    Test question answering - print raw output
    """
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese_pretrain_mrc_roberta_wwm'

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()

    context = '我是王哥, 我是一个工人, 我的喜好是钓鱼和喝茶'
    question = '我叫啥?'

    # Tokenize
    inputs = tokenizer(question, context, return_tensors='pt')

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # Print raw output
    print(f'Input: {question}')
    print(f'Context: {context}')
    print(f'Output: {outputs}')
    print(f'Start logits shape: {outputs.start_logits.shape}')
    print(f'End logits shape: {outputs.end_logits.shape}')
    print(f'Start logits: {outputs.start_logits}')
    print(f'End logits: {outputs.end_logits}')


# todo 5. Summarization (Raw Output)
def dm05_test_summarization():
    """
    Test summarization - print raw output
    Note: transformers 5.x removed 'summarization' task
    """
    model_path = '/Users/haifeng/Documents/develop/AI/models/distilbart-cnn-12-6'

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    # Input text
    text = '''China has been growing at a rapid pace over the past decade.
    The country has invested heavily in infrastructure and technology.
    Many people are moving to cities for better opportunities.
    The government has implemented various policies to support economic growth.'''

    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)

    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            min_length=10,
            num_beams=4,
            early_stopping=True
        )

    # Decode
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f'Input: {text[:100]}...')
    print(f'Summary: {summary}')
    print(f'Raw output: {outputs}')


# todo 6. Named Entity Recognition (NER)
def dm06_test_ner():
    """Test named entity recognition"""
    task = 'ner'
    model_path = '/Users/haifeng/Documents/develop/AI/models/roberta-base-finetuned-cluener2020'
    my_model = pipeline(task, model=model_path)

    text = '我在北京清华大学学习计算机'
    output = my_model(text)

    print(f'Input: {text}')
    print(f'Output: {output}')


if __name__ == '__main__':
    # dm01_test_classification()
    # dm02_test_feature_extraction()
    # dm03_test_fill_mask()
    # dm04_test_question_answering()
    # dm05_test_summarization()
    dm06_test_ner()
