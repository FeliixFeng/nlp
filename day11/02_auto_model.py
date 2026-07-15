"""
    Transformers AutoModel Examples

    Direct model loading without pipeline
    More control over model internals
"""

import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification
)


# todo 1. Text Classification (AutoModel)
def dm01_text_classification():
    """Test text classification using AutoModel"""
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese_sentiment'

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Input text
    text = '人生该如何起头'

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # Print raw output
    print(f'Input: {text}')
    print(f'Output logits: {outputs.logits}')
    print(f'Prediction: {torch.argmax(outputs.logits, dim=-1).item()}')



# todo 2. Feature Extraction (AutoModel)
def dm02_feature_extraction():
    """Test feature extraction using AutoModel"""
    model_path = '/Users/haifeng/Documents/develop/AI/models/bert-base-chinese'

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # Input text
    text = '人生该如何起头'

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)

    # Print raw output
    print(f'Input: {text}')
    print(f'Last hidden state shape: {outputs.last_hidden_state.shape}')
    print(f'Pooler output shape: {outputs.pooler_output.shape}')
    print(f'Features (first 10): {outputs.pooler_output[0][:10]}')


# todo 3. Fill Mask (完型填空)
def dm03_fill_mask():
    """
    Test fill mask (cloze task) using AutoModel
    完型填空任务：预测被遮蔽的词
    """
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese-bert-wwm'

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. Load model
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.eval()

    # 3. Tokenize input
    text = '我想明天去[MASK]家吃饭.'
    inputs = tokenizer(text, return_tensors='pt')
    print(f'Input: {text}')
    print(f'Inputs: {inputs}')

    # 4. Predict
    with torch.no_grad():
        outputs = model(**inputs)

    print(f'Output logits shape: {outputs.logits.shape}')  # [1, 12, 21128]

    # 5. Get prediction for [MASK] position (index 6)
    mask_pred_idx = torch.argmax(outputs.logits[0][6]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([mask_pred_idx])

    print(f'Predicted token: {predicted_token}')


# todo 4. Question Answering (阅读理解/抽取式问答)
def dm04_question_answering():
    """
    Test question answering using AutoModel
    阅读理解任务(抽取式问答)：给定文本和问题，从文本中抽取答案
    """
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese_pretrain_mrc_roberta_wwm'

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. Load model
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()

    # 3. Input text
    context = '我叫张三 我是一个程序员 我的喜好是打篮球'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']

    # 4. Feed data to model
    for question in questions:
        inputs = tokenizer(question, context, return_tensors='pt')
        print(f'inputs---> {inputs}')

        with torch.no_grad():
            output = model(**inputs)

        print(f'output---> {output}')

        # Get answer span from start and end logits
        start = torch.argmax(output.start_logits)
        end = torch.argmax(output.end_logits) + 1

        # Convert token ids back to text
        answer = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start:end])
        print(f'question: {question}, answer: {answer}')
        print()


# todo 5. Summarization (文本摘要)
def dm05_summarization():
    """
    Test text summarization using AutoModel
    文本摘要任务：输入一段长文本，输出简短的摘要
    """
    model_path = '/Users/haifeng/Documents/develop/AI/models/distilbart-cnn-12-6'

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    # 3. Input text
    text = (
        "BERT is a transformers model pretrained on a large corpus of English data "
        "in a self-supervised fashion. This means it was pretrained on the raw texts "
        "only, with no humans labelling them in any way (which is why it can use lots "
        "of publicly available data) with an automatic process to generate inputs and "
        "labels from those texts. More precisely, it was pretrained with two objectives: "
        "Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the "
        "words in the input then run the entire masked sentence through the model and has "
        "to predict the masked words. This is different from traditional recurrent neural "
        "networks (RNNs) that usually see the words one after the other, or from autoregressive "
        "models like GPT which internally mask the future tokens. It allows the model to learn "
        "a bidirectional representation of the sentence."
    )

    # 4. Generate summary
    inputs = tokenizer([text], return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], max_length=50, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f'Original text length: {len(text.split())} words')
    print(f'Summary: {summary}')


# todo 6. NER (命名实体识别)
def dm06_ner():
    """
    Test Named Entity Recognition using AutoModel
    NER任务：识别文本中的人名(PER)、地名(LOC)、组织(ORG)等实体
    标签格式：B-PER(人名开始), I-PER(人名内部), B-LOC(地名开始), I-LOC(地名内部), O(非实体)
    """
    model_path = '/Users/haifeng/Documents/develop/AI/models/roberta-base-finetuned-cluener2020'

    # 1. Load tokenizer, model and config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model.eval()

    # 2. Tokenize input
    text = '我爱北京天安门，天安门上太阳升'
    inputs = tokenizer(text, return_tensors='pt')
    print(f'input_ids shape: {inputs.input_ids.shape}')

    # 3. Predict
    with torch.no_grad():
        logits = model(inputs.input_ids).logits
    print(f'logits shape: {logits.shape}')  # [1, seq_len, num_labels]

    # 4. Get predictions and display
    input_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    predictions = torch.argmax(logits, dim=-1)[0]

    print(f'\nText: {text}')
    print(f'{"Token":<10} {"Label"}')
    print('-' * 25)
    for token, pred_id in zip(input_tokens, predictions):
        label = config.id2label[pred_id.item()]
        if label != 'O':  # Only show entities
            print(f'{token:<10} {label}')


# todo 7. Fill Mask (具体模型加载方式)
def dm07_bert_fill_mask():
    """
    Fill mask using specific model classes (BertTokenizer, BertForMaskedLM)
    对比 Auto 方式，这种是直接指定具体模型类
    """
    from transformers import BertTokenizer, BertForMaskedLM

    # 1. Load tokenizer
    model_path = '/Users/haifeng/Documents/develop/AI/models/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 2. Load model
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()

    # 3. Tokenize
    text = '我想明天去[MASK]家吃饭'
    inputs = tokenizer(text, return_tensors='pt')
    print(f'inputs---> {inputs}')

    # 4. Predict
    with torch.no_grad():
        output = model(**inputs)

    print(f'output---> {output}')
    print(f'output.logits shape: {output.logits.shape}')  # [1, 11, 21128]

    # 5. Get prediction
    mask_pred_idx = torch.argmax(output.logits[0][6]).item()
    print(f'Predicted token: {tokenizer.convert_ids_to_tokens([mask_pred_idx])}')


if __name__ == '__main__':
    # dm01_text_classification()
    # dm02_feature_extraction()
    # dm03_fill_mask()
    # dm04_question_answering()
    # dm05_summarization()
    # dm06_ner()
    dm07_bert_fill_mask()
