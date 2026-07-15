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


if __name__ == '__main__':
    dm01_text_classification()
