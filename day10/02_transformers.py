

import torch
from transformers import pipeline
import numpy as np

def dm01_test_classification():
    task = 'sentiment-analysis'
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese_sentiment'
    my_model = pipeline(task, model=model_path)

    output = my_model('我爱北京天安门, 天安门上太阳升!')

    print(f'output: {output}')


# 特征抽取任务
def dm02_test_feature_extraction():
    task = 'feature-extraction'
    model_path = '/Users/haifeng/Documents/develop/AI/models/bert-base-chinese'
    my_model = pipeline(task, model=model_path)

    output = my_model('人生改如何起头')

    # print(f'output: {output}')
    print(f'shape: {type(output)}')
    print(f'shpae:{np.array(output).shape}')


def dm03_test_fill_mask():
    task = 'fill-mask'
    model_path = '/Users/haifeng/Documents/develop/AI/models/chinese-bert-wwm'
    my_model = pipeline(task, model=model_path)

    output = my_model('我[MASK]你')

    print(f'output: {output}')
    print(f'shape: {type(output)}')


if __name__ == '__main__':
    # dm01_test_classification()
    # dm02_test_feature_extraction()
    dm03_test_fill_mask()