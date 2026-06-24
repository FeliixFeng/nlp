



"""
    Text Analysis with Seaborn & WordCloud
    - Load dataset and visualize
    - Label distribution
    - Sentence length distribution
"""

import jieba
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jieba.posseg as pseg
from itertools import chain
from wordcloud import WordCloud

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def dm01_label_sns_countplot():
    # Plot label distribution (train & dev)
    plt.style.use('fivethirtyeight')

    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    sns.countplot(x='label', data=train_data, hue='label', legend=False)
    plt.title('train_label')
    plt.tight_layout()
    plt.show()

    sns.countplot(x='label', data=dev_data, hue='label', legend=False)
    plt.title('dev_label')
    plt.tight_layout()
    plt.show()


def dm02_len_sns_countplot_distplot():
    # Plot sentence length distribution
    plt.style.use('fivethirtyeight')

    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # Add sentence length column
    train_data['sentence_length'] = train_data['sentence'].apply(len)

    # Count plot: each length as a bar
    sns.countplot(x='sentence_length', data=train_data)
    plt.title('sentence_length')
    plt.xticks(rotation=90)
    plt.show()

    # Histogram with KDE curve
    sns.histplot(x='sentence_length', data=train_data, kde=True)
    plt.title('sentence_length')
    plt.tight_layout()
    plt.show()

    # Same for dev set
    dev_data['sentence_length'] = dev_data['sentence'].apply(len)
    sns.countplot(x='sentence_length', data=dev_data)
    plt.title('sentence_length')
    plt.xticks(rotation=90)
    plt.show()

    sns.histplot(x='sentence_length', data=train_data, kde=True)
    plt.title('sentence_length')
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    # dm01_label_sns_countplot()
    dm02_len_sns_countplot_distplot()















