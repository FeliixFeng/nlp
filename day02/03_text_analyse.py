



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


def dm03_sns_stripplot():
    # Scatter plot of sentence length by label (positive/negative)
    plt.style.use('fivethirtyeight')

    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # Add sentence length column
    train_data['sentence_length'] = train_data['sentence'].apply(len)

    # Strip plot: y=length, x=label
    sns.stripplot(y='sentence_length', x='label', data=train_data, hue='label')
    plt.show()

    # Same for dev set
    dev_data['sentence_length'] = dev_data['sentence'].apply(len)
    sns.stripplot(y='sentence_length', x='label', data=dev_data)
    plt.show()


def dm04_cal_wordcount():
    # Count unique words in train & dev sets
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # Seg every sentence, flatten to one list, then use set to get unique words
    # chain(*map(...)) unpacks map object and chains all lists into one iterator
    train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))
    print(f"train vocab count: {len(train_vocab)}")

    dev_vocab = set(chain(*map(lambda x: jieba.lcut(x), dev_data['sentence'])))
    print(f"dev vocab count: {len(dev_vocab)}")


def get_a_list(text):
    # Extract adjectives (POS tag 'a') from text
    r = []
    for g in pseg.lcut(text):
        if g.flag == 'a':  # 'a' = adjective
            r.append(g.word)
    return r


def get_word_cloud(keywords_list):
    # Generate word cloud from keywords list
    wordcloud = WordCloud(font_path="./SimHei.ttf", max_words=100, background_color='white')
    keywords_string = " ".join(keywords_list)
    wordcloud.generate(keywords_string)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # dm01_label_sns_countplot()
    # dm02_len_sns_countplot_distplot()
    # dm03_sns_stripplot()
    dm04_cal_wordcount()















