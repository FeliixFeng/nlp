"""
    Jieba Chinese Word Segmentation
    - cut(): generator mode
    - lcut(): list mode
    - cut_all: True=all words, False=precise mode
"""

import jieba


def dm01():
    """
    Basic jieba usage:
    1. cut() returns generator
    2. lcut() returns list directly
    3. cut_all=True: extract all possible words
    4. cut_all=False: precise segmentation (default)
    """
    content = '软件工程专业的学长在武汉纺织大学的实验室里，吹着二十四度的空调除湿，手里熟练地用MacBook的M3芯片跑着uv环境。'

    # cut_all=True: all possible words
    result1 = jieba.cut(content, cut_all=True)
    list1 = list(result1)
    print(f'cut_all=True: {list1}')

    # cut_all=False: precise mode (default)
    list2 = jieba.lcut(content, cut_all=False)
    print(f'cut_all=False: {list2}')


def dm02():
    """
    Search engine mode
    - cut_for_search(): more granular segmentation
    """
    content = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下访问了缅甸仰光'

    # Search engine mode: split long words
    result = jieba.cut_for_search(content)
    print(f'search mode: {list(result)}')


def dm03():
    """
    Custom dictionary:
    1. add_word(): add single word
    2. load_userdict(): load from file
    """
    # 1. Add single word
    jieba.add_word('武汉纺织大学', freq=10000)
    jieba.add_word('计算机学院', freq=10000)
    text = '武汉纺织大学的计算机学院很厉害'
    print(f'add_word: {jieba.lcut(text)}')

    # 2. Load from file
    jieba.load_userdict('dict.txt')
    text2 = '我在武汉纺织大学学习pytorch和transformer'
    print(f'load_userdict: {jieba.lcut(text2)}')


if __name__ == '__main__':
    dm01()
    print()
    dm02()
    print()
    dm03()
