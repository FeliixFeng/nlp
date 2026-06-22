"""
    POS Tagging (Part of Speech)
    - 词性标注：给每个词标注词性
    - jieba.posseg: pos = part of speech

    Common POS Tags:
    n  - 名词 (noun)
    v  - 动词 (verb)
    a  - 形容词 (adjective)
    r  - 代词 (pronoun)
    ns - 地名 (place name)
    nt - 机构名 (organization)
"""

import jieba
import jieba.posseg as pseg


def dm01():
    """
    Basic POS tagging:
    - pseg.lcut(): returns list of (word, tag) pairs
    """
    content = '我爱北京天安门'

    # POS tagging
    result = pseg.lcut(content)
    print(f'result: {result}')

    # Print word and its POS tag
    print('\nWord -> POS Tag:')
    for word, tag in result:
        print(f'{word} -> {tag}')


if __name__ == '__main__':
    dm01()