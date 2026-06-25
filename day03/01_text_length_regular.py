"""
    Text Length Regularization
    - Pad short sequences with 0
    - Truncate long sequences
"""

from tensorflow.keras.preprocessing import sequence

# Fixed length for all sequences
cutlen = 10


def padding(x_train):
    # Use Keras pad_sequences: truncate from front, pad at back
    return sequence.pad_sequences(x_train, maxlen=cutlen, truncating='pre', padding='post')


def padding_custom(x_train):
    # Custom padding implementation
    list1 = []
    for sentence in x_train:
        if len(sentence) > cutlen:
            # Truncate: keep first cutlen words
            list1.append(sentence[:cutlen])
        else:
            # Pad: append 0s to reach cutlen
            padding_len = cutlen - len(sentence)
            list1.append(sentence + [0] * padding_len)

    return list1


if __name__ == '__main__':
    # Test data: two sentences with different lengths
    x_train = [
        [1, 23, 5, 32, 55, 23, 45, 12, 45, 88, 23, 44, 11],
        [12, 21, 23, 22, 3]
    ]

    # result = padding(x_train)
    result = padding_custom(x_train)
    print(result)