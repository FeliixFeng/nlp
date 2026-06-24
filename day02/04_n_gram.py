"""
n-gram:
    Overview:
        Consecutive n words/chars, use as features to analyze text patterns.
    Types:
        uni-gram (1-gram): each word/char separately
        bi-gram (2-gram): consecutive 2 words
        tri-gram (3-gram): consecutive 3 words
    Purpose:
        Help computer better understand text patterns.
"""

# Todo 1: n value, usually 2 or 3, here use 2 as example
ngram_range = 2


# Todo 2: Function to generate n-gram features
def create_ngram(input_list):
    # 1. Sliding window to get n-gram features
    # i = 0 -> input_list[0:] -> [1, 3, 2, 1, 5, 3]
    # i = 1 -> input_list[1:] -> [3, 2, 1, 5, 3]
    sliced_lists = [input_list[i:] for i in range(ngram_range)]

    # 2. Use zip() to combine sliced lists
    ngram_tuples = zip(*sliced_lists)

    # 3. Convert to set (deduplicate)
    return set(ngram_tuples)


# Todo 3: Test code
if __name__ == '__main__':
    # 1. Define input list
    input_list = [1, 3, 2, 1, 5, 3]

    # 2. Call function to generate n-gram features
    result = create_ngram(input_list)

    # 3. Print result
    print(result)
