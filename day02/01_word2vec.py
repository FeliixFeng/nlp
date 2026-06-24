"""
    Word2Vec with fasttext
"""

import fasttext


def dm01_train_save():
    # Train word2vec model and save to file
    my_model = fasttext.train_unsupervised('./data/wh02aa')

    my_model.save_model('./model/wh02aa.bin')
    print("finish train model")


def dm02_get_word_vector():
    # Get word vector for a word
    my_model = fasttext.load_model('./model/wh02aa.bin')

    results = my_model.get_word_vector("the")
    print(f'type: {type(results)}')
    print(f'shape: {results.shape}')
    print(f'value: {results}')


def dm03_get_similarity():
    # Find similar words
    my_model = fasttext.load_model('./model/wh02aa.bin')
    results = my_model.get_nearest_neighbors("dog")
    print(f'type: {type(results)}')
    print(f'results: {results}')


def dm04_set_hyperparameter():
    # Train model with custom hyperparameters
    my_model = fasttext.train_unsupervised(
        input='./data/wh02aa',
        model='cbow',
        dim=50,
        epoch=1,
        lr=0.01,
        thread=5
    )

    my_model.save_model('./model/wh02aa-set.bin')
 

if __name__ == '__main__':
    # dm01_train_save()
    # dm02_get_word_vector()
    # dm03_get_similarity()
    dm04_set_hyperparameter()
