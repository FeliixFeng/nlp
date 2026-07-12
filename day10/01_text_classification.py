"""
    FastText Text Classification

    FastText: Facebook's text classification tool
    - Fast training
    - Supports OOV (out-of-vocabulary) words
    - Good for text classification tasks
"""

import fasttext
import os

# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# todo 1. Basic training
def dm01_basic():
    """Train a simple text classification model"""
    print("=" * 50)
    print("dm01: Basic Training")
    print("=" * 50)

    train_path = os.path.join(BASE_DIR, 'data', 'cooking_train.txt')
    model = fasttext.train_supervised(train_path)

    result1 = model.predict('How is mass egg-frying performed?')
    print(f'Predict 1: {result1}')

    result2 = model.predict('Using liquid nitrogen for tenderizing octopus?')
    print(f'Predict 2: {result2}')


# todo 2. Use preprocessed data
def dm02_preprocessed():
    """Train with preprocessed data"""
    print("=" * 50)
    print("dm02: Preprocessed Data")
    print("=" * 50)

    train_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.train')
    valid_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.valid')

    model = fasttext.train_supervised(train_path)

    result1 = model.predict('how is mass egg-frying performed?')
    print(f'Predict 1: {result1}')

    result2 = model.predict('using liquid nitrogen for tenderizing octopus?')
    print(f'Predict 2: {result2}')

    # Test on validation set
    result3 = model.test(valid_path)
    print(f'Validation: {result3}')


# todo 3. Increase epochs
def dm03_epoch():
    """Train with more epochs"""
    print("=" * 50)
    print("dm03: More Epochs")
    print("=" * 50)

    train_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.train')
    valid_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.valid')

    # Increase epochs to 25
    model = fasttext.train_supervised(train_path, epoch=25)

    result1 = model.predict('how is mass egg-frying performed?')
    print(f'Predict 1: {result1}')

    result2 = model.predict('using liquid nitrogen for tenderizing octopus?')
    print(f'Predict 2: {result2}')

    # Test on validation set
    result3 = model.test(valid_path)
    print(f'Validation: {result3}')


# todo 4. Adjust learning rate
def dm04_learning_rate():
    """Train with different learning rates"""
    print("=" * 50)
    print("dm04: Learning Rate Tuning")
    print("=" * 50)

    train_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.train')
    valid_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.valid')

    # Try different learning rates
    for lr in [0.1, 0.5, 1.0]:
        model = fasttext.train_supervised(train_path, epoch=25, lr=lr)
        result = model.test(valid_path)
        print(f'lr={lr}: precision={result[1]:.4f}, recall={result[2]:.4f}')


# todo 5. Adjust word n-grams
def dm05_ngrams():
    """Train with different n-gram settings"""
    print("=" * 50)
    print("dm05: N-grams Tuning")
    print("=" * 50)

    train_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.train')
    valid_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.valid')

    # Try different n-gram settings
    for word_ngrams in [1, 2, 3]:
        model = fasttext.train_supervised(train_path, epoch=25, lr=0.5, wordNgrams=word_ngrams)
        result = model.test(valid_path)
        print(f'wordNgrams={word_ngrams}: precision={result[1]:.4f}, recall={result[2]:.4f}')


# todo 6. Save and load model
def dm06_save_load():
    """Save and load model"""
    print("=" * 50)
    print("dm06: Save and Load Model")
    print("=" * 50)

    train_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.train')

    # Train model
    model = fasttext.train_supervised(train_path, epoch=25, lr=0.5, wordNgrams=2)

    # Save model
    model_path = os.path.join(BASE_DIR, 'data', 'my_model.bin')
    model.save_model(model_path)
    print(f'Model saved to: {model_path}')

    # Load model
    loaded_model = fasttext.load_model(model_path)
    result = loaded_model.predict('how to make pasta?')
    print(f'Loaded model predict: {result}')


# todo 7. Get word vectors
def dm07_word_vectors():
    """Get word vectors from model"""
    print("=" * 50)
    print("dm07: Word Vectors")
    print("=" * 50)

    train_path = os.path.join(BASE_DIR, 'data', 'cooking.pre.train')

    model = fasttext.train_supervised(train_path, epoch=25, lr=0.5)

    # Get word vector
    word_vec = model.get_word_vector('pasta')
    print(f'Word "pasta" vector shape: {word_vec.shape}')
    print(f'First 10 values: {word_vec[:10]}')

    # Get sentence vector
    sentence_vec = model.get_sentence_vector('how to make pasta')
    print(f'Sentence vector shape: {sentence_vec.shape}')


if __name__ == '__main__':
    # dm01_basic()
    # dm02_preprocessed()
    # dm03_epoch()
    # dm04_learning_rate()
    # dm05_ngrams()
    # dm06_save_load()
    dm07_word_vectors()
