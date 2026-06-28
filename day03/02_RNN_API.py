"""
    Case: Demo traditional RNN implementation

    RNN Introduction:
        Overview:
            Recurrent Neural Network, mainly handles sequential data.
            Sequential data: later data depends on previous data.

        Classification by Input & Output:
            N vs N: input n, output n - poetry, couplets...
            N vs 1: sentiment analysis, text classification, intent recognition...
            1 vs N: generation tasks, image captioning...
            N vs M: Seq2Seq, machine translation...  # Most used

        Classification by Internal Structure:
            Traditional RNN:
                Input layer, Hidden layer(Embedding, RNN layer), Output layer
            LSTM:
                Forget gate, Input gate, Cell state, Output gate
            Bi-LSTM:
                Forward LSTM + Backward LSTM, then concatenate
            GRU:
                Reset gate, Update gate
            Bi-GRU:
                Forward GRU + Backward GRU, then concatenate
"""