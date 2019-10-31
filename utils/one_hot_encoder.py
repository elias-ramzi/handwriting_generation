from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, allow_multiple=True):
        self.allow_multiple = allow_multiple

    def fit(self, X, y=None):
        """
        X is a list of strings
        """
        vocab = set("".join(X))
        self.vocab = dict(zip(vocab, range(len(vocab))))
        self.inv_vocab = dict(zip(range(len(vocab)), vocab))
        return self

    def transform(self, X):
        """
        X is a list of strings
        """
        encoded = np.zeros((len(X), len(self.vocab)))
        for i, text in enumerate(X):
            if self.allow_multiple:
                voc = Counter(text)
                for letter, n in voc.items():
                    encoded[i][self.vocab[letter]] = n
            else:
                voc = set(text)
                for letter in voc:
                    encoded[i][self.vocab[letter]] = 1

        return encoded

    def inverse_transform(self, s):
        """
        It does not really computes the inverse transform
        It makes it easier to debug
        """
        sentence = {}
        for i, count in enumerate(s):
            # count == 1 if not self.allow_multiple
            if count > 0:
                sentence[self.inv_vocab[int(i)]] = count
        return sentence


if __name__ == '__main__':
    texts = [
        'welcome',
        'elwcome',
        'elias',
        'test',
    ]

    enc = OneHotEncoder()
    out = enc.fit_transform(texts)
    print(out)
