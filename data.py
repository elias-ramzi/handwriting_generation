import re
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from utils import OneHotEncoder


class Data(ABC):

    def __init__(
        self,
        path_to_data='data/strokes-py3.npy',
        path_to_sentences='data/sentences.txt',
        clean_text=True,
        train_split=0.8,
    ):
        self.path_to_data = path_to_data
        self.path_to_sentences = path_to_sentences
        self.clean_text = clean_text
        self.encoder = OneHotEncoder()
        self.train_split = train_split

    @property
    def strokes(self):
        if not hasattr(self, '_strokes'):
            strokes = np.load(self.path_to_data, allow_pickle=True)
            self.num_val = int(len(strokes) * self.train_split)
            self._strokes = strokes[:self.num_val].copy()
            self._validation = strokes[self.num_val:].copy()
            strokes = np.vstack(self._strokes.copy())
            strokes = strokes[:, 1:]
            self.mean1, self.mean2 = np.mean(strokes, axis=0)
            self.std1, self.std2 = np.std(strokes, axis=0)
        return self._strokes.copy()

    @property
    def validation_strokes(self):
        if not hasattr(self, '_validation'):
            _ = self.strokes
        return self._strokes.copy()

    def prepare_text(self, text):
        _ = self.sentences
        text = re.sub('[^.,a-zA-Z!?\-\'" \n]', '#', text)
        text = text.split('\n')
        text = self.encoder.transform(text)[0]
        text = np.vstack((
            text,
            self.char_padding*(self.char_length-text.shape[0])
        ))
        return tf.dtypes.cast(text.reshape((1,) + text.shape), float)

    @property
    def sentences(self):
        if not hasattr(self, '_sentences'):
            with open(self.path_to_sentences) as f:
                texts = f.read()

            if self.clean_text:
                texts = re.sub('[^.,a-zA-Z!?\-\'" \n]', '#', texts)

            texts = texts.split('\n')
            self.num_val = int(len(texts) * self.train_split)
            sentence = texts[:self.num_val]
            validation_sentence = texts[:self.num_val]
            self._sentences = self.encoder.fit_transform(sentence)
            self._validation_sentences = self.encoder.transform(validation_sentence)
        return self._sentences.copy()

    @property
    def validation_sentences(self):
        if not hasattr(self, '_validation_sentences'):
            _ = self.sentences
        return self._validation_sentences.copy()

    def scale_back(self, strokes):
        if hasattr(strokes, "numpy"):
            strokes = strokes.numpy()
        if not hasattr(self, 'std1'):
            _ = self.strokes
        strokes[:, 1] = strokes[:, 1] * self.std1 + self.mean1
        strokes[:, 2] = strokes[:, 2] * self.std2 + self.mean2
        return strokes

    @abstractmethod
    def batch_generator(self, sequence_lenght, batch_size=10):
        raise NotImplementedError


class DataPrediction(Data):

    def __init__(
        self,
        path_to_data='data/strokes-py3.npy',
        train_split=0.8,
        scale=True,
    ):
        super(DataPrediction, self).__init__(
            path_to_data=path_to_data,
            train_split=train_split
        )
        self.scale = scale

    def batch_generator(self, shuffle=False, data_type='train'):
        assert data_type in ['train', 'validation']
        # We want (x3, x1, x2) --> (x1, x2, x3)
        if data_type == 'train':
            all_strokes = self.strokes
        elif data_type == 'validation':
            all_strokes = self.validation_strokes
        while True:
            if shuffle:
                np.random.shuffle(all_strokes)
            for strokes in all_strokes:
                strokes = strokes.copy()
                if self.scale:
                    strokes[:, 1] = (strokes[:, 1] - self.mean1) / self.std1
                    strokes[:, 2] = (strokes[:, 2] - self.mean2) / self.std2
                strokes = strokes.reshape((1,) + strokes.shape)
                # We want (x3, x1, x2) --> (x1, x2, x3)
                strokes = tf.dtypes.cast(tf.gather(strokes, [1, 2, 0], axis=2), dtype=float)
                yield strokes[:, :-1, :], strokes[:, 1:, :]


class DataSynthesis(Data):

    def __init__(
        self,
        path_to_sentences='data/sentences.txt',
        clean_text=True,
        path_to_data='data/strokes-py3.npy',
        train_split=0.8,
        scale=True,
    ):
        super(DataSynthesis, self).__init__(
            path_to_data=path_to_data,
            path_to_sentences=path_to_sentences,
            clean_text=clean_text,
            train_split=train_split,
        )
        self.scale = scale

    def batch_generator(self, batch_size=1, shuffle=True, data_type='train'):
        assert data_type in ['train', 'validation']
        if data_type == 'train':
            all_strokes = self.strokes
            all_sentences = self.sentences
        elif data_type == 'validation':
            all_strokes = self.validation_strokes
            all_sentences = self.validation_sentences
        idx = np.arange(0, len(all_sentences))
        while True:
            if shuffle:
                np.random.shuffle(idx)

            for it in idx:
                strokes, sentences = all_strokes[it].copy(), all_sentences[it].copy()
                if self.scale:
                    strokes[:, 1] = (strokes[:, 1] - self.mean1) / self.std1
                    strokes[:, 2] = (strokes[:, 2] - self.mean2) / self.std2
                # We want (x3, x1, x2) --> (x1, x2, x3)
                strokes = tf.dtypes.cast(tf.gather(strokes, [1, 2, 0], axis=1), dtype=float)
                sentences = tf.dtypes.cast(sentences, dtype=float)
                yield strokes[:-1, :], sentences, strokes[1:, :]
