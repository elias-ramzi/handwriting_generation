"""
Implementing the Handwriting Prediction section of
the paper found here https://arxiv.org/pdf/1308.0850.pdf by Alex Graves
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM,
    Dense, Add, Concatenate
)
from tqdm import tqdm

from models.base_model import BaseModel


class HandWritingPrediction(BaseModel):

    def __init__(
        self,
        train_seq_length=700,
        _compile=True,
        lstm='stacked',
        regularizer_type='gaussian',
        reg_mean=0.,
        reg_std=0.,
        reg_l2=0.,
        lr=0.0001,
        rho=0.95,
        momentum=0.9,
        epsilon=0.0001,
        centered=True,
        inf_type='sum',
        inf_n_jobs=-2,
        inf_backend='threading',
        verbose=True,
    ):
        super(HandWritingPrediction, self).__init__(
            regularizer_type=regularizer_type,
            mean=reg_mean,
            std=reg_std,
            l2=reg_l2,
            lr=lr,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
            inf_type=inf_type,
            inf_n_jobs=inf_n_jobs,
            inf_backend=inf_backend,
        )
        self.train_seq_length = train_seq_length
        self.compile = _compile
        assert lstm in ['single', 'stacked'],\
            "You should pass either lstm='single or stacked'"
        self.lstm = lstm
        self.regularizer = self.regularization()
        self.verbose = verbose

    def _make_model(self, seq_length):

        strokes = Input((seq_length, 3))

        if self.lstm == 'single':
            self.num_layers = 1
            lstm = LSTM(
                900,
                name='h1',
                return_sequences=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
            )(strokes)

        elif self.lstm == 'stacked':
            self.num_layers = 3
            lstm1 = LSTM(
                400,
                return_sequences=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
                name='h1',
                )(strokes)

            # skip1 = Dense(400, name='Wih2', use_bias=False)(strokes)
            _input2 = Concatenate(name='Skip1')([strokes, lstm1])
            lstm2 = LSTM(
                400,
                return_sequences=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
                name='h2',
            )(_input2)

            # skip2 = Dense(400, name='Wih3', use_bias=False)(strokes)
            _input3 = Concatenate(name='Skip2')([strokes, lstm2])
            lstm3 = LSTM(
                400,
                return_sequences=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
                name='h3',
                )(_input3)

            # skip31 = Dense(400, name='Wh1y', use_bias=False)(lstm1)
            # skip32 = Dense(400, name='Wh2y', use_bias=False)(lstm2)
            lstm = Concatenate(name='Skip3')([lstm1, lstm2, lstm3])

        y_hat = Dense(121, name='MixtureCoef')(lstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        model = Model(inputs=strokes, outputs=mixture_coefs)

        if self.compile:
            optimizer = tf.keras.optimizers.RMSprop(
                    lr=self.lr,
                    rho=self.rho,
                    momentum=self.momentum,
                    epsilon=self.epsilon,
                    centered=self.centered,
                    clipvalue=10,
                )

            model.compile(
                optimizer,
                loss=self.loss_function,
            )

        else:
            self.to_clip += [f'h{i}/kernel:0' for i in range(1, self.num_layers+1)]
            self.to_clip += [f'h{i}/recurrent_kernel:0' for i in range(1, self.num_layers+1)]
            self.to_clip += [f'h{i}/bias:0' for i in range(1, self.num_layers+1)]

        return model

    def make_model(self, load_weights=None):
        self.model = self._make_model(self.train_seq_length)
        if load_weights is not None:
            self.model.load_weights(load_weights)
        if self.verbose:
            self.model.summary()

    def make_infer_model(self, weights_path=None):
        self.infer_model = self._make_model(1)
        if weights_path is None:
            self.model.save_weights('models/trained/models_tmp.h5')
            self.infer_model.load_weights('models/trained/models_tmp.h5')
        else:
            self.infer_model.load_weights(weights_path)

    def infer(self, length=700, inf_type=None, weights_path=None):
        if not hasattr(self, 'infer_model'):
            self.make_infer_model(weights_path=weights_path)
        X = tf.zeros((1, 1, 3))
        strokes = []
        for _ in tqdm(range(length), desc='Creating a series of strokes'):
            mixture_coefs = self.infer_model.predict(X)
            end_stroke, x, y = self._infer(mixture_coefs, inf_type)
            X = np.array([x, y, end_stroke])
            X = X.reshape((1, 1, 3))
            strokes.append((end_stroke, x, y))
        return np.vstack(strokes)
