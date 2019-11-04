"""
Implementing the Handwriting Prediction section of
the paper found here https://arxiv.org/pdf/1308.0850.pdf by Alex Graves
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM,
    Dense, Concatenate
)
from tqdm import tqdm

from models.base_model import BaseModel


class HandWritingPrediction(BaseModel):

    def __init__(
        self,
        train_seq_length=700,
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
        assert lstm in ['single', 'stacked'],\
            "You should pass either lstm='single or stacked'"
        self.lstm = lstm
        self.regularizer = self.regularization()
        self.verbose = verbose

    def _make_model(self, seq_length, train=True):
        if not train:
            seq_length = 1
        strokes = Input((seq_length, 3))
        output_states = []

        if self.lstm == 'single':
            stateh1 = Input(900)
            statec1 = Input(900)
            input_states = [stateh1, statec1]
            self.num_layers = 1
            self.hidden_dim = 900
            lstm, stateh1, statec1 = LSTM(
                900,
                name='h1',
                return_sequences=True,
                return_state=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
            )(strokes, initial_state=input_states)
            output_states += [stateh1, statec1]

        elif self.lstm == 'stacked':
            stateh1 = Input(400)
            statec1 = Input(400)
            stateh2 = Input(400)
            statec2 = Input(400)
            stateh3 = Input(400)
            statec3 = Input(400)
            input_states = [stateh1, statec1, stateh2, statec2, stateh3, statec3]
            self.num_layers = 3
            self.hidden_dim = 400
            lstm1, stateh1, statec1 = LSTM(
                400,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
                name='h1',
                )(strokes, initial_state=input_states[0:2])
            output_states += [stateh1, statec1]

            # skip1 = Dense(400, name='Wih2', use_bias=False)(strokes)
            _input2 = Concatenate(name='Skip1')([strokes, lstm1])
            lstm2, stateh2, statec2 = LSTM(
                400,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
                name='h2',
            )(_input2, initial_state=input_states[2:4])
            output_states += [stateh2, statec2]

            # skip2 = Dense(400, name='Wih3', use_bias=False)(strokes)
            _input3 = Concatenate(name='Skip2')([strokes, lstm2])
            lstm3, stateh3, statec3 = LSTM(
                400,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=self.regularizer,
                recurrent_regularizer=self.regularizer,
                name='h3',
                )(_input3, initial_state=input_states[4:6])
            output_states += [stateh3, statec3]

            # skip31 = Dense(400, name='Wh1y', use_bias=False)(lstm1)
            # skip32 = Dense(400, name='Wh2y', use_bias=False)(lstm2)
            lstm = Concatenate(name='Skip3')([lstm1, lstm2, lstm3])

        y_hat = Dense(121, name='MixtureCoef')(lstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        model = Model(
            inputs=[strokes, input_states],
            outputs=[mixture_coefs, output_states]
        )

        # Used for gradient cliping the lstm's
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

    def infer(self, seed=0, inf_type=None, weights_path=None, reload=False):
        if not hasattr(self, 'infer_model') or reload:
            self.make_infer_model(weights_path=weights_path)
        np.random.seed(seed)
        length = np.random.randint(400, 1200)
        print()
        print("Generating a random sentence of \033[92m {length}\033[00m strokes")
        print()

        X = tf.zeros((1, 1, 3))
        input_states = [tf.zeros((1, self.hidden_dim))] * 2 * self.num_layers
        strokes = []
        for _ in tqdm(range(length), desc='Creating a series of strokes'):
            mixture_coefs, output_states = self.infer_model([X, input_states], training=False)
            end_stroke, x, y = self._infer(mixture_coefs, inf_type)
            X = np.array([x, y, end_stroke]).reshape((1, 1, 3))
            input_states = output_states
            strokes.append((end_stroke, x, y))
        return np.vstack(strokes)
