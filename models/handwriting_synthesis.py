import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM,
    Dense, Concatenate,
)

from models.base_model import BaseModel


class SamplingFinished(Exception):
    pass


class HandWritingSynthesis(BaseModel):

    def __init__(
        self,
        vocab_size,
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
        super(HandWritingSynthesis, self).__init__(
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
        self.vocab_size = vocab_size
        self.regularizer = self.regularization()
        self.verbose = verbose
        self.training = False
        self.num_layers = 3

    def set_train(self):
        self.training = True

    def set_predict(self):
        self.training = False

    def attention(self, sentence, h1):
        alpha = tf.math.exp(h1[:, :, 0:10]),
        beta = tf.math.exp(h1[:, :, 10, 20]),
        kappa = tf.math.exp(h1[:, :, 20, 30])
        kappa += tf.concat(tf.zeros(1), kappa[:-1])
        w = []
        phi = []
        # this sum forces to have a batch_size of one
        U = tf.reduce_sum(sentence)
        for u in range(U):
            weight = tf.reduce_sum(
                alpha
                * tf.math.exp(- beta * (kappa - u)**2),
                axis=2,)
            phi.append(weight)
            w.append(sentence * weight)

        if not self.training:
            last_phi = tf.reduce_sum(
                alpha
                * tf.math.exp(- beta * (kappa - (U+1)**2)),
                axis=2,)
            phi = np.array(phi)
            # Heuristic described in paper phi(t, U+1) > phi(t, u) for 0<u<U+1
            if np.all(phi < last_phi):
                raise SamplingFinished('Sampling finished')

        return tf.reduce_sum(w)

    def _make_model(self):

        strokes = Input((None, 3))
        sentence = Input(self.vocab_size)
        in_window = Input(self.vocab_size)
        stateh1 = Input(900)
        statec1 = Input(900)
        stateh2 = Input(900)
        statec2 = Input(900)
        stateh3 = Input(900)
        statec3 = Input(900)
        input_states = [in_window, stateh1, statec1, stateh2, statec2, stateh3, statec3]

        _input1 = Concatenate()([in_window, strokes])
        lstm1, stateh1, statec1 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h1',
            )(_input1, initial_state=input_states[0:2])

        _window_coef = Dense(30)(lstm1)
        phi, out_window = self.attention(sentence, _window_coef)
        output_states = [out_window]
        output_states += [stateh1, statec1]

        _input2 = Concatenate(name='Skip1')([out_window, strokes, lstm1])
        lstm2, stateh2, statec2 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h2',
        )(_input2, initial_state=input_states[2:4])
        output_states += [stateh1, statec1]

        _input3 = Concatenate(name='Skip2')([out_window, strokes, lstm2])
        lstm3, stateh3, statec3 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h3',
            )(_input3, initial_state=input_states[4:6])
        output_states += [stateh1, statec1]

        lstm = Concatenate(name='Skip3')([lstm1, lstm2, lstm3])
        y_hat = Dense(121, name='MixtureCoef')(lstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        model = Model(
            inputs=[strokes, input_states],
            outputs=[mixture_coefs, output_states]
        )

        # Used for gradient cliping
        self.to_clip += [f'h{i}/kernel:0' for i in range(1, 3+1)]
        self.to_clip += [f'h{i}/recurrent_kernel:0' for i in range(1, 3+1)]
        self.to_clip += [f'h{i}/bias:0' for i in range(1, 3+1)]

        return model

    def make_model(self, load_weights=None):
        self.set_train()
        self.model = self._make_model()
        if load_weights is not None:
            self.model.load_weights(load_weights)
        if self.verbose:
            self.model.summary()

    def infer(self, sentence, inf_type=None, weights_path=None):
        self.set_predict()
        X = tf.zeros((1, 1, 3))
        input_states = [tf.zeros(1, self.vocab_size)]
        input_states += [tf.zeros((1, self.hidden_dim))] * 2 * self.num_layers
        strokes = []
        length = 1
        while True:
            try:
                mixture_coefs, output_states = self.infer_model([X, input_states], training=False)
                end_stroke, x, y = self._infer(mixture_coefs, inf_type)
                X = np.array([x, y, end_stroke]).reshape((1, 1, 3))
                input_states = output_states
                strokes.append((end_stroke, x, y))
                print(f"Stroke {length} computed", end='\r')
                length += 1
            except SamplingFinished:
                print(f"Sampling finished, produced sequence of length : {length}")
                break
        return np.vstack(strokes)
