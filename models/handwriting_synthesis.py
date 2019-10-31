import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM,
    Dense, Concatenate,
)

from models.base_model import BaseModel


class HandWritingSynthesis(BaseModel):

    def __init__(
        self,
        vocab_size=700,
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
        self.compile = _compile
        self.regularizer = self.regularization()
        self.verbose = verbose

    def attention(self, sentence, h1):
        alpha = tf.math.exp(h1[:, :, 0:10]),
        beta = tf.math.exp(h1[:, :, 10, 20]),
        kappa = tf.math.exp(h1[:, :, 20, 30])
        kappa += tf.concat(tf.zeros(1), kappa[:-1])
        w = []
        for u in range(h1.shape[1]):
            w.append(
                sentence
                * tf.reduce_sum(
                    alpha
                    * tf.math.exp(- beta * (kappa - u)**2),
                    axis=2,)
            )

        return tf.reduce_sum(w)

    def _make_model(self):

        strokes = Input((None, 3))
        sentence = Input(self.vocab_size)

        lstm1, previous_state = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            name='h1',
            )(strokes)

        _window_coef = Dense(30)(lstm1)
        window = self.compute_window(sentence, _window_coef)

        _input2 = Concatenate(name='Skip1')([strokes, window, lstm1])
        lstm2 = LSTM(
            400,
            return_sequences=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h2',
        )(_input2)

        _input3 = Concatenate(name='Skip2')([strokes, window, lstm2])
        _input3 = lstm2
        lstm = LSTM(
            400,
            return_sequences=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h3',
            )(_input3)

        lstm = Concatenate(name='Skip3')([lstm1, lstm2, lstm])
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
            self.to_clip += [f'h{i}/kernel:0' for i in range(1, 3+1)]
            self.to_clip += [f'h{i}/recurrent_kernel:0' for i in range(1, 3+1)]
            self.to_clip += [f'h{i}/bias:0' for i in range(1, 3+1)]

        return model

    def make_model(self, load_weights=None):
        self.model = self._make_model()
        if load_weights is not None:
            self.model.load_weights(load_weights)
        if self.verbose:
            self.model.summary()
