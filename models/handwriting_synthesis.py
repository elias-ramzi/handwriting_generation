import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RNN,
    Dense, Concatenate,
)

from models.base_model import BaseModel
from models.custom_layer import WindowedLSTMCell


class SamplingFinished(Exception):
    pass


class HandWritingSynthesis(BaseModel, Model):

    def __init__(
        self,
        vocab_size=61,
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
        verbose=False,
    ):
        Model.__init__(self)
        BaseModel.__init__(
            self,
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
        self.hidden_dim = 400

        self.windowedlstm = RNN(
            cell=WindowedLSTMCell(
                units=400,
                window_size=self.vocab_size,
                mixtures=30,
            ),
            return_sequences=True,
            return_state=True,
            name='h1'
        )

        self.lstm2 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            name='h2',
        )

        self.lstm3 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            name='h3',
            )

        self.mixture = Dense(121, name='MixtureCoef')

        # Used for gradient cliping the lstm's
        self.to_clip += ['h{}/kernel:0'.format(i) for i in range(1, 3+1)]
        self.to_clip += ['h{}/recurrent_kernel:0'.format(i) for i in range(1, 3+1)]
        self.to_clip += ['h{}/bias:0'.format(i) for i in range(1, 3+1)]

        self.optimizer = tf.keras.optimizers.RMSprop(
            lr=self.lr,
            rho=self.rho,
            momentum=self.momentum,
            epsilon=self.epsilon,
            centered=self.centered,
        )

    def call(self, strokes, sentence, states):
        wlstm1, stateh1, statec1, out_window, kappa, phi, alpha, beta = self.windowedlstm(
            inputs=strokes,
            initial_state=states[0:2] + states[-5:],
            constants=sentence,
        )
        lstm1 = wlstm1[:, :, :400]
        out_window = wlstm1[:, :, 400:]

        _input2 = tf.concat([strokes, out_window, lstm1], axis=-1, name='skip1')
        lstm2, stateh2, statec2 = self.lstm2(_input2, initial_state=states[2:4])

        _input3 = tf.concat([strokes, out_window, lstm2], axis=-1, name='Skip2')
        lstm3, stateh3, statec3 = self.lstm3(_input3, initial_state=states[4:6])

        lstm = tf.concat([lstm1, lstm2, lstm3], axis=-1, name='Skip3')
        y_hat = self.mixture(lstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        output_states = [
            stateh1, statec1,
            stateh2, statec2,
            stateh3, statec3,
            out_window, kappa, phi, alpha, beta,
        ]

        return mixture_coefs, output_states

    def train(self, strokes, sentence, states, targets):
        with tf.GradientTape() as tape:
            outputs = self(strokes, sentence, states)
            predictions, _ = outputs
            targets = tf.dtypes.cast(targets, dtype=float)
            loss = self.loss_function(targets, predictions)

            gradients = tape.gradient(loss, self.trainable_variables)

        # Clips gradient for output Dense layer
        gradients[-1] = tf.clip_by_value(gradients[-1], -100.0, 100.0)
        gradients[-2] = tf.clip_by_value(gradients[-2], -100.0, 100.0)

        # Clips gradient for LSTM layers
        for i, grad in enumerate(gradients):
            name = self.trainable_variables[i].name
            if name in self.to_clip:
                gradients[i] = tf.clip_by_value(gradients[i], -10.0, 10.0)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def validation(self, strokes, sentence, states, targets):
        outputs = self(strokes, sentence, states)
        predictions, _ = outputs
        targets = tf.dtypes.cast(targets, dtype=float)
        loss = self.loss_function(targets, predictions)
        return loss

    def infer(
        self, sentence, bias=None,
        inf_type='max',
        weights_path=None, reload=False,
        verbose=None, seed=None
    ):
        np.random.seed(seed)
        if verbose:
            msg = ("Writing : \033[92m {sentence}\033[00m,"
                   " \033[93m {length}\033[00m strokes computed")
        else:
            msg = "Writing, \033[93m {length}\033[00m strokes computed"

        X = np.array([[[0., 0., 1.]]])
        input_states = [
            tf.zeros((1, self.hidden_dim)),  # stateh1
            tf.zeros((1, self.hidden_dim)),  # statec1
            tf.zeros((1, self.hidden_dim)),  # stateh2
            tf.zeros((1, self.hidden_dim)),  # statec2
            tf.zeros((1, self.hidden_dim)),  # stateh3
            tf.zeros((1, self.hidden_dim)),  # statec3
            tf.zeros((1, self.vocab_size)),  # in_window
            tf.zeros((1, 10)),  # kappa
            tf.zeros((1, 1)),  # phi
            tf.zeros((1, 10)),  # alpha
            tf.zeros((1, 10)),  # beta
        ]
        strokes = []
        length = 1
        windows = []
        phis = []
        kappas = []
        alphas = []
        betas = []

        while length < 1300:
            try:
                mixture_coefs, output_states =\
                    self(X, sentence, input_states, training=False)

                # Heuristic described in paper phi(t, U+1) > phi(t, u) for 0<u<U+1
                kappa = output_states[-4]
                phi = output_states[-3]
                alpha = output_states[-2]
                beta = output_states[-1]
                last_phi = tf.reduce_sum(
                    alpha * tf.math.exp(- beta * (kappa - sentence.shape[1]+1)**2),
                    axis=1,)
                if phi < last_phi:
                    raise SamplingFinished
                phis.append(phi)
                kappas.append(kappa)
                windows.append(output_states[-5])
                alphas.append(alpha)
                betas.append(beta)

                end_stroke, x, y = self._infer(mixture_coefs, inf_type=inf_type, bias=bias)

                # Next inputs
                X = np.array([x, y, end_stroke]).reshape((1, 1, 3))
                # as the output states in the model the window
                # is the slice of the output (which is a sequence)  [line 96]
                output_states[-5] = output_states[-5][:, 0, :]
                input_states = output_states
                # Our sentence written
                strokes.append((end_stroke, x, y))

                print(msg.format(sentence=verbose, length=length), end='\r')
                length += 1

            except SamplingFinished:
                break

        print()
        print("Sampling finished, produced "
              "sequence of length :\033[92m {}\033[00m".format(length))
        return np.vstack(strokes), windows, phis, kappas, alphas, betas
