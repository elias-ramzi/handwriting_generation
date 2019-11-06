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


class HandWritingSynthesis(BaseModel):

    def __init__(
        self,
        char_length=64,
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
        self.char_length = char_length
        self.vocab_size = vocab_size
        self.regularizer = self.regularization()
        self.verbose = verbose
        self.training = False
        self.num_layers = 3
        self.hidden_dim = 400

    def _make_model(self):

        strokes = Input((1, 3), batch_size=1, name='strokes')
        sentence = Input((self.char_length, self.vocab_size), batch_size=1, name='sentence')
        in_window = Input((self.vocab_size), batch_size=1, name='inWindow')
        kappa = Input(10, batch_size=1, name='kappa')
        phi = Input(self.char_length + 1, batch_size=1, name='phi')
        stateh1 = Input(400, batch_size=1, name='stateh1')
        statec1 = Input(400, batch_size=1, name='statec1')
        stateh2 = Input(400, batch_size=1, name='stateh2')
        statec2 = Input(400, batch_size=1, name='statec2')
        stateh3 = Input(400, batch_size=1, name='stateh3')
        statec3 = Input(400, batch_size=1, name='statec3')
        input_states = [
            stateh1, statec1,
            in_window, kappa, phi, sentence,
            stateh2, statec2,
            stateh3, statec3,
        ]

        wlstm1, stateh1, statec1, out_window, kappa, phi, _ = RNN(
            cell=WindowedLSTMCell(
                units=400,
                window_size=self.vocab_size,
                char_length=self.char_length,
                mixtures=30,
            ),
            return_sequences=True,
            return_state=True,
            name='h1'
        )(
            inputs=strokes,
            initial_state=input_states[0:6]
        )
        lstm1 = wlstm1[:, :, :400]
        out_window = wlstm1[:, :, 400:]

        _input2 = Concatenate(name='skip1')([strokes, out_window, lstm1])
        lstm2, stateh2, statec2 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h2',
        )(_input2, initial_state=input_states[6:8])

        _input3 = Concatenate(name='Skip2')([strokes, out_window, lstm2])
        lstm3, stateh3, statec3 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h3',
            )(_input3, initial_state=input_states[8:])

        lstm = Concatenate(name='Skip3')([lstm1, lstm2, lstm3])
        y_hat = Dense(121, name='MixtureCoef')(lstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        output_states = [
            stateh1, statec1,
            out_window, kappa, phi, sentence,
            stateh2, statec2,
            stateh3, statec3
        ]
        model = Model(
            inputs=[strokes, input_states],
            outputs=[mixture_coefs, output_states]
        )

        # Used for gradient cliping the lstm's
        self.to_clip += ['h{}/kernel:0'.format(i) for i in range(1, 3+1)]
        self.to_clip += ['h{}/recurrent_kernel:0'.format(i) for i in range(1, 3+1)]
        self.to_clip += ['h{}/bias:0'.format(i) for i in range(1, 3+1)]

        return model

    def make_model(self, load_weights=None):
        self.model = self._make_model()
        if load_weights is not None:
            self.model.load_weights(load_weights)
        if self.verbose:
            self.model.summary()

    def infer(
        self, sentence, bias=None,
        inf_type='max',
        weights_path=None, reload=False,
        verbose=None, seed=None
    ):
        np.random.seed(seed)
        if not hasattr(self, 'model') or reload:
            self.make_model(weights_path=weights_path)
        if verbose:
            msg = ("Writing : \033[92m {sentence}\033[00m,"
                   " \033[93m {length}\033[00m strokes computed")
        else:
            msg = "Writing, \033[93m {length}\033[00m strokes computed"

        X = np.array([[[0., 0., 1.]]])
        input_states = [
            tf.zeros((1, self.hidden_dim)),  # stateh1
            tf.zeros((1, self.hidden_dim)),  # statec1
            tf.zeros((1, self.vocab_size)),  # in_window
            tf.zeros((1, 10)),  # kappa
            tf.zeros((1, self.char_length + 1)),  # phi
            sentence,  # sentence
            tf.zeros((1, self.hidden_dim)),  # stateh2
            tf.zeros((1, self.hidden_dim)),  # statec2
            tf.zeros((1, self.hidden_dim)),  # stateh3
            tf.zeros((1, self.hidden_dim)),  # statec3
        ]
        strokes = []
        length = 1
        windows = []
        phis = []
        kappas = []

        while length < 1300:
            try:
                mixture_coefs, output_states =\
                    self.model([X, input_states], training=False)
                windows.append(output_states[2])
                # Heuristic described in paper phi(t, U+1) > phi(t, u) for 0<u<U+1
                phi = output_states[4]
                if tf.reduce_all(phi[0, :-1] < phi[0, -1]):
                    raise SamplingFinished
                phis.append(phi[0, :-1])
                kappas.append(output_states[3])

                end_stroke, x, y = self._infer(mixture_coefs, inf_type=inf_type, bias=bias)

                # Next inputs
                X = np.array([x, y, end_stroke]).reshape((1, 1, 3))
                # as the output states in the model the window
                # is the slice of the output (which is a sequence)  [line 96]
                output_states[2] = output_states[2][:, 0, :]
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
        return np.vstack(strokes), windows, phis, kappas
