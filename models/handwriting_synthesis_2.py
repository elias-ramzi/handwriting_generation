import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, RNN,
    Dense,
)

from models.base_model import BaseModel
from models.custom_layer_2 import WindowedLSTMCell


class SamplingFinished(Exception):
    pass


class HandWritingSynthesis(BaseModel):

    def __init__(
        self,
        char_length=53,
        vocab_size=60,
        sequence_lenght=1191,
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
        self.sequence_lenght = sequence_lenght
        self.regularizer = self.regularization()
        self.verbose = verbose
        self.training = False
        self.num_layers = 3
        self.hidden_dim = 400

    def _make_model(self):

        strokes = Input((None, 3), batch_size=1, name='strokes')

        stateh1 = Input(400, batch_size=1, name='stateh1')
        statec1 = Input(400, batch_size=1, name='statec1')
        stateh2 = Input(400, batch_size=1, name='stateh2')
        statec2 = Input(400, batch_size=1, name='statec2')
        stateh3 = Input(400, batch_size=1, name='stateh3')
        statec3 = Input(400, batch_size=1, name='statec3')
        sentence = Input((self.char_length, self.vocab_size), batch_size=1, name='sentence')
        in_window = Input((self.vocab_size), batch_size=1, name='inWindow')
        kappa = Input(10, batch_size=1, name='kappa')
        phi = Input(self.char_length + 1, batch_size=1, name='phi')
        input_states = [
            stateh1, statec1,
            stateh2, statec2,
            stateh3, statec3,
            in_window, kappa, phi, sentence,
        ]

        ouputs = RNN(
            cell=WindowedLSTMCell(
                units=400,
                vocab_size=self.vocab_size,
                char_length=self.char_length,
                mixtures=30,
            ),
            return_sequences=True,
            return_state=True,
        )(
            inputs=strokes,
            initial_state=input_states
        )
        wlstm = ouputs[0]

        y_hat = Dense(121, name='MixtureCoef')(wlstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        model = Model(
            inputs=[strokes, input_states],
            outputs=[mixture_coefs, ouputs[:1]]
        )

        # Used for gradient cliping the lstm's
        self.to_clip = ['rnn/recurrent_kernel:0', 'rnn/kernel:0', 'rnn/bias:0']

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
        verbose=None
    ):
        if not hasattr(self, 'model') or reload:
            self.make_model(weights_path=weights_path)
        if verbose:
            msg = ("Writing : \033[92m {sentence}\033[00m,"
                   " \033[93m {length}\033[00m strokes computed")
        else:
            msg = "Writing, \033[93m {length}\033[00m strokes computed"

        X = tf.zeros((1, 1, 3))
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

        while length < 1300:
            try:
                mixture_coefs, output_states =\
                    self.model([X, input_states], training=False)

                # Heuristic described in paper phi(t, U+1) > phi(t, u) for 0<u<U+1
                phi = output_states[4]
                if tf.reduce_all(phi[0, :-1] < phi[0, -1]):
                    raise SamplingFinished

                end_stroke, x, y = self._infer(mixture_coefs, inf_type=inf_type, bias=bias)

                # Next inputs
                X = np.array([x, y, end_stroke]).reshape((1, 1, 3))
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
        return np.vstack(strokes)
