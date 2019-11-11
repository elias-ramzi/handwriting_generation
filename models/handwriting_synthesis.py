import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, RNN, Dense, Concatenate, Input

from models.base_model import BaseModel
from models.custom_layer import WindowedLSTMCell


class HandWritingSynthesis(BaseModel):

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
        verbose=False,
    ):
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
        )
        self.vocab_size = vocab_size
        self.regularizer = self.regularization()
        self.verbose = verbose
        self.training = False
        self.num_layers = 3
        self.hidden_dim = 400

        self.optimizer = tf.keras.optimizers.RMSprop(
            lr=self.lr,
            rho=self.rho,
            momentum=self.momentum,
            epsilon=self.epsilon,
            centered=self.centered,
        )

    def __call__(self, strokes, sentence, states):
        if not hasattr(self, 'model'):
            self.make_model()
        self.model(strokes, sentence, states)

    def make_model(self):
        strokes = Input((None, 3))
        sentence = Input((None, self.vocab_size))
        stateh1 = Input((400,))
        statec1 = Input((400,))
        stateh2 = Input((400,))
        statec2 = Input((400,))
        stateh3 = Input((400,))
        statec3 = Input((400,))
        in_window = Input((self.vocab_size,))
        kappa_prev = Input((10,))
        phi_prev = Input((1,))
        alpha_prev = Input((10,))
        beta_prev = Input((10,))

        states = [
            stateh1, statec1,
            stateh2, statec2,
            stateh3, statec3,
            in_window, kappa_prev, phi_prev, alpha_prev, beta_prev,
        ]

        wlstm1, stateh1, statec1, out_window, kappa, phi, alpha, beta = RNN(
            cell=WindowedLSTMCell(
                units=400,
                window_size=self.vocab_size,
                mixtures=30,
            ),
            return_sequences=True,
            return_state=True,
            name='h1'
        )(
            inputs=strokes,
            initial_state=states[0:2] + states[-5:],
            constants=sentence,
        )
        lstm1 = wlstm1[:, :, :400]
        out_window = wlstm1[:, :, 400:]

        _input2 = Concatenate(axis=-1, name='skip1')([strokes, out_window, lstm1])
        lstm2, stateh2, statec2 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            name='h2',
        )(_input2, initial_state=states[2:4])

        _input3 = Concatenate(axis=-1, name='Skip2')([strokes, out_window, lstm2])
        lstm3, stateh3, statec3 = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            name='h3',
        )(_input3, initial_state=states[4:6])

        lstm = Concatenate(axis=-1, name='Skip3')([lstm1, lstm2, lstm3])
        y_hat = Dense(121, name='MixtureCoef')(lstm)

        e = tf.nn.sigmoid(-y_hat[:, :, 0: 1])
        pi, mu1, mu2, sigma1, sigma2, rho = tf.split(
            y_hat[:, :, 1:],
            num_or_size_splits=6,
            axis=2,
        )
        pi = tf.nn.softmax(pi)
        sigma1 = tf.math.exp(sigma1) + 10**(-4)
        sigma2 = tf.math.exp(sigma2) + 10**(-4)
        rho = tf.nn.tanh(rho)
        mixture_coefs = Concatenate(name='output', axis=-1)([
            e, pi, mu1, mu2,
            sigma1, sigma2, rho,
            ])

        output_states = [
            stateh1, statec1,
            stateh2, statec2,
            stateh3, statec3,
            out_window, kappa, phi, alpha, beta,
        ]

        model = Model(inputs=[strokes, sentence, states], outputs=[mixture_coefs, output_states])
        model.summary()
        self.model = model

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
        for i, grad in enumerate(gradients[:-2]):
            gradients[i] = tf.clip_by_value(gradients[i], -10.0, 10.0)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.windowedlstm.cell.phi = []
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
        self.windowedlstm.cell.phi = []
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
        kappas = []
        alphas = []
        betas = []

        while length < 1300:
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
            if phi <= last_phi:
                break
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

        print()
        print("Sampling finished, produced "
              "sequence of length :\033[92m {}\033[00m".format(length))
        phis = self.windowedlstm.cell.phi
        return np.vstack(strokes), windows, phis, kappas, alphas, betas
