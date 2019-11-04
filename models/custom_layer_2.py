import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.layers import LSTM, Concatenate, Dense


class WindowedLSTMCell(Layer):

    def __init__(
        self,
        units,
        char_length,
        vocab_size,
        mixtures,
        **kwargs
    ):
        super(WindowedLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.char_length = char_length
        self.vocab_size = vocab_size
        self.mixtures = mixtures
        self.output_size = units * 3
        self.state_size = data_structures.NoDependency([
            self.units,  # h1[t-1]
            self.units,  # c1[t-1]
            self.units,  # h2[t-1]
            self.units,  # c2[t-1]
            self.units,  # h3[t-1]
            self.units,  # c3[t-1]
            self.vocab_size,  # window[t-1]
            self.mixtures//3,  # kappa[t-1]
            self.char_length+1,  # phi[t-1]
            tf.TensorShape((self.char_length, self.vocab_size)),  # sentence
        ])

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self._trainable_weights = []
        self.lstm1 = LSTM(
                self.units,
                return_state=True,
                name='h1',
            )
        self.lstm1.build(input_shape)
        self._trainable_weights += self.lstm1.trainable_weights
        self.dense = Dense(30)
        self.dense.build(input_shape)
        self._trainable_weights += self.dense.trainable_weights
        self.lstm2 = LSTM(
                self.units,
                return_state=True,
                name='h2',
            )
        self.lstm2.build(input_shape)
        self._trainable_weights += self.lstm2.trainable_weights
        self.lstm3 = LSTM(
                self.units,
                return_state=True,
                name='h3',
            )
        self.lstm3.build(input_shape)
        self._trainable_weights += self.lstm3.trainable_weights
        self.build = True
        super(WindowedLSTMCell, self).build(input_shape)

    def _compute_window(self, sentence, h, kappa_prev):
        alpha, beta, kappa = tf.split(h, num_or_size_splits=3, axis=1)
        alpha = tf.math.exp(alpha)
        beta = tf.math.exp(beta)
        kappa = kappa_prev + tf.math.exp(kappa)
        phi = []
        # indexing starts at 1 in the paper
        U = tf.range(start=1, limit=self.char_length+1)
        U = tf.dtypes.cast(tf.stack([U]*10, axis=1), dtype=float)
        phi = alpha * tf.math.exp(- beta * (kappa - U)**2)
        phi = tf.reduce_sum(phi, axis=1)
        phi = tf.reshape(phi, (1, self.char_length))
        w = tf.squeeze(tf.matmul(phi, sentence), axis=1)
        # Heuristic described in paper phi(t, U+1) > phi(t, u) for 0<u<U+1
        last_phi = tf.reduce_sum(
            alpha * tf.math.exp(- beta * (kappa - self.char_length+1)**2),
            axis=1,)
        last_phi = tf.reshape(last_phi, (1, 1))
        phi = tf.concat((phi, last_phi), axis=-1)
        return w, kappa, phi

    def call(self, inputs, states):
        _input1 = Concatenate(name='In')([inputs, states[-4]])
        _input1 = tf.expand_dims(_input1, 1)
        lstm1, stateh1, statec1 = LSTM(
                self.units,
                return_state=True,
                name='h1',
            )(inputs=_input1, initial_state=states[0:2])

        window_coef = Dense(30)(lstm1)
        out_window, kappa, phi = self._compute_window(states[-1], window_coef, states[-3])

        _input2 = Concatenate()([inputs, out_window, lstm1])
        _input2 = tf.expand_dims(_input2, 1)
        lstm2, stateh2, statec2 = LSTM(
            self.units,
            return_state=True,
            name='h2',
        )(_input2, initial_state=states[2:4])

        _input3 = Concatenate(name='Skip2')([inputs, out_window, lstm2])
        _input3 = tf.expand_dims(_input3, 1)
        lstm3, stateh3, statec3 = LSTM(
            self.units,
            return_state=True,
            name='h3',
            )(_input3, initial_state=states[4:6])

        output_states = [
            stateh1, statec1,
            stateh2, statec2,
            stateh3, statec3,
            out_window, kappa, phi, states[-1],
        ]

        output = Concatenate()([lstm1, lstm2, lstm3])
        return output, output_states

    def get_initial_state(self, inputs=None, batch_size=1, dtype=None):
        return [tf.zeros(shape) for shape in self.state_size]
