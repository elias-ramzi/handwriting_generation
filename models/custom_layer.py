import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.training.tracking import data_structures


class WindowedLSTMCell(Layer):
    """
    adapted from
    https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/layers/recurrent.py#L2043
    """

    def __init__(
        self,
        units,
        window_size,
        char_length,
        mixtures,
        activation='tanh',
        recurrent_activation='sigmoid',
        # use_bias=True, force to True to make it easier to implement
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        window_initializer='glorot_uniform',
        bias_initializer='zeros',
        mixture_initializer='glorot_uniform',
        bias_mixture_initializer='zeros',
        # unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        window_regularizer=None,
        bias_regularizer=None,
        mixture_regularizer=None,
        bias_mixture_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        # dropout=0.,
        # recurrent_dropout=0., Not implementing dropout for now
        # implementation=1,  the default is 2 in LSTM layer
        **kwargs,
    ):
        super(WindowedLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.window_size = window_size
        self.char_length = char_length
        self.mixtures = mixtures
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        # self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.window_initializer = initializers.get(window_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.mixture_initializer = initializers.get(mixture_initializer)
        self.bias_mixture_initializer = initializers.get(bias_mixture_initializer)
        # self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.window_regularizer = regularizers.get(window_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.mixture_regularizer = regularizers.get(mixture_regularizer)
        self.bias_mixture_regularizer = regularizers.get(bias_mixture_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # self.dropout = min(1., max(0., dropout))
        # self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        # self.implementation = implementation
        # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
        # and fixed after 2.7.16. Converting the state_size to wrapper around
        # NoDependency(), so that the base_layer.__setattr__ will not convert it to
        # ListWrapper. Down the stream, self.states will be a list since it is
        # generated from nest.map_structure with list, and tuple(list) will work
        # properly.
        # Force batchs_size to one
        self.state_size = data_structures.NoDependency([
            self.units,  # h[t-1]
            self.units,  # c[t-1]
            self.window_size,  # window[t-1]
            self.mixtures//3,  # kappa[t-1]
            self.char_length+1,  # phi[t-1]
            tf.TensorShape((self.char_length, self.window_size)),  # sentence
        ])

        self.output_size = self.units + self.window_size

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.window_kernel = self.add_weight(
            shape=(self.window_size, self.units * 4),
            name='window_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.mixture_kernel = self.add_weight(
            shape=(self.units, self.mixtures),
            name='mixture_kernel',
            initializer=self.mixture_initializer,
            regularizer=self.mixture_regularizer,)

        def bias_initializer(_, *args, **kwargs):
            return K.concatenate([
                self.bias_initializer((self.units,), *args, **kwargs),
                initializers.Ones()((self.units,), *args, **kwargs),
                self.bias_initializer((self.units * 2,), *args, **kwargs),
                ])

        self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)

        self.bias_mixture = self.add_weight(
          shape=(self.mixtures,),
          name='bias_mixture',
          initializer='zeros',
          regularizer=self.bias_mixture_initializer,
          constraint=self.bias_mixture_regularizer)

        self.built = True

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def _compute_window(self, sentence, h, kappa_prev):
        h_hat = tf.matmul(h, self.mixture_kernel)
        h_hat = K.bias_add(h_hat, self.bias_mixture)
        alpha, beta, kappa = tf.split(h_hat, num_or_size_splits=3, axis=1)
        alpha = tf.math.exp(alpha)
        beta = tf.math.exp(beta)
        kappa = kappa_prev + tf.math.exp(kappa)
        phi = []
        # indexing starts at 1 in the paper
        U = tf.range(start=1, limit=self.char_length+1)
        U = tf.dtypes.cast(tf.stack([U]*10, axis=1), dtype=float)
        phi = alpha * tf.math.exp(- beta * (kappa - U)**2)
        phi = tf.reduce_sum(phi, axis=1, keepdims=True)
        w = tf.reduce_sum(sentence * phi, axis=1)
        phi = tf.reshape(phi, (1, self.char_length))
        # Heuristic described in paper phi(t, U+1) > phi(t, u) for 0<u<U+1
        last_phi = tf.reduce_sum(
            alpha * tf.math.exp(- beta * (kappa - self.char_length+1)**2),
            axis=1,)
        last_phi = tf.reshape(last_phi, (1, 1))
        phi = tf.concat((phi, last_phi), axis=-1)
        return w, kappa, phi

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        w_tm1 = states[2]  # previous window
        kappa_tm1 = states[3]
        # Hack to get pass the _standardize_args call in RNN
        # that forces to input to be a tensor (and not a list)
        sentence = states[-1]

        z = K.dot(inputs, self.kernel)
        z += K.dot(h_tm1, self.recurrent_kernel)
        z += K.dot(w_tm1, self.window_kernel)
        z = K.bias_add(z, self.bias)
        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)

        w, kappa, phi = self._compute_window(sentence, h, kappa_tm1)

        output = tf.concat((h, w), axis=-1)
        return output, [h, c, w, kappa, phi, sentence]

    def get_config(self):
        config = {
            'units':
                self.units,
            'window_size':
                self.window_size,
            'mixtures':
                self.mixtures,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'window_initializer':
                initializers.serialize(self.window_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'mixture_initializer':
                initializers.serialize(self.mixture_initializer),
            'bias_mixture_initializer':
                initializers.serialize(self.bias_mixture_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'window_initializer_regularizer':
                regularizers.serialize(self.window_initializer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'mixture_regularizer':
                regularizers.serialize(self.mixture_regularizer),
            'bias_mixture_regularizer':
                regularizers.serialize(self.bias_mixture_regularizer),
            # 'dropout':
            #     self.dropout,
            # 'recurrent_dropout':
            #     self.recurrent_dropout,
            # 'implementation':
            #     self.implementation
        }
        base_config = super(WindowedLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=1, dtype=None):
        return [tf.zeros(shape) for shape in self.state_size]
