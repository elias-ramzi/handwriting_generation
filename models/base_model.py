from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow_probability import edward2 as ed


class GaussianRegularizer(Regularizer):
    def __init__(self, mean=0., std=0.075):
        self.mean = tf.cast(mean, float)
        self.std = tf.cast(std, float)

    def __call__(self, x):
        if not self.mean and not self.std:
            return tf.constant(0.)
        regularization = tf.reduce_sum((
            tf.random.normal(x.shape, mean=self.mean, stddev=self.std)
            + x
        ))
        return regularization

    def get_config(self):
        return {'mean': float(self.mean), 'std': float(self.std)}


class BaseModel(ABC):

    def __init__(
        self,
        regularizer_type='l2',
        mean=0.,
        std=0.075,
        l2=0.01,
        lr=0.0001,
        rho=0.95,
        momentum=0.9,
        epsilon=0.0001,
        centered=True,
        inf_type='max',
        inf_n_jobs=-2,
        inf_backend='threading',
    ):
        assert regularizer_type in ['gaussian', 'l2'],\
            "Only supports l2 and gaussian regularization"
        self.regularizer_type = regularizer_type
        self.mean = mean
        self.std = std
        self.l2 = l2
        self.lr = lr
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered
        self.to_clip = []
        assert inf_type in ['sum', 'max'], "inf_type must be sum or max"
        self.inf_type = inf_type
        self.inf_n_jobs = inf_n_jobs
        self.inf_backend = inf_backend

    def regularization(self):
        if self.regularizer_type == 'gaussian':
            return GaussianRegularizer(self.mean, self.std)
        return tf.keras.regularizers.l2(self.l2)

    def debug(self, X):
        """
        Not elegant but usefull
        """
        import ipdb; ipdb.set_trace()
        return X

    @staticmethod
    def _Normal(x1, x2, mu1, mu2, sigma1, sigma2, rho):
        __x1 = x1 - mu1
        __x2 = x2 - mu2
        __sigma = sigma1*sigma2
        __rho = 1 - rho**2 + 10**(-4)
        Z = (
            __x1**2 / sigma1**2
            + __x2**2 / sigma2**2
            - (2*rho*__x1*__x2)/__sigma
        )
        det = (2*np.pi*__sigma*tf.math.sqrt(__rho)) + (10**(-7))
        return tf.math.exp(-Z / (2 * __rho)) / det

    @staticmethod
    def _mixture_coefs(y_hat):
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
        return tf.concat([
            e, pi, mu1, mu2,
            sigma1, sigma2, rho,
            ], axis=-1)

    @staticmethod
    def loss_function(y_true, y_pred):
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
        x1 = y_true[:, :, 0]
        x2 = y_true[:, :, 1]
        x3 = y_true[:, :, 2]
        x1, x2, x3 = tf.split(y_true, num_or_size_splits=3, axis=2)

        e = y_pred[:, :, 0]
        pi, mu1, mu2, sigma1, sigma2, rho = tf.split(
            y_pred[:, :, 1:],
            num_or_size_splits=6,
            axis=2,
        )

        # XXX: This uses the definition of the paper
        # in the data the stroke is indicated by the first
        # coordinate of X
        bernoulli = x3*tf.math.log(e + 10**(-4)) + (1-x3)*tf.math.log(1-e + 10**(-4))

        normal = pi * BaseModel._Normal(x1, x2, mu1, mu2, sigma1, sigma2, rho)
        normal = tf.math.log(tf.reduce_sum(normal, axis=2) + 10**(-7))

        loss = tf.math.reduce_sum(
            - normal
            - bernoulli,
            axis=1
        )
        loss = tf.reduce_mean(loss)  # / normal.shape[1]
        return loss

    def train(self, inputs, targets, load_weights=None):
        """
        adapted from
        https://github.com/tensorflow/tensorflow/issues/28707
        """
        if not hasattr(self, 'model'):
            self.make_model(load_weights=load_weights)

        optimizer = tf.keras.optimizers.RMSprop(
                lr=self.lr,
                rho=self.rho,
                momentum=self.momentum,
                epsilon=self.epsilon,
                centered=self.centered,
        )

        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            predictions = outputs[0]
            targets = tf.dtypes.cast(targets, dtype=float)
            loss = self.loss_function(targets, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

        # Clips gradient for output Dense layer
        gradients[-1] = tf.clip_by_value(gradients[-1], -100.0, 100.0)
        gradients[-2] = tf.clip_by_value(gradients[-2], -100.0, 100.0)

        # Clips gradient for LSTM layers
        for i, grad in enumerate(gradients[:-2]):
            gradients[i] = tf.clip_by_value(gradients[i], -10.0, 10.0)

        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def _infer(self, mixture_coefs, inf_type='max', bias=None):
        if inf_type is None:
            inf_type = self.inf_type
        e = mixture_coefs[0, 0, 0]
        pi, mu1, mu2, sigma1, sigma2, rho = tf.split(
            mixture_coefs[0, 0, 1:],
            num_or_size_splits=6,
            axis=0,
        )

        if bias is not None:
            sigma1 = tf.math.exp(sigma1 - bias)
            sigma2 = tf.math.exp(sigma2 - bias)
            pi = tf.nn.softmax(pi*(1+bias))

        end_stroke = 1 if np.random.rand() < e else 0

        if inf_type == 'sum':
            mu = tf.stack((mu1, mu2), axis=1)
            autocov = sigma1*sigma2*rho
            cov_matrix = tf.stack((
                tf.stack((sigma1, autocov), axis=1),
                tf.stack((autocov, sigma2), axis=1)),
                axis=1,
            )
            _offsets = (ed.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov_matrix)
                        * tf.stack((pi, pi), axis=1))
            offsets = tf.reduce_mean(_offsets, axis=0)
        else:
            idx_max = tf.argmax(pi)
            autocov = rho[idx_max] * sigma1[idx_max] * sigma2[idx_max]
            cov_matrix = [[sigma1[idx_max]**2, autocov], [autocov, sigma2[idx_max]**2]]
            offsets = ed.MultivariateNormalFullCovariance(
                loc=[mu1[idx_max], mu2[idx_max]],
                covariance_matrix=cov_matrix,
            )
        offsets = offsets.numpy()
        return end_stroke, offsets[0], offsets[1]
