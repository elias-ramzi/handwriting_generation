from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer


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
        inf_type='sum',
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

    @abstractmethod
    def _make_model(self):
        raise NotImplementedError

    @staticmethod
    def _Normal(x1, x2, mu1, mu2, sigma1, sigma2, rho):
        __x1 = tf.stack([x1]*20, axis=-1) - mu1
        __x2 = tf.stack([x2]*20, axis=-1) - mu2
        __sigma = sigma1*sigma2
        __rho = 1 - rho**2 + 10**(-4)
        Z = (
            __x1**2 / sigma1**2
            + __x2**2 / sigma2**2
            - (2*rho*__x1*__x2)/__sigma
        )
        det = (2*np.pi*__sigma*tf.math.sqrt(__rho))
        return tf.math.exp(-Z / (2 * __rho)) / det

    @staticmethod
    def _mixture_coefs(y_hat):
        e = tf.nn.sigmoid(-y_hat[:, :, 0: 1])
        pi = tf.nn.softmax(y_hat[:, :, 1: 21])
        mu1 = y_hat[:, :, 21: 41]
        mu2 = y_hat[:, :, 41: 61]
        sigma1 = tf.math.exp(y_hat[:, :, 61: 81]) + 10**(-4)
        sigma2 = tf.math.exp(y_hat[:, :, 81: 101]) + 10**(-4)
        rho = tf.nn.tanh(y_hat[:, :, 101:121])
        return tf.concat([
            e, pi, mu1, mu2,
            sigma1, sigma2, rho,
            ], axis=-1)

    @staticmethod
    def loss_function(y_true, y_pred):
        x1 = y_true[:, :, 0]
        x2 = y_true[:, :, 1]
        x3 = y_true[:, :, 2]

        e = y_pred[:, :, 0]
        pi = y_pred[:, :, 1: 21]
        mu1 = y_pred[:, :, 21: 41]
        mu2 = y_pred[:, :, 41: 61]
        sigma1 = y_pred[:, :, 61: 81]
        sigma2 = y_pred[:, :, 81: 101]
        rho = y_pred[:, :, 101:121]

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
        return tf.reduce_mean(loss)  # / normal.shape[1]

    def train(self, strokes, targets, load_weights=None):
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
            predictions = self.model(strokes, training=True)
            loss = self.loss_function(targets, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

        # Clips gradient for output Dense layer
        gradients[-1] = tf.clip_by_value(gradients[-1], -100.0, 100.0)
        gradients[-2] = tf.clip_by_value(gradients[-2], -100.0, 100.0)

        # Clips gradient for LSTM layers
        for i, grad in enumerate(gradients):
            name = self.model.trainable_variables[i].name
            if name in self.to_clip:
                gradients[i] = tf.clip_by_value(gradients[i], -10.0, 10.0)

        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @staticmethod
    def _sample(
        pi=None,
        mu1=None, mu2=None,
        sigma1=None, sigma2=None,
        rho=None,
    ):
        _autocov = rho * sigma1 * sigma2
        cov_matrix = [[sigma1**2, _autocov], [_autocov, sigma2**2]]
        return np.random.multivariate_normal([mu1, mu2], cov_matrix, 1)

    def _infer(self, mixture_coefs, inf_type=None):
        if inf_type is None:
            inf_type = self.inf_type
        e = np.squeeze(mixture_coefs[:, :, :1])
        pi = np.squeeze(mixture_coefs[:, :, 1: 21])
        mu1 = np.squeeze(mixture_coefs[:, :, 21: 41])
        mu2 = np.squeeze(mixture_coefs[:, :, 41: 61])
        sigma1 = np.squeeze(mixture_coefs[:, :, 61: 81])
        sigma2 = np.squeeze(mixture_coefs[:, :, 81: 101])
        rho = np.squeeze(mixture_coefs[:, :, 101:121])

        end_stroke = 1 if np.random.rand() < e else 0

        if inf_type == 'sum':
            _offsets = Parallel(
                n_jobs=self.inf_n_jobs,
                backend=self.inf_backend,
            )(delayed(BaseModel._sample)(
                pi=pi[i],
                mu1=mu1[i], mu2=mu2[i],
                sigma1=sigma1[i], sigma2=sigma2[i],
                rho=rho[i],
            ) for i in range(20))
            offsets = np.sum(_offsets, axis=0)
        else:
            idx_max = pi.argmax()
            offsets = BaseModel._sample(
                pi=pi[idx_max],
                mu1=mu1[idx_max], mu2=mu2[idx_max],
                sigma1=sigma1[idx_max], sigma2=sigma2[idx_max],
                rho=rho[idx_max],
            )
        offsets = np.squeeze(offsets)
        return end_stroke, offsets[0], offsets[1]
