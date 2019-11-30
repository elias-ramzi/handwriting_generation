from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed


class BaseModel(ABC):

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

    def _infer(self, mixture_coefs, bias=None):
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

        # autocov = sigma1*sigma2*rho
        # __sigma1 = sigma1**2
        # __sigma2 = sigma2**2
        #
        # pis = tfd.Categorical(pi)
        # components = [
        #     tfd.MultivariateNormalFullCovariance(
        #         loc=[mu1[k], mu2[k]],
        #         covariance_matrix=[[__sigma1[k], autocov[k]], [autocov[k], __sigma2[k]]]
        #     )
        #     for k in range(len(pi))
        # ]
        # x1, x2 = tfd.Mixture(cat=pis, components=components).sample().numpy()

        idx_max = tf.argmax(pi)
        autocov = rho[idx_max] * sigma1[idx_max] * sigma2[idx_max]
        cov_matrix = [[sigma1[idx_max]**2, autocov], [autocov, sigma2[idx_max]**2]]
        offsets = ed.MultivariateNormalFullCovariance(
            loc=[mu1[idx_max], mu2[idx_max]],
            covariance_matrix=cov_matrix,
        )
        x1, x2 = offsets.numpy()

        return end_stroke, x1, x2
