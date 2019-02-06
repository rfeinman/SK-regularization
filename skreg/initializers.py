from __future__ import division, print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.initializers import Initializer


class GaussianInit(Initializer):
    def __init__(self, Cov, rank):
        Cov = Cov.astype(np.float32)
        mu = np.zeros(Cov.shape[0], dtype=np.float32)
        self.mvn = tfp.distributions.MultivariateNormalFullCovariance(mu, Cov)
        self.rank = rank

    def __call__(self, shape, dtype=None):
        """
        shape: (k,k,c,m)
        """
        k,_,c,m = shape
        if self.rank == 2:
            W = self.mvn.sample((c,m)) # (c,m,k*k)
            W = tf.reshape(W, [c,m,k,k]) # (c,m,k*k) -> (c,m,k,k)
            W = tf.transpose(W, perm=[2,3,0,1]) # (c,m,k,k) -> (k,k,c,m)
        elif self.rank == 3:
            W = self.mvn.sample(m) # (m,k*k*c)
            W = tf.reshape(W, [m,k,k,c]) # (m,k*k*c) -> (m,k,k,c)
            W = tf.transpose(W, perm=[1,2,3,0]) # (m,k,k,c) -> (k,k,c,m)
        else:
            raise Exception

        return W