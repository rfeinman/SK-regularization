from __future__ import division, print_function
import numpy as np
import tensorflow as tf

from .util import check_psd


class GaussianReg(object):
    def __init__(self, alpha, Cov, rank):
        """
        :param alpha: [float]
        :param Cov: [(dim,dim) ndarray]
        :param rank: [int]
        """
        assert rank in [2,3]
        # make sure covariance matrix is positive semi-definite
        check_psd(Cov)

        # Gaussian dimensionality
        dim = Cov.shape[0]

        # normalize covariance so that det(Cov) = 1
        s, logdet = np.linalg.slogdet(Cov)
        assert s > 0
        if np.abs(logdet) > 1e-3:
            scale = np.exp(-logdet/dim)
            Cov = scale*Cov
        # precision matrix
        S = np.linalg.inv(Cov)

        self.rank = rank
        self.alpha = alpha
        self.S = tf.constant(S, dtype=tf.float32)

    def __call__(self, W):
        """
        :param W: [(k,k,c,n) tensor]
        :return cost: [() tensor]
        """
        k = tf.shape(W)[0]
        n = tf.shape(W)[3]
        if self.rank == 2:
            # (k,k,c,n) -> (k*k,c*n)
            W = tf.reshape(W, [k**2,-1])
            # (k*k,c*n) -> (c*n,k*k)
            W = tf.transpose(W)
        elif self.rank == 3:
            # (k,k,c,n) -> (n,k,k,c)
            W = tf.transpose(W, perm=[3,0,1,2])
            # (n,k,k,c) -> (n,k*k*c)
            W = tf.reshape(W, [n,-1])
        else:
            raise Exception
        losses = tf.matmul(W, tf.matmul(self.S, tf.transpose(W)))
        cost = self.alpha*tf.trace(losses)

        return cost
