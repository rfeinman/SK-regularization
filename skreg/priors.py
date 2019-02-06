import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold

from . import util


class Gaussian():
    def __init__(self, data_shape, Cov=None):
        # store input shape
        self.data_shape = data_shape
        # set mean
        self.mean = np.zeros(np.product(data_shape), dtype=np.float32)

        # set cov
        if Cov is None:
            self._fitted = False
        else:
            util.check_psd(Cov)
            self.Cov = Cov
            self._fitted = True

    def sample(self, N):
        """
        :param N: [int] number of samples to draw
        :return samples: [(N, data_shape) ndarray] filter samples
        """
        assert self._fitted
        samples = stats.multivariate_normal.rvs(self.mean, self.Cov, size=N)
        samples = samples.astype(np.float32)
        samples = np.reshape(samples, (N,)+self.data_shape)

        return samples

    def score(self, samples):
        """
        :param samples: [(N, data_shape) ndarray] samples to score log-prob
        :return scores: [(N,) ndarray] log-probabilities
        """
        self.check_samples(samples)
        assert self._fitted
        scores = np.zeros(len(samples), dtype=np.float32)
        for i,s in enumerate(samples):
            scores[i] = stats.multivariate_normal.logpdf(
                s.flatten(), mean=self.mean, cov=self.Cov
            )

        return scores

    def entropy(self):
        """
        :return entropy: [float] entropy of the Gaussian
        """
        assert self._fitted
        mvn = stats.multivariate_normal(mean=self.mean, cov=self.Cov)
        entropy = mvn.entropy()

        return entropy

    def save(self, path):
        """

        :param path:
        :return:
        """
        fname = path + '_Cov.npy'
        if os.path.isfile(fname):
            warnings.warn('Over-writing existing save file')
        np.save(fname, self.Cov.astype(np.float32))

    def check_samples(self, samples):
        if len(samples.shape) == 3:
            _, k1, k2 = samples.shape
        elif len(samples.shape) == 4:
            _, k1, k2, _ = samples.shape
        else:
            raise Exception
        assert k1 == k2


class CorrelatedGaussian(Gaussian):
    def __init__(self, data_shape):
        super(CorrelatedGaussian, self).__init__(data_shape)

    def fit(self, samples, alphas=np.logspace(-7,-1,100)):
        """
        :param samples:
        :param alphas:
        """
        self.check_samples(samples)
        D = samples.reshape(samples.shape[0], -1)
        # estimate covariance + Tikhonov regularization, using CV to
        # select mixing parameter
        Cov, results = tikhonov_cov(D, alphas=alphas)
        self._tikhonov = results[0][0]

        # check that covariance is PSD
        util.check_psd(Cov)
        self.Cov = Cov
        self._fitted = True

    @property
    def tikhonov(self):
        assert self._fitted, \
            "'tikhonov' property only applies to fitted Gaussians"
        return self._tikhonov

def tikhonov_cov(D, alphas, cv=3):
    m, n = D.shape
    results = {}
    for alpha in alphas:
        results[alpha] = tikhonov_trial(D, alpha, cv)
    results = sorted(results.items(), key=lambda x: -x[1])
    best_alpha = results[0][0]

    if best_alpha in [alphas[0], alphas[-1]]:
        warnings.warn("Boundary value %0.7f selected for 'alpha' Tikhonov "
                      "reg. parameter. Better result is possible." % best_alpha)
    Cov = util.cov(D) + best_alpha*np.eye(n)

    return Cov, results

def tikhonov_trial(D, alpha, cv=3):
    # store dimensionality
    m, n = D.shape
    assert m >= cv
    # loop through the folds
    scores = np.zeros(cv)
    kf = KFold(cv, shuffle=True, random_state=31)
    for i, (train, test) in enumerate(kf.split(D)):
        Cov = util.cov(D[train]) + alpha*np.eye(n)
        scores[i] = stats.multivariate_normal.logpdf(
            D[test], mean=np.zeros(n), cov=Cov
        ).mean()

    return scores.mean()


class IndependentGaussian(Gaussian):
    def __init__(self, data_shape, var=None):
        if var is not None:
            Cov = var*np.eye(np.product(data_shape), dtype=np.float32)
        else:
            Cov = None
        super(IndependentGaussian, self).__init__(data_shape, Cov)

    def fit(self, samples):
        """
        :param samples:
        :param vars:
        """
        self.check_samples(samples)
        var = samples.reshape(-1).var()
        Cov = var*np.eye(np.product(self.data_shape), dtype=np.float32)

        # check that covariance is PSD
        util.check_psd(Cov)
        self.Cov = Cov
        self._fitted = True
