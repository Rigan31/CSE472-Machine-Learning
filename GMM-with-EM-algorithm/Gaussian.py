import time
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class GaussianMixtureModel:
    def __init__(self, data, k, isKFinal):
        self.data = data
        self.k = k
        self.isKFinal = isKFinal

    def fit(self):
        n, m = self.data.shape

        # Initialize means, covariances, and mixing coefficients
        pi = np.ones(self.k) / self.k
        means = self.data[np.random.choice(n, self.k, replace=False)]
        covs = [np.eye(m) for _ in range(self.k)]

        #print("shAEPE MEANS: ", means.shape)
        #print(means)

        # Initialize responsibilities
        log_likelihood = -np.inf

        # to draw for m = 2
        if m == 2 and self.isKFinal:
            plt.ion()
            fig = plt.figure()

        # EM algorithm
        for it in range(100):
            # E-step
            responsibilities = np.zeros((n, self.k))
            for i in range(self.k):
                responsibilities[:, i] = pi[i] * multivariate_normal.pdf(self.data, means[i], covs[i])
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step
            pi = responsibilities.sum(axis=0) / n
            means = np.dot(responsibilities.T, self.data) / responsibilities.sum(axis=0, keepdims=True).T
            covs = [np.dot(responsibilities[:, i] * (self.data - means[i]).T,
                           (self.data - means[i])) / responsibilities[:, i].sum() for i in range(self.k)]

            new_log_likelihood = 0

            for i in range(n):
                likelihood = 0
                for j in range(self.k):
                    likelihood += pi[j] * multivariate_normal.pdf(self.data[i], means[j], covs[j])
                new_log_likelihood += np.log(likelihood)

            # Check for convergence
            if np.abs(new_log_likelihood - log_likelihood) < 1e-6:
                break
            log_likelihood = new_log_likelihood

            if m == 2 and self.isKFinal:
                plt.clf()
                plt.scatter(self.data[:, 0], self.data[:, 1], color='#68BBE3')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title(f'Gaussian Mixture Model (k={self.k}) - Iteration {it + 1}')
                x, y = np.mgrid[self.data[:, 0].min():self.data[:, 0].max():.01,
                       self.data[:, 1].min():self.data[:, 1].max():.01]
                pos = np.empty(x.shape + (2,))
                pos[:, :, 0] = x
                pos[:, :, 1] = y
                for i in range(self.k):
                    rv = multivariate_normal(mean=means[i], cov=covs[i])
                    plt.contour(x, y, rv.pdf(pos))
                #clear previous drawing

                fig.canvas.draw()

                fig.canvas.flush_events()
                time.sleep(0.1)


        return means, covs, pi, log_likelihood

