import numpy as np
import matplotlib.pyplot as plt
from Gaussian import GaussianMixtureModel

if __name__ == '__main__':
    data = np.loadtxt('dataset/data2D_a1.txt')
    log_likelihoods = []
    kmax = 10
    for k in range(1, kmax+1):
        gmm = GaussianMixtureModel(data, k, False)
        means, covs, pi, log_likelihood = gmm.fit()
        log_likelihoods.append(log_likelihood)
        print("k: ", k, " log_likelihood: ", log_likelihood)

    plt.plot(range(1, kmax+1), log_likelihoods)
    plt.xlabel('Number of components (k)')
    plt.ylabel('Converged log-likelihood')
    plt.show()

    # Final Number of components (k)
    k_star = np.argmax(log_likelihoods) + 1
    print(f'k* = {k_star}')
    finalGmm = GaussianMixtureModel(data, k_star, True)
    means, covs, pi, log_likelihood = finalGmm.fit()
    print(f'Final log-likelihood = {log_likelihood}')
    print(f'Final means = {means}')
    print(f'Final covs = {covs}')
    print(f'Final pi = {pi}')

