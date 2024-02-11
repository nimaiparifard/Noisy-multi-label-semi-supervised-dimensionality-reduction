import numpy as np
from scipy.sparse import diags

def label_propagation(X, Y, alpha, sigma, max_iter=100, tol=1e-4):
    n, D = X.shape
    l, C = Y.shape
    u = n - l

    # Step 1: Construct neighborhood graph
    W = np.exp(-np.square(np.linalg.norm(X[:, np.newaxis] - X, axis=2)) / (2 * sigma**2))

    # Step 2: Symmetrically normalize adjacency matrix
    D = diags(np.sum(W, axis=1))
    D_sqrt_inv = diags(1 / np.sqrt(np.sum(W, axis=1)))
    W_hat = D_sqrt_inv @ W @ D_sqrt_inv

    # Step 3: Calculate stochastic matrix
    T = D_sqrt_inv @ W @ D_sqrt_inv

    # Step 4: Compute soft labels iteratively
    F = np.zeros((n, C))
    F[:l] = Y
    F_new = F.copy()

    for _ in range(max_iter):
        F = F_new.copy()
        F_new = alpha * T @ F + (1 - alpha) * Y

        if np.linalg.norm(F_new - F) < tol:
            break

    # Step 4 (continued): Convert soft labels to hard labels
    Y_hat = np.where(F_new > 0.5, 1, 0)

    return Y_hat

# Example usage
X = np.random.rand(100, 10)  # Input data points
Y = np.random.randint(0, 2, size=(80, 5))  # Known labels
alpha = 0.8  # Hyperparameter for label propagation
sigma = 0.5  # Hyperparameter for neighborhood graph

Y_hat = label_propagation(X, Y, alpha, sigma)