import numpy as np
from scipy.linalg import eigh


def dependence_maximization(X, F_tilde, d):
    n, D = X.shape
    # Step 1: Construct matrix XTHFFTHX
    H = np.eye(n) - np.ones((n, n)) / n
    K = X @ X.T
    L = F_tilde @ F_tilde.T
    M = X.T @ H @ L @ H @ X

    # Step 2: Eigendecompose XTH˜F˜FTHX
    _, P = eigh(M, eigvals=(D - d, D - 1))

    return P

# y_tilda = np.load("Y_tilda.npy")
# X = np.load("train_dataset.npy")
# P = dependence_maximization(X, y_tilda, 20)
# np.save("P.npy", P)