import numpy as np
from scipy.sparse import diags

def calculate_adjacency_matrix(X, sigma):
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i][j] = np.exp(-np.square(np.linalg.norm(X[i] - X[j], axis=0)) / sigma ** 2)
    return W

def label_propagation(X, Y, l, alpha, sigma):
    n, D = X.shape
    C = Y.shape[1]
    u = n - l

    # Step 1: Construct neighborhood graph
    W = calculate_adjacency_matrix(X, sigma)

    # Step 2: Symmetrically normalize adjacency matrix
    D = diags(np.sum(W, axis=1))
    D_sqrt_inv = diags(1 / np.sqrt(np.sum(W, axis=1)))
    W_hat = D_sqrt_inv @ W @ D_sqrt_inv

    # Step 3: Calculate stochastic matrix
    T = D_sqrt_inv @ W_hat
    I = np.identity(n=T.shape[0])
    # Step 4: Compute soft labels iteratively
    # F = np.zeros((n, C))
    F = np.linalg.inv((I - alpha * T)) @ (I - alpha * I) @ Y
    # Step 4 (continued): Convert soft labels to hard labels
    Y_tilda = np.zeros((n, C))
    Y_tilda[:l] = Y[:l]
    Y_tilda[l:] = np.where(F[l:] > 0.5, 1, F[l:])
    return Y_tilda, F


from dataset.datasets import create_dataset, make_labels_to_semi_supervised_task

train_dataset, train_labels, test_dataset, test_labels = create_dataset()
semi_supervised_labels, l = make_labels_to_semi_supervised_task(train_labels, 0.3)
Y_, F = label_propagation(train_dataset, semi_supervised_labels, l, alpha=0.6, sigma=1)

np.save('Y_tilda.npy', Y_)
np.save('train_dataset.npy', train_dataset)
np.save('train_labels.npy', train_labels)
np.save('test_dataset.npy', test_dataset)
np.save('test_labels.npy', test_labels)