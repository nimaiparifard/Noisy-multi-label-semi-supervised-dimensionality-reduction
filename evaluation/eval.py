import numpy as np
from sklearn.metrics import hamming_loss, f1_score, average_precision_score

def evaluate_metrics(true_labels, predicted_labels):
    # Hamming loss
    hl = hamming_loss(true_labels, predicted_labels)

    # Macro F1-score
    maf1 = f1_score(true_labels, predicted_labels, average='macro')

    # Micro F1-score
    mif1 = f1_score(true_labels, predicted_labels, average='micro')

    # Ranking loss
    rl = np.mean(np.sum(np.logical_xor(true_labels, predicted_labels), axis=1))

    # Average precision
    ap = average_precision_score(true_labels, predicted_labels, average='macro')

    # One-error
    oe = np.mean(np.max(predicted_labels, axis=1))

    # Coverage
    cov = np.mean(np.sum(predicted_labels, axis=1))

    return hl, maf1, mif1, rl, ap, oe, cov

