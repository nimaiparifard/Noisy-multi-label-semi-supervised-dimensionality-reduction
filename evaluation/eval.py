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
    ap = average_precision_score(true_labels, predicted_labels.toarray(), average='macro')

    # One-error
    oe = np.mean(np.max(predicted_labels.toarray(), axis=1))

    # Coverage
    cov = np.mean(np.sum(predicted_labels.toarray(), axis=1))

    return hl, maf1, mif1, rl, ap, oe, cov

# # Assuming you have the true labels and predicted labels as binary matrices
# true_labels = np.array([[1, 0, 1, 0],
#                         [0, 1, 1, 0],
#                         [1, 1, 0, 1]])
#
# predicted_labels = np.array([[1, 0, 1, 0],
#                              [0, 1, 0, 0],
#                              [1, 0, 0, 1]])
#
# # Calculate evaluation metrics
# hl, maf1, mif1, rl, ap, oe, cov = evaluate_metrics(true_labels, predicted_labels)
#
# # Print the results
# print("Hamming Loss (HL):", hl)
# print("Macro F1-score (MaF1):", maf1)
# print("Micro F1 (MiF1):", mif1)
# print("Ranking Loss (RL):", rl)
# print("Average Precision (AP):", ap)
# print("One-error (OE):", oe)
# print("Coverage (Cov):", cov)