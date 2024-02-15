import numpy as np
import pandas as pd
#%%
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

def create_dataset():
    # Dataset parameters
    num_samples = 8000
    num_features = 320
    num_classes = 4
    num_samples_per_class = num_samples // num_classes

    # Generate dataset using make_multilabel_classification
    X, y = make_multilabel_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_classes=num_classes,
        random_state=42
    )

    # Inject noise into labels
    noise_fraction = 0.1
    num_noisy_labels = int(noise_fraction * num_samples)
    noisy_label_indices = np.random.choice(num_samples, num_noisy_labels, replace=False)
    noisy_class_indices = np.random.choice(num_classes, num_noisy_labels, replace=True)
    y[noisy_label_indices, noisy_class_indices] = 1 - y[noisy_label_indices, noisy_class_indices]

    # Split into train and test sets
    test_fraction = 0.2
    num_test_samples = int(test_fraction * num_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test_samples, random_state=42)

    return X_train, y_train, X_test, y_test


def make_labels_to_semi_supervised_task(labels, percentage):
    # Set semi supervised labels to zero
    semi_supervised_labels = np.zeros_like(labels, dtype=np.float16)
    semi_supervised_labels[:int(labels.shape[0] * percentage)] = labels[:int(labels.shape[0] * percentage)]
    return semi_supervised_labels, int(labels.shape[0] * percentage)