import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment


def cluster_accuracy(y_true, y_pred):
    """Compute clustering accuracy with Hungarian algorithm."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    confusion_matrix = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        confusion_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    total = sum([confusion_matrix[i, j] for i, j in zip(row_ind, col_ind)])
    return total / y_pred.size


def gmm_unsupervised(features_path, labels_path, n_components=2):
    # Load data
    x = np.load(features_path)
    y = np.load(labels_path).flatten()

    # Normalize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # random_seed = 2648
    random_seed = 2708

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
    accuracies = []

    for train_idx, test_idx in kf.split(x, y):
        x_test = x[test_idx]
        y_test = y[test_idx]

        # Fit GMM to test fold
        gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', n_init=10, random_state=random_seed)
        y_pred = gmm.fit_predict(x_test)

        acc = cluster_accuracy(y_test, y_pred)
        accuracies.append(acc)

    return np.mean(accuracies)


features_path =  "./your_features.npy"
labels_path =  "./your_label.npy"

# # 
# Run GMM clustering
mean_accuracy_gmm = gmm_unsupervised(features_path, labels_path)
print(f"Mean Clustering Accuracy (GMM): {mean_accuracy_gmm:.4f}")