
import numpy as np
from sklearn.cluster import SpectralClustering
print("Importing libraries...")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
print("Importing libraries...")
def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    confusion_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        confusion_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    return sum(confusion_matrix[i, j] for i, j in zip(row_ind, col_ind)) / y_pred.size

def spectral_unsupervised_grid(features_path, labels_path, n_clusters=2):
    x = np.load(features_path)
    y = np.load(labels_path).flatten()

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    random_seed = 2648
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

    best_score = 0
    best_params = {}

    # Define hyperparameter grid
    neighbor_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assign_grid = ['kmeans', 'discretize']

    for n_neighbors in neighbor_grid:
        for assign_method in assign_grid:
            accuracies = []
            for train_idx, test_idx in kf.split(x, y):
                x_test = x[test_idx]
                y_test = y[test_idx]

                n_neighbors_safe = min(n_neighbors, len(x_test) - 1)
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=n_neighbors_safe,
                    assign_labels=assign_method,
                    random_state=random_seed
                )

                try:
                    y_pred = clustering.fit_predict(x_test)
                    acc = cluster_accuracy(y_test, y_pred)
                    accuracies.append(acc)
                except Exception as e:
                    print(f"Skipping config: neighbors={n_neighbors}, assign={assign_method}, error: {e}")
                    continue

            mean_acc = np.mean(accuracies)
            print(f"n_neighbors={n_neighbors}, assign_labels={assign_method}, accuracy={mean_acc:.4f}")
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = {
                    'n_neighbors': n_neighbors,
                    'assign_labels': assign_method
                }

    print(f"\nBest Accuracy: {best_score:.4f}")
    print(f"Best Params: {best_params}")
    return best_score, best_params


features_path =  "./your_features.npy"
labels_path =  "./your_label.npy"


# Run Spectral Clustering
mean_accuracy_spectral, best_params = spectral_unsupervised_grid(features_path, labels_path)
print(f"Mean Clustering Accuracy (Spectral): {mean_accuracy_spectral:.4f}")
print(f"Best Params: {best_params}")
