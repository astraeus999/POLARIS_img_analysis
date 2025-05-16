import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def linearsvc_classify(features_path, labels_path, search=False):
    # Load features and labels
    x = np.load(features_path)
    y = np.load(labels_path).flatten()  # Ensure labels are 1D
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    best_params_list = []  # To store the best parameters for each fold

    for train_index, test_index in kf.split(x, y):
        # Split data into training and testing sets
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Perform hyperparameter search if enabled
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10,100]}
            classifier = GridSearchCV(LinearSVC(max_iter=10000), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=100, max_iter=10000)

        # Train the classifier
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)

        if search:
            # Save the best parameters for this fold
            best_params_list.append(classifier.best_params_)

        # Test accuracy
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # Validation accuracy
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if i not in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            classifier = GridSearchCV(LinearSVC(max_iter=10000), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10, max_iter=10000)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    if search:
        print("Best parameters for each fold:")
        for i, params in enumerate(best_params_list):
            print(f"Fold {i + 1}: {params}")

        # Find the most frequently selected parameter
        from collections import Counter
        param_counts = Counter(tuple(sorted(params.items())) for params in best_params_list)
        most_common_params = param_counts.most_common(1)[0]
        print(f"\nMost frequently selected parameter: {dict(most_common_params[0])}")
        print(f"Frequency: {most_common_params[1]} out of {len(best_params_list)} folds")

    # Return mean validation and test accuracies
    return np.mean(accuracies_val), np.mean(accuracies)


features_path =  "./your_features.npy"
labels_path =  "./your_label.npy"


# Perform linear SVC classification
mean_val_accuracy, mean_test_accuracy = linearsvc_classify(features_path, labels_path, search=True)
print(f"Mean Validation Accuracy: {mean_val_accuracy:.4f}")
print(f"Mean Test Accuracy: {mean_test_accuracy:.4f}")
