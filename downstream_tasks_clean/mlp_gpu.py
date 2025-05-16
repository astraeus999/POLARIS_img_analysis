import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def mlp_classify_gpu_with_search(features_path, labels_path, param_grid, epochs=50, batch_size=32):
    # Load features and labels
    x = np.load(features_path)
    y = np.load(labels_path).flatten()

    # Convert to PyTorch tensors and move to GPU if available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    # Stratified K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)
    best_params = None
    best_accuracy = 0

    # Iterate over all combinations of hyperparameters
    for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
        for learning_rate in param_grid['learning_rate']:
            accuracies = []

            for train_index, test_index in kf.split(x.cpu().numpy(), y.cpu().numpy()):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Define the model, loss function, and optimizer
                model = MLP(input_size=x.shape[1], hidden_layer_sizes=hidden_layer_sizes, output_size=len(torch.unique(y))).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(x_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    outputs = model(x_test)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
                    accuracies.append(accuracy)

            # Calculate mean accuracy for this parameter combination
            mean_accuracy = np.mean(accuracies)
            print(f"Params: hidden_layer_sizes={hidden_layer_sizes}, learning_rate={learning_rate}, Accuracy: {mean_accuracy:.4f}")

            # Update best parameters if this combination is better
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = {'hidden_layer_sizes': hidden_layer_sizes, 'learning_rate': learning_rate}

    print(f"Best Parameters: {best_params}, Best Accuracy: {best_accuracy:.4f}")
    return best_params, best_accuracy

features_path =  "./your_features.npy"
labels_path =  "./your_label.npy"


param_grid = {
    'hidden_layer_sizes': [(10,),(20,), (50,), (100,), ],
    'learning_rate': [0.001, 0.01, 0.1]
}

best_params, best_accuracy = mlp_classify_gpu_with_search(features_path, labels_path, param_grid, epochs=50, batch_size=32)
print(f"Best Parameters: {best_params}, Best Accuracy: {best_accuracy:.4f}")