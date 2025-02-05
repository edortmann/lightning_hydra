import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Dummy dataset
def create_dataset(num_samples=1000, input_dim=10, output_dim=1):
    X = torch.rand(num_samples, input_dim)
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, output_dim)
    return TensorDataset(X, y)


# Define the model class with the multi-layer sum head
class MultiLayerSumHeadModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MultiLayerSumHeadModel, self).__init__()
        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.heads.append(nn.Linear(hidden_dim, output_dim))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.heads.append(nn.Linear(hidden_dim, output_dim))

        self.activation = nn.ReLU()

    def forward(self, x):
        hidden_representations = []
        for layer in self.layers:
            x = self.activation(layer(x))
            hidden_representations.append(x)

        # Compute prediction as sum of outputs from all heads
        outputs = [head(h) for head, h in zip(self.heads, hidden_representations)]
        return sum(outputs)


# Define a standard model that uses only the last layer for prediction
class StandardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StandardModel, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.head = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.head(x)


# Training setup
def train_model(model, train_loader, test_loader, epochs=20, learning_rate=0.001, accuracy_threshold=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

            # Calculate training accuracy
            train_correct += torch.sum(torch.abs(predictions - y_batch) < accuracy_threshold).item()
            train_total += y_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                test_loss += loss.item() * X_batch.size(0)

                # Calculate test accuracy
                test_correct += torch.sum(torch.abs(predictions - y_batch) < accuracy_threshold).item()
                test_total += y_batch.size(0)

        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies


# Analyzing norms of weights in the heads
def analyze_head_weights(model):
    if hasattr(model, 'heads'):
        norms = [torch.norm(head.weight, p=2).item() for head in model.heads]
    else:
        norms = [torch.norm(model.head.weight, p=2).item()]
    return norms


# Plot and save results
def plot_results(epochs, train_losses, test_losses, train_accuracies, test_accuracies, sum_head_weight_norms,
                 standard_weight_norms, num_layers, timestamp):
    plt.figure(figsize=(16, 12))

    # Train Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses[0], label="Sum Head Model Loss")
    plt.plot(epochs, train_losses[1], label="Standard Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
    plt.title("Train Loss Comparison")
    plt.legend()
    plt.xticks(epochs)

    # Test Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, test_losses[0], label="Sum Head Model Loss")
    plt.plot(epochs, test_losses[1], label="Standard Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Comparison")
    plt.legend()
    plt.xticks(epochs)

    # Train Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_accuracies[0], label="Sum Head Model Accuracy")
    plt.plot(epochs, train_accuracies[1], label="Standard Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy Comparison")
    plt.legend()
    plt.xticks(epochs)

    # Test Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, test_accuracies[0], label="Sum Head Model Accuracy")
    plt.plot(epochs, test_accuracies[1], label="Standard Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.xticks(epochs)

    plt.tight_layout()
    plt.savefig(f"loss_accuracy_comparison_{timestamp}.png")

    # Weight Norms Comparison
    plt.figure(figsize=(8, 6))
    plt.bar(
        [f"Head {i + 1}" for i in range(num_layers)],
        sum_head_weight_norms,
        label="Sum Head Model Norms",
        alpha=0.7
    )
    plt.bar(
        ["Standard Model"],
        standard_weight_norms,
        label="Standard Model Norm",
        alpha=0.7
    )
    plt.ylabel("Weight Norms")
    plt.title("Weight Norms Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')

    plt.savefig(f"weight_norms_comparison_{timestamp}.png")


# Main script
if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Hyperparameters
    input_dim = 10
    hidden_dim = 32
    num_layers = 4
    output_dim = 1
    batch_size = 64
    epochs = 20
    accuracy_threshold = 0.1

    # Dataset and dataloader
    dataset = create_dataset(input_dim=input_dim, output_dim=output_dim)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    sum_head_model = MultiLayerSumHeadModel(input_dim, hidden_dim, num_layers, output_dim)
    standard_model = StandardModel(input_dim, hidden_dim, num_layers, output_dim)

    # Train and evaluate the MultiLayerSumHeadModel
    print("Training MultiLayerSumHeadModel:")
    sum_head_train_losses, sum_head_test_losses, sum_head_train_accuracies, sum_head_test_accuracies = train_model(
        sum_head_model, train_loader, test_loader, epochs=epochs, accuracy_threshold=accuracy_threshold
    )

    sum_head_weight_norms = analyze_head_weights(sum_head_model)
    print("Weight norms of the heads in MultiLayerSumHeadModel:", sum_head_weight_norms)

    # Train and evaluate the StandardModel
    print("\nTraining StandardModel:")
    standard_train_losses, standard_test_losses, standard_train_accuracies, standard_test_accuracies = train_model(
        standard_model, train_loader, test_loader, epochs=epochs, accuracy_threshold=accuracy_threshold
    )

    standard_weight_norms = analyze_head_weights(standard_model)
    print("Weight norm of the head in StandardModel:", standard_weight_norms)

    # Plot results
    plot_results(
        list(range(1, epochs + 1)),
        [sum_head_train_losses, standard_train_losses],
        [sum_head_test_losses, standard_test_losses],
        [sum_head_train_accuracies, standard_train_accuracies],
        [sum_head_test_accuracies, standard_test_accuracies],
        sum_head_weight_norms,
        standard_weight_norms,
        num_layers,
        timestamp
    )
