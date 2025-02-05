import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# Define the model class with the multi-layer sum head
class MultiLayerSumHeadModel(nn.Module):
    def __init__(self, backbone, num_classes, hidden_dim, num_heads):
        super(MultiLayerSumHeadModel, self).__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_heads)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        with torch.no_grad():  # Freeze backbone during forward
            x = self.backbone(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        outputs = [head(x) for head in self.heads]
        return sum(outputs)

# Define a standard model that uses only the last layer for prediction
class StandardModel(nn.Module):
    def __init__(self, backbone, num_classes, hidden_dim):
        super(StandardModel, self).__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():  # Freeze backbone during forward
            x = self.backbone(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

# Training and evaluation functions
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=20):
    model.to(device)
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # Evaluate on test set
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(correct / total)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies

# Plot and save results
def plot_results(epochs, train_losses, test_losses, train_accuracies, test_accuracies, sum_head_weight_norms, standard_weight_norms, num_heads, timestamp):
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
    plt.savefig(f"CNN_loss_accuracy_comparison_{timestamp}.png")

    # Weight Norms Comparison
    plt.figure(figsize=(8, 6))
    plt.bar(
        [f"Head {i+1}" for i in range(num_heads)],
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

    plt.savefig(f"CNN_weight_norms_comparison_{timestamp}.png")


# Main script
if __name__ == "__main__":

    # Hyperparameters
    batch_size = 128
    epochs = 5
    learning_rate = 0.001
    num_classes = 10  # CIFAR-10 classes
    hidden_dim = 512  # Output dimension of ResNet backbone
    num_heads = 3

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading and transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ResNet input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # CIFAR-10 stats
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Backbone model
    resnet18 = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(resnet18.children())[:-2])

    # Initialize models
    sum_head_model = MultiLayerSumHeadModel(backbone, num_classes, hidden_dim, num_heads)
    standard_model = StandardModel(backbone, num_classes, hidden_dim)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_sum = optim.Adam(sum_head_model.heads.parameters(), lr=learning_rate)
    optimizer_standard = optim.Adam(standard_model.head.parameters(), lr=learning_rate)

    # Train and evaluate models
    print("Training MultiLayerSumHeadModel:")
    sum_head_train_losses, sum_head_test_losses, sum_head_train_accuracies, sum_head_test_accuracies = train_model(
        sum_head_model, train_loader, test_loader, criterion, optimizer_sum, device, epochs
    )

    print("\nTraining StandardModel:")
    standard_train_losses, standard_test_losses, standard_train_accuracies, standard_test_accuracies = train_model(
        standard_model, train_loader, test_loader, criterion, optimizer_standard, device, epochs
    )

    # Analyze head weights
    sum_head_weight_norms = [torch.norm(head.weight, p=2).item() for head in sum_head_model.heads]
    standard_weight_norm = [torch.norm(standard_model.head.weight, p=2).item()]

    print("Weight norms of the heads in MultiLayerSumHeadModel:", sum_head_weight_norms)
    print("Weight norm of the head in StandardModel:", standard_weight_norm)

    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results(
        list(range(1, epochs + 1)),
        [sum_head_train_losses, standard_train_losses],
        [sum_head_test_losses, standard_test_losses],
        [sum_head_train_accuracies, standard_train_accuracies],
        [sum_head_test_accuracies, standard_test_accuracies],
        sum_head_weight_norms,
        standard_weight_norm,
        num_heads,
        timestamp
    )