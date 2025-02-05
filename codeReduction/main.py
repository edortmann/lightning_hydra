import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import argparse


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to calculate Frobenius norm of model parameters
def frobenius_norm(model):
    norm = 0
    for param in model.parameters():
        norm += torch.norm(param, p="fro") ** 2
    return torch.sqrt(norm).item()


# Load the MNIST dataset
def data_creation():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="catdog/data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="catdog/data", train=False, transform=transform, download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


# Initialize the model, loss function, and optimizer
def initialize(device, weight_decay):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    return model, criterion, optimizer

def training(device, train_loader, model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        train_acc = 100 * correct / total
        print(f"Accuracy of the model on the train images: {train_acc:.2f}%")
    return train_acc


# Test the model (optional)
def testing(device, test_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        print(f"Accuracy of the model on the 10,000 test images: {test_acc:.2f}%")
    return test_acc


# Training loop
def main(device, train_loader, test_loader, model, criterion, optimizer, results, num_epochs):
    train_acc = training(device, train_loader, model, criterion, optimizer, num_epochs=num_epochs)
    # Calculate Frobenius norm after training
    frobenius_norm_value = frobenius_norm(model)
    print(f"Frobenius Norm of the model after training: {frobenius_norm_value:.4f}")
    test_acc = testing(device, test_loader, model)
    # Append results to the list
    results.append(
        {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "frobenius_norm": frobenius_norm_value,
            "margin": 1 / frobenius_norm_value,
            "train-test acc": train_acc - test_acc,
        }
    )


# Run the experiment several times and save results in a CSV file
def run_experiments(
    device,
    train_loader,
    test_loader,
    num_runs=1,
    output_file="results.csv",
    weight_decay=0.0,
    num_epochs= 5,
):
    results = []
    for run in range(num_runs):
        model, criterion, optimizer = initialize(device, weight_decay=weight_decay)
        main(device, train_loader, test_loader, model, criterion, optimizer, results, num_epochs=num_epochs)
        print(f"Run {run+1}/{num_runs} completed.")

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Run the main function with a specific weight decay
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--number_of_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results_hyper_3.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = data_creation()
    run_experiments(
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        num_runs=args.number_of_runs,
        output_file=args.output_dir,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs
    )
