import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import argparse
import lightning as L
from lightning.pytorch import loggers as pl_loggers
import torchmetrics


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def prepare_data(self):
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.train, self.val = random_split(mnist_full, [55000, 5000])
        self.test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
        self.train, self.val = random_split(cifar_full, [45000, 5000])
        self.test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class CIFARCatDogDataModule(L.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        cifar_full_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
        # Filter to only include 'cat' (label 3) and 'dog' (label 5)
        cat_dog_indices = [i for i, label in enumerate(cifar_full_train.targets) if label in [3, 5]]
        cifar10_cat_dog = torch.uitls.data.Subset(cifar_full_train, cat_dog_indices)
        self.train, self.val = random_split(cifar10_cat_dog, [45000, 5000])

        cifar_full_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
        cat_dog_indices = [i for i, label in enumerate(cifar_full_test.targets) if label in [3, 5]]
        self.test = torch.uitls.data.Subset(cifar_full_test, cat_dog_indices)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


# Function to calculate Frobenius norm of model parameters
def frobenius_norm(model):
    norm = 0
    for param in model.parameters():
        norm += torch.norm(param, p="fro") ** 2
    return torch.sqrt(norm).item()


class LitCNN(L.LightningModule):
    def __init__(self, weight_decay, input_channels=1, num_classes=10, optimizer_name='adam', lr=1e-3):
        super().__init__()

        # Model architecture
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Dynamically set the number of output classes

        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.train_acc(out, y)
        self.log_dict({'train_loss': loss, 'train_acc': self.train_acc}, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        frob_norm = frobenius_norm(self)
        self.log_dict({'frob_norm': frob_norm, 'margin': 1 / frob_norm})

    def on_train_end(self):
        frobenius_norm_value = frobenius_norm(self)
        print(f"Frobenius Norm of the model after training: {frobenius_norm_value:.4f}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.test_acc(out, y)
        self.log_dict({'test_loss': loss, 'test_acc': self.test_acc}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adamw':
            return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer_name}')


# Run the experiment several times
def run_experiments(
    num_runs=1,
    output_file="results.csv",
    weight_decay=0.0,
    num_epochs=5,
    batch_size=64,
    learning_rate=1e-4,
):
    for run in range(num_runs):

        # MNIST training
        dm_mnist = MNISTDataModule(batch_size=batch_size)
        model_mnist = LitCNN(weight_decay=weight_decay, input_channels=1, num_classes=10, optimizer_name='adam', lr=1e-3)  # MNIST: input_channels=1
        trainer = L.Trainer(accelerator='gpu', max_epochs=num_epochs, default_root_dir=output_file.split('.csv')[0], enable_checkpointing=False)
        trainer.fit(model_mnist, dm_mnist)
        trainer.test(model_mnist, dm_mnist)

        # CIFAR10 training
        dm_cifar = CIFAR10DataModule(batch_size=batch_size)
        model_cifar = LitCNN(weight_decay=weight_decay, input_channels=3, num_classes=10, optimizer_name='sgd', lr=1e-2)  # CIFAR10: input_channels=3
        trainer.fit(model_cifar, dm_cifar)
        trainer.test(model_cifar, dm_cifar)

        # CIFAR10 (only cat and dog) training
        dm_cifar_catdog = CIFARCatDogDataModule(batch_size=64)
        model_cifar_catdog = LitCNN(weight_decay=weight_decay, input_channels=3, num_classes=2, optimizer_name='adamw', lr=1e-4)
        trainer.fit(model_cifar_catdog, dm_cifar_catdog)
        trainer.test(model_cifar_catdog, dm_cifar_catdog)

        print(f"Run {run+1}/{num_runs} completed.")


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

    run_experiments(
        num_runs=args.number_of_runs,
        output_file=args.output_dir,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
