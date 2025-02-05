import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
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
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


# Function to calculate Frobenius norm of model parameters
def frobenius_norm(model):
    norm = 0
    for param in model.parameters():
        norm += torch.norm(param, p="fro") ** 2
    return torch.sqrt(norm).item()


class LitCNN(L.LightningModule):
    def __init__(self, weight_decay, learning_rate=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        self.train_acc(out, y)
        self.log_dict({'train_loss': loss, 'train_acc': self.train_acc}, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        frob_norm = frobenius_norm(self)
        self.log_dict({'frob_norm': frob_norm, 'margin': 1 / frob_norm})

    def on_train_end(self):
        frobenius_norm_value = frobenius_norm(self)
        print(f"Frobenius Norm of the model after training: {frobenius_norm_value:.4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        self.test_acc(out, y)
        self.log_dict({'test_loss': loss, 'test_acc': self.test_acc}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


# Run the experiment several times
def run_experiments(
    num_runs=1,
    output_file="results.csv",
    weight_decay=0.0,
    num_epochs=5,
    batch_size=64,
    learning_rate=2e-5,
):
    for run in range(num_runs):
        mnist = MNISTDataModule(batch_size=batch_size)
        model = LitCNN(weight_decay=weight_decay)

        trainer = L.Trainer(accelerator='gpu', max_epochs=num_epochs, default_root_dir=output_file.split('.csv')[0], enable_checkpointing=False)
        trainer.fit(model=model, datamodule=mnist)

        trainer.test(model=model, datamodule=mnist)

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
