import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import lightning as L
import torchmetrics
import hydra
from omegaconf import DictConfig
import os


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


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_experiments(cfg: DictConfig):
    # Retrieve configuration from Hydra
    num_runs = cfg.experiment.num_runs
    output_file = cfg.experiment.output_file
    weight_decay = cfg.model.weight_decay
    learning_rate = cfg.model.learning_rate
    batch_size = cfg.data.batch_size
    num_epochs = cfg.trainer.max_epochs
    data_dir = cfg.data.data_dir

    for run in range(num_runs):
        mnist = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
        model = LitCNN(weight_decay=weight_decay, learning_rate=learning_rate)

        trainer = L.Trainer(
            accelerator=cfg.trainer.accelerator,
            max_epochs=num_epochs,
            enable_checkpointing=False
        )
        trainer.fit(model=model, datamodule=mnist)
        trainer.test(model=model, datamodule=mnist)

        print(f"Run {run+1}/{num_runs} completed.")


if __name__ == "__main__":
    run_experiments()
