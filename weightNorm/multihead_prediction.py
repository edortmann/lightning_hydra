import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

import matplotlib.pyplot as plt
from datetime import datetime
from torchinfo import summary


# -------------------------------------------------------------------------------
# Dummy dataset creation
# -------------------------------------------------------------------------------
def create_dataset(num_samples=1000, input_dim=10, output_dim=1):
    """
    This function generates a simple random dataset where X is random values
    and y is the sum of those values, with some added noise.
    """
    X = torch.rand(num_samples, input_dim)
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, output_dim)
    return TensorDataset(X, y)


# -------------------------------------------------------------------------------
# Define the model class with the multi-layer sum head
# -------------------------------------------------------------------------------
class MultiLayerSumHeadModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MultiLayerSumHeadModel, self).__init__()
        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()

        # First layer and first head
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.heads.append(nn.Linear(hidden_dim, output_dim))

        # Additional layers and heads
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.heads.append(nn.Linear(hidden_dim, output_dim))

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Pass input through each layer, store the intermediate representations,
        and sum the outputs of each head to produce a final prediction.
        """
        hidden_representations = []
        for layer in self.layers:
            x = self.activation(layer(x))
            hidden_representations.append(x)

        # Compute prediction as sum of outputs from all heads
        outputs = [head(h) for head, h in zip(self.heads, hidden_representations)]
        return sum(outputs)


# -------------------------------------------------------------------------------
# Define a standard model that uses only the last layer for prediction
# -------------------------------------------------------------------------------
class StandardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StandardModel, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        head_dim = num_layers  # We expand to 'num_layers' dimension in the first head
        self.head1 = nn.Linear(hidden_dim, head_dim)
        self.head2 = nn.Linear(head_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Pass input through stacked layers, then through
        two final heads to get the output.
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.head1(x)
        x = self.head2(x)
        return x


# -------------------------------------------------------------------------------
# PyTorch Lightning modules
# -------------------------------------------------------------------------------
class MultiLayerSumHeadLightningModule(L.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        # Save hyperparameters for checkpointing/logging
        self.save_hyperparameters()

        # Instantiate the underlying nn.Module (the multi-head model).
        self.model = MultiLayerSumHeadModel(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            output_dim=model_config.output_dim
        )

        # MSE Loss for the regression
        self.criterion = nn.MSELoss()
        # Accuracy threshold from config
        self.accuracy_threshold = train_config.accuracy_threshold
        # Learning rate from config
        self.learning_rate = train_config.learning_rate

        # to access all batch outputs at the end of the epoch
        self.validation_step_outputs = []

        # Keep track of losses and accuracies every epoch (for plotting).
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        predictions = self(X_batch)
        loss = self.criterion(predictions, y_batch)

        # Calculate training accuracy (using absolute error < threshold)
        train_correct = torch.sum(torch.abs(predictions - y_batch) < self.accuracy_threshold).item()
        train_accuracy = train_correct / y_batch.size(0)

        # Log the loss and accuracy for progress bar or watchers
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        predictions = self(X_batch)
        loss = self.criterion(predictions, y_batch)

        # Calculate validation accuracy
        val_correct = torch.sum(torch.abs(predictions - y_batch) < self.accuracy_threshold).item()
        val_accuracy = val_correct / y_batch.size(0)

        output = {
            "val_loss": loss,
            "val_acc": torch.tensor(val_accuracy, device=self.device)
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to aggregate the average metrics.
        We append the results to lists so we can plot later.
        """
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().item()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean().item()
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # free up the memory
        self.validation_step_outputs.clear()


class StandardLightningModule(L.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate the underlying nn.Module (the standard model).
        self.model = StandardModel(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            output_dim=model_config.output_dim
        )

        self.criterion = nn.MSELoss()
        self.accuracy_threshold = train_config.accuracy_threshold
        self.learning_rate = train_config.learning_rate

        # to access all batch outputs at the end of the epoch
        self.validation_step_outputs = []

        # Track losses and accuracies for each epoch.
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        predictions = self(X_batch)
        loss = self.criterion(predictions, y_batch)

        train_correct = torch.sum(torch.abs(predictions - y_batch) < self.accuracy_threshold).item()
        train_accuracy = train_correct / y_batch.size(0)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        predictions = self(X_batch)
        loss = self.criterion(predictions, y_batch)

        val_correct = torch.sum(torch.abs(predictions - y_batch) < self.accuracy_threshold).item()
        val_accuracy = val_correct / y_batch.size(0)

        output = {
            "val_loss": loss,
            "val_acc": torch.tensor(val_accuracy, device=self.device)
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().item()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean().item()
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # free up the memory
        self.validation_step_outputs.clear()


# -------------------------------------------------------------------------------
# Analyzing norms of weights in the heads
# -------------------------------------------------------------------------------
def analyze_head_weights(model):
    """
    Analyze the norm of the weights in each head of the model.
    For the multi-head model, we analyze each head separately.
    For the standard model, we combine the two final linear layers
    (head1, head2) for a single norm.
    """
    # 'model' here is the underlying PyTorch nn.Module inside the LightningModule.
    if hasattr(model, 'heads'):
        # MultiLayerSumHeadModel logic
        norms = [torch.linalg.norm(head.weight).item() for head in model.heads]
    else:
        # StandardModel logic
        norms = [
            torch.linalg.norm(model.head1.weight).item()
            + torch.linalg.norm(model.head2.weight).item()
        ]
    return norms


# -------------------------------------------------------------------------------
# Plotting results
# -------------------------------------------------------------------------------
def plot_results(
    epochs,
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    sum_head_weight_norms,
    standard_weight_norms,
    num_layers,
    timestamp
):
    """
    Create and save plots of training/test loss, accuracy, and
    a bar chart comparing the norm of head weights.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(top=0.8)
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # We plot only test losses to declutter, as in original code.
    line2, = ax1.plot(epochs, test_losses[0], 'b--o', label="MultiHead Test Loss")
    line4, = ax1.plot(epochs, test_losses[1], 'r-o', label="Standard Test Loss")
    ax1.set_xlabel("Epoch", fontsize=BIGGER_SIZE)
    ax1.set_ylabel("Loss", color="k", fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.tick_params(axis='both', labelsize=MEDIUM_SIZE)

    # Create a second y-axis for accuracy.
    ax2 = ax1.twinx()
    line6, = ax2.plot(epochs, test_accuracies[0], 'g--x', label="MultiHead Test Acc")
    line8, = ax2.plot(epochs, test_accuracies[1], 'm-x', label="Standard Test Acc")
    ax2.set_ylabel("Accuracy", color="k")
    ax2.tick_params(axis='y', labelcolor="k")

    # Combine legends from both axes
    lines = [line2, line4, line6, line8]
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines, labels,
        loc='lower left',
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode='expand',
        borderaxespad=0,
        ncol=2
    )

    plt.tight_layout()
    plt.savefig(f"combined_loss_accuracy_{timestamp}.png")
    plt.show()

    # Second plot: comparing weight norms across heads
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    head_labels = [f"Head {i + 1}" for i in range(num_layers)]
    ax2.bar(head_labels, sum_head_weight_norms, label="MultiHead Norms", alpha=0.7)
    ax2.bar(["Head"], standard_weight_norms, label="Standard Model Norm", alpha=0.7)
    ax2.set_ylabel("Weight Norm")
    ax2.legend(loc="upper left")
    ax2.grid(axis='y', linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"weight_norms_comparison_{timestamp}.png")
    plt.show()


# -------------------------------------------------------------------------------
# DataModule for creating and splitting the dataset
# -------------------------------------------------------------------------------
class SumDatasetDataModule(L.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.dataset = None

    def setup(self, stage=None):
        """
        Called by Lightning to set up datasets for train/val/test splits.
        """
        # Use the same logic from the original code to create a dataset
        # and split into train/test.
        self.dataset = create_dataset(
            num_samples=self.data_config.num_samples,
            input_dim=self.data_config.input_dim,
            output_dim=self.data_config.output_dim
        )
        train_size = int(self.data_config.split_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.data_config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.data_config.batch_size, shuffle=False)


# -------------------------------------------------------------------------------
# The main function uses Hydra to parse config, sets up everything, and runs training.
# -------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare the data
    data_module = SumDatasetDataModule(cfg.data)
    data_module.setup()

    # -----------------------------
    # Initialize multi-head model
    # -----------------------------
    sum_head_module = MultiLayerSumHeadLightningModule(cfg.model, cfg.train)
    sum_head_module.to(device)

    # Print summary
    print("Summary of MultiLayerSumHeadModel:")
    summary(sum_head_module.model, input_size=(1, cfg.model.input_dim), device=str(device))

    print("\nTraining MultiLayerSumHeadModel:")
    trainer_sum_head = Trainer(max_epochs=cfg.train.epochs)
    trainer_sum_head.fit(sum_head_module, data_module)

    # Access final train/val metrics (for plotting)
    sum_head_train_losses = sum_head_module.train_losses
    sum_head_val_losses = sum_head_module.val_losses
    sum_head_train_accuracies = sum_head_module.train_accuracies
    sum_head_val_accuracies = sum_head_module.val_accuracies

    # Evaluate weight norms of the heads
    sum_head_weight_norms = analyze_head_weights(sum_head_module.model)
    print("Weight norms of the heads in MultiLayerSumHeadModel:", sum_head_weight_norms)

    # -----------------------------
    # Initialize standard model
    # -----------------------------
    standard_module = StandardLightningModule(cfg.model, cfg.train)
    standard_module.to(device)

    print("\nSummary of StandardModel:")
    summary(standard_module.model, input_size=(1, cfg.model.input_dim), device=str(device))

    print("\nTraining StandardModel:")
    trainer_standard = Trainer(max_epochs=cfg.train.epochs)
    trainer_standard.fit(standard_module, data_module)

    # Access final train/val metrics
    standard_train_losses = standard_module.train_losses
    standard_val_losses = standard_module.val_losses
    standard_train_accuracies = standard_module.train_accuracies
    standard_val_accuracies = standard_module.val_accuracies

    # Evaluate weight norms
    standard_weight_norms = analyze_head_weights(standard_module.model)
    print("Weight norm of the head in StandardModel:", standard_weight_norms)

    # ------------------------------------------------
    # Plotting results
    # ------------------------------------------------
    epochs_list = list(range(0, cfg.train.epochs + 1))
    plot_results(
        epochs_list,
        [sum_head_train_losses, standard_train_losses],
        [sum_head_val_losses, standard_val_losses],
        [sum_head_train_accuracies, standard_train_accuracies],
        [sum_head_val_accuracies, standard_val_accuracies],
        sum_head_weight_norms,
        standard_weight_norms,
        cfg.model.num_layers,
        timestamp
    )


if __name__ == "__main__":
    main()
