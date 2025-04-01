import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import pytorch_lightning as L
from pytorch_lightning import Trainer

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torchinfo import summary
import math


# -------------------------------------------------------------------------------
# Multihead Model (Using Intermediate Features)
# -------------------------------------------------------------------------------
class CNNMultiLayerSumHeadModel(nn.Module):
    def __init__(self, backbone, backbone_feature_dims, hidden_dim, num_classes):
        """
        Args:
            backbone: A pretrained ResNet model (the full model so that intermediate layers are accessible).
            backbone_feature_dims (list of int): The number of channels (feature dimensions) at each intermediate layer.
                For example, for ResNet18: [64, 128, 256, 512] corresponding to layer1, layer2, layer3, layer4.
            hidden_dim: The size of the hidden representation (after mapping the pooled features).
            num_classes: Number of output classes.
        """
        super(CNNMultiLayerSumHeadModel, self).__init__()
        self.backbone = backbone  # Expecting a full ResNet model.
        # Freeze the backbone parameters.
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.ReLU()

        # Create a projection ("squeeze") layer and a corresponding head for each intermediate feature.
        self.squeeze_layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        for feature_dim in backbone_feature_dims:
            self.squeeze_layers.append(nn.Linear(feature_dim, hidden_dim))
            self.heads.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        """
        Forward pass that extracts intermediate features from the frozen backbone,
        then applies global pooling, a squeeze projection, and a classification head
        for each intermediate layer. The final output is the sum of these head outputs.
        """
        # Extract intermediate features from the backbone.
        # The backbone is frozen so we use torch.no_grad() to avoid tracking gradients.
        with torch.no_grad():
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            # Extract features from each of the four main block-layers.
            f1 = self.backbone.layer1(x)   # shape: [B, 64, H1, W1]
            f2 = self.backbone.layer2(f1)  # shape: [B, 128, H2, W2]
            f3 = self.backbone.layer3(f2)  # shape: [B, 256, H3, W3]
            f4 = self.backbone.layer4(f3)  # shape: [B, 512, H4, W4]

        features = [f1, f2, f3, f4]
        outputs = []

        # For each feature map, apply global pooling, a squeeze projection, then the classification head.
        for i, f in enumerate(features):
            pooled = self.global_pool(f)  # Shape becomes [B, C, 1, 1]
            pooled = pooled.view(pooled.size(0), -1)  # Flatten to [B, C]
            hidden = self.activation(self.squeeze_layers[i](pooled))  # Map to hidden_dim and activate.
            head_out = self.heads[i](hidden)  # Compute the head’s logits.
            outputs.append(head_out)

        # Sum the outputs from all heads.
        return sum(outputs)


# -------------------------------------------------------------------------------
# Standard Model (Single Head, Scaled Up)
# -------------------------------------------------------------------------------
class CNNStandardModel(nn.Module):
    def __init__(self, backbone, backbone_out_dim, hidden_dim_std, num_classes):
        """
        Args:
            backbone: A full pretrained ResNet model.
            backbone_out_dim: The channel dimension of the backbone's final output (e.g., 512 for ResNet18).
            hidden_dim_std: The hidden dimension used in the projection layer.
                (Chosen so that the total parameter count of the new layers roughly matches the multihead model.)
            num_classes: Number of output classes.
        """
        super(CNNStandardModel, self).__init__()
        self.backbone = backbone
        # Freeze the backbone parameters.
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.ReLU()

        # A single projection from the backbone's final feature (512 for ResNet18) to a larger hidden space.
        self.proj = nn.Linear(backbone_out_dim, hidden_dim_std)
        # A single prediction head.
        self.head = nn.Linear(hidden_dim_std, num_classes)

    def forward(self, x):
        """
        Forward pass that runs the (frozen) backbone, applies global pooling,
        then passes through a single projection layer and a single classifier head.
        """
        with torch.no_grad():
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.proj(x))
        return self.head(x)


# -------------------------------------------------------------------------------
# PyTorch Lightning modules to encapsulate training/validation logic
# -------------------------------------------------------------------------------
class CNNMultiLayerSumHeadLightningModule(L.LightningModule):
    def __init__(self, model_config, train_config, backbone):
        super().__init__()
        self.save_hyperparameters()

        # Build the underlying nn.Module (multi-head).
        self.model = CNNMultiLayerSumHeadModel(
            backbone=backbone,
            backbone_feature_dims=model_config.backbone_feature_dims,
            hidden_dim=model_config.hidden_dim_multi,
            num_classes=model_config.num_classes
        )

        # Cross-entropy loss for classification
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = train_config.learning_rate

        # We track training/validation losses & accuracies for plotting.
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        We only optimize the newly introduced layers (the heads & squeeze layers) because the backbone is frozen.
        """
        params = list(self.model.squeeze_layers.parameters()) + list(self.model.heads.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)

        # Save for end-of-epoch
        self.training_step_outputs.append((loss.item(), accuracy))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """
        Aggregate the training outputs and clear.
        """
        avg_loss = np.mean([x[0] for x in self.training_step_outputs])
        avg_acc = np.mean([x[1] for x in self.training_step_outputs])

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)

        out_dict = {"val_loss": loss.item(), "val_acc": accuracy}
        self.validation_step_outputs.append(out_dict)
        return out_dict

    def on_validation_epoch_end(self):
        """
        Aggregate the validation outputs and clear.
        """
        avg_loss = np.mean([x["val_loss"] for x in self.validation_step_outputs])
        avg_acc = np.mean([x["val_acc"] for x in self.validation_step_outputs])

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)
        self.validation_step_outputs.clear()


class CNNStandardLightningModule(L.LightningModule):
    def __init__(self, model_config, train_config, backbone):
        """
        LightningModule for the single-head CNN model.
        """
        super().__init__()
        self.save_hyperparameters()

        # Build the underlying nn.Module (standard single-head).
        self.model = CNNStandardModel(
            backbone=backbone,
            backbone_out_dim=model_config.backbone_out_dim,
            hidden_dim_std=model_config.hidden_dim_std,
            num_classes=model_config.num_classes
        )

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = train_config.learning_rate

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        Only optimize the newly added projection + head parameters,
        since the backbone is frozen.
        """
        params = list(self.model.proj.parameters()) + list(self.model.head.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)

        self.training_step_outputs.append((loss.item(), accuracy))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = np.mean([x[0] for x in self.training_step_outputs])
        avg_acc = np.mean([x[1] for x in self.training_step_outputs])
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)

        out_dict = {"val_loss": loss.item(), "val_acc": accuracy}
        self.validation_step_outputs.append(out_dict)
        return out_dict

    def on_validation_epoch_end(self):
        avg_loss = np.mean([x["val_loss"] for x in self.validation_step_outputs])
        avg_acc = np.mean([x["val_acc"] for x in self.validation_step_outputs])
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)
        self.validation_step_outputs.clear()


# -------------------------------------------------------------------------------
# Weight norm analysis
# -------------------------------------------------------------------------------
def analyze_weight_norms_multihead(model):
    """
    Multihead: Return a list of L2-norms for each head's weight.
    """
    return [torch.linalg.norm(head.weight).item() for head in model.heads]

def analyze_weight_norm_standard(model):
    """
    Standard: Return a single L2-norm for the final head's weight.
    """
    return torch.linalg.norm(model.head.weight).item()


# -------------------------------------------------------------------------------
# Plotting results
# -------------------------------------------------------------------------------
def plot_results_combined(
    epochs,
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    sum_head_weight_norms,
    standard_weight_norms,
    num_heads,
    timestamp
):
    """
    Plots train/test loss and accuracy in a single figure,
    plus a separate figure comparing the head weight norms.
    """
    fig, ax1 = plt.subplots(figsize=(10, 8))

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

    # Plot test (val) losses on the left y-axis.
    line2, = ax1.plot(epochs, test_losses[0], 'b--o', label="MultiHead Val Loss")
    line4, = ax1.plot(epochs, test_losses[1], 'r-o', label="SingleHead Val Loss")
    ax1.set_xlabel("Epoch", fontsize=BIGGER_SIZE)
    ax1.set_ylabel("Loss", color="k", fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.tick_params(axis='both', labelsize=MEDIUM_SIZE)

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    line6, = ax2.plot(epochs, test_accuracies[0], 'g--x', label="MultiHead Val Acc")
    line8, = ax2.plot(epochs, test_accuracies[1], 'm-x', label="SingleHead Val Acc")
    ax2.set_ylabel("Accuracy", color="k")
    ax2.tick_params(axis='y', labelcolor="k")

    # Combine legends
    lines = [line2, line4, line6, line8]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
               mode='expand', borderaxespad=0, ncol=2)

    plt.tight_layout()
    plt.savefig(f"cnn_multihead_combined_loss_accuracy_{timestamp}.png")
    plt.show()

    # Create a separate figure for the weight norm comparison.
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    plt.subplots_adjust(top=0.8)
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

    head_labels = [f"Head {i + 1}" for i in range(num_heads)]
    ax2.bar(head_labels, sum_head_weight_norms, label="MultiHead Weight Norms", alpha=0.7)
    ax2.bar(["Head"], standard_weight_norms, label="SingleHead Weight Norm", alpha=0.7)
    ax2.set_ylabel("Weight Norm")
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0)
    ax2.grid(axis='y', linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"cnn_multihead_weight_norms_comparison_{timestamp}.png")
    plt.show()


def plot_normalized_weight_norms(multihead_norms_normalized, std_norm_normalized, timestamp):
    """
    Plot the normalized weight norms (RMS) for the multi-head model vs single-head.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.subplots_adjust(top=0.8)
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

    head_labels = [f"Head {i + 1}" for i in range(len(multihead_norms_normalized))]
    ax.bar(head_labels, multihead_norms_normalized, label="MultiHead Normalized Norm", alpha=0.7)
    ax.bar(["Head"], [std_norm_normalized], label="SingleHead Normalized Norm", alpha=0.7)
    ax.set_ylabel("Normalized Weight Norm (RMS)")
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0)
    ax.grid(axis='y', linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"cnn_multihead_normalized_weight_norms_comparison_{timestamp}.png")
    plt.show()


# -------------------------------------------------------------------------------
# LightningDataModule for loading CIFAR-10 with the specified transforms
# -------------------------------------------------------------------------------
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_config, train_config):
        super().__init__()
        self.data_config = data_config
        self.train_config = train_config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for ResNet input
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),  # CIFAR-10 stats
        ])

        # Download/load dataset
        train_dataset = datasets.CIFAR10(
            root=self.data_config.data_dir,
            train=True,
            transform=transform,
            download=True
        )
        val_dataset = datasets.CIFAR10(
            root=self.data_config.data_dir,
            train=False,
            transform=transform,
            download=True
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.data_config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.data_config.batch_size, shuffle=False)


# -------------------------------------------------------------------------------
# Main script with Hydra
# -------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="cnn_config")
def main(cfg: DictConfig):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Hyperparameters from config
    batch_size = cfg.data.batch_size
    epochs = cfg.train.epochs
    learning_rate = cfg.train.learning_rate
    num_classes = cfg.model.num_classes
    hidden_dim_multi = cfg.model.hidden_dim_multi
    hidden_dim_std = cfg.model.hidden_dim_std
    backbone_feature_dims = cfg.model.backbone_feature_dims
    backbone_out_dim = cfg.model.backbone_out_dim
    num_layers = cfg.model.num_layers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataModule
    data_module = CIFAR10DataModule(cfg.data, cfg.train)
    data_module.setup()

    # Load a pretrained ResNet backbone (e.g. resnet18).
    # Keep it "full" so we can manually access conv1, layer1, etc.
    if cfg.model.backbone_name == "resnet18":
        backbone = models.resnet18(pretrained=True)
    else:
        raise ValueError("This script currently only supports resnet18 as the backbone.")

    #######################
    # Multihead Model
    #######################
    print("\nInitializing MultiLayerSumHeadModel.")
    multihead_module = CNNMultiLayerSumHeadLightningModule(cfg.model, cfg.train, backbone)
    multihead_module.to(device)

    # Summaries
    print("Multihead Model Summary:")
    summary(
        multihead_module.model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        device=str(device)
    )

    print("\nTraining MultiLayerSumHeadModel:")
    trainer_sum_head = Trainer(max_epochs=epochs)
    trainer_sum_head.fit(multihead_module, data_module)

    # Access final train/val metrics
    sum_head_train_losses = multihead_module.train_losses
    sum_head_test_losses = multihead_module.val_losses
    sum_head_train_accuracies = multihead_module.train_accuracies
    sum_head_test_accuracies = multihead_module.val_accuracies

    # Analyze weight norms for the multi-head
    multihead_weight_norms = analyze_weight_norms_multihead(multihead_module.model)
    print("Multihead heads weight norms:", multihead_weight_norms)

    #######################
    # Standard Model
    #######################
    print("\nInitializing StandardModel.")
    standard_module = CNNStandardLightningModule(cfg.model, cfg.train, backbone)
    standard_module.to(device)

    print("Standard Model Summary:")
    summary(
        standard_module.model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        device=str(device)
    )

    print("\nTraining StandardModel:")
    trainer_standard = Trainer(max_epochs=epochs)
    trainer_standard.fit(standard_module, data_module)

    # Access final train/val metrics
    standard_train_losses = standard_module.train_losses
    standard_test_losses = standard_module.val_losses
    standard_train_accuracies = standard_module.train_accuracies
    standard_test_accuracies = standard_module.val_accuracies

    # Analyze weight norms for single-head
    standard_weight_norm = analyze_weight_norm_standard(standard_module.model)
    print("Standard model head weight norm:", standard_weight_norm)

    # ----------------------------------------------------------------------------
    # Plot combined (loss & accuracy) + weight norm bar chart
    # ----------------------------------------------------------------------------
    epochs_list = list(range(0, epochs + 1))
    plot_results_combined(
        epochs_list,
        [sum_head_train_losses, standard_train_losses],
        [sum_head_test_losses, standard_test_losses],
        [sum_head_train_accuracies, standard_train_accuracies],
        [sum_head_test_accuracies, standard_test_accuracies],
        multihead_weight_norms,
        standard_weight_norm,
        num_layers,
        timestamp
    )

    ##########################################
    # Compute normalized weight norms (RMS)
    ##########################################
    multihead_norms_normalized = []
    for head in multihead_module.model.heads:
        weight = head.weight.data
        num_params = weight.numel()
        normalized_norm = torch.linalg.norm(weight).item() / math.sqrt(num_params)
        multihead_norms_normalized.append(normalized_norm)

    std_weight = standard_module.model.head.weight.data
    std_norm_normalized = torch.linalg.norm(std_weight).item() / math.sqrt(std_weight.numel())

    print("Multihead heads normalized weight norms:", multihead_norms_normalized)
    print("Standard model normalized weight norm:", std_norm_normalized)

    # Plot the normalized weight norm comparison
    plot_normalized_weight_norms(multihead_norms_normalized, std_norm_normalized, timestamp)


if __name__ == "__main__":
    main()
