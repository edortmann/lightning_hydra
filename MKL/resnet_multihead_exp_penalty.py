import os
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import lightning as L
import hydra
from omegaconf import OmegaConf, DictConfig


# -----------------------------------------------------------------------------
# Fusion module
# -----------------------------------------------------------------------------
class UnitVarianceBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine=True)
        self.s = 1. / math.sqrt(num_features)

    def forward(self, x):
        return self.bn(x) * self.s


class FCFusion(nn.Module):
    def __init__(self, in_features: list, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList([
            nn.Sequential(
                UnitVarianceBatchNorm(features),
                nn.LeakyReLU(inplace=True),
                nn.Linear(in_features=features, out_features=out_features, bias=bias)
            ) for features in in_features
        ])

    def forward(self, inputs):
        # 'inputs' is a list of tensors (one per head)
        return sum(layer(x) for layer, x in zip(self.layers, inputs))

    @property
    def sub_weights(self):
        # Return the weights of the final linear layers in each branch
        return [layer[2].weight for layer in self.layers]


# -----------------------------------------------------------------------------
# Multihead Model with FCFusion
# -----------------------------------------------------------------------------
class CNNMultiLayerFusionModel(nn.Module):
    def __init__(self, backbone, backbone_feature_dims, hidden_dim, num_classes):
        """
        Args:
            backbone: A pretrained ResNet model.
            backbone_feature_dims (list of int): Feature dimensions for each intermediate layer.
            hidden_dim: Dimension after the squeeze (projection) layers.
            num_classes: Number of output classes.
        """
        super(CNNMultiLayerFusionModel, self).__init__()
        self.backbone = backbone
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.ReLU()
        # Squeeze layers: map each intermediate feature to hidden_dim
        self.squeeze_layers = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in backbone_feature_dims
        ])
        # Fusion layer using FCFusion: it takes a list of vectors of length hidden_dim
        self.fusion = FCFusion(in_features=[hidden_dim] * len(backbone_feature_dims),
                               out_features=num_classes,
                               bias=True)

    def forward(self, x):
        # Extract intermediate features from the frozen backbone
        with torch.no_grad():
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            f1 = self.backbone.layer1(x)
            f2 = self.backbone.layer2(f1)
            f3 = self.backbone.layer3(f2)
            f4 = self.backbone.layer4(f3)
        features = [f1, f2, f3, f4]
        hidden_list = []
        for i, f in enumerate(features):
            pooled = self.global_pool(f)
            pooled = pooled.view(pooled.size(0), -1)
            hidden = self.activation(self.squeeze_layers[i](pooled))
            hidden_list.append(hidden)
        # Fuse the hidden representations from all heads
        out = self.fusion(hidden_list)
        return out


# -----------------------------------------------------------------------------
# Standard Model (Single Head, Scaled Up)
# -----------------------------------------------------------------------------
class CNNStandardModel(nn.Module):
    def __init__(self, backbone, backbone_out_dim, hidden_dim_std, num_classes):
        super(CNNStandardModel, self).__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.ReLU()
        self.proj = nn.Linear(backbone_out_dim, hidden_dim_std)
        self.head = nn.Linear(hidden_dim_std, num_classes)

    def forward(self, x):
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


# -----------------------------------------------------------------------------
# PyTorch Lightning Modules for training and evaluation
# -----------------------------------------------------------------------------
class LitMultiHeadClassifier(L.LightningModule):
    def __init__(self, model, criterion, learning_rate, momentum, exp_reg_rate):
        """
        Args:
            model: The multihead model.
            criterion: Loss function.
            learning_rate: Learning rate.
            momentum: Momentum for SGD.
            exp_reg_rate: Regularization rate for the exponential penalty.
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.exp_reg_rate = exp_reg_rate

        # to access all batch outputs at the end of the epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Lists to record metrics per epoch (for plotting later)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        # List to track the Frobenius norm of each head per epoch
        self.epoch_head_norms = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        # ----- Exponential Penalty: penalize head weights more for further layers -----
        penalty = 0.0
        # Iterate over heads in the fusion layer and add a weighted penalty.
        for i, w in enumerate(self.model.fusion.sub_weights):
            penalty += self.exp_reg_rate * math.exp(i) * torch.linalg.norm(w, ord='fro')
        loss = loss + penalty
        # ---------------------------------------------------------------------------------
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == targets).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        output = {'loss': loss, 'acc': acc}
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean().item()
        avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean().item()
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)
        # ----- Tracking Frobenius Norms for each head -----
        self.model.eval()
        with torch.no_grad():
            head_norms = [torch.linalg.norm(w, ord='fro').item() for w in self.model.fusion.sub_weights]
        self.epoch_head_norms.append(head_norms)
        self.model.train()
        # ------------------------------------------------------
        # free up the memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == targets).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        output = {'val_loss': loss, 'val_acc': acc}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().item()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean().item()
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # free up the memory
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.SGD([
            {'params': self.model.squeeze_layers.parameters()},
            {'params': self.model.fusion.parameters(), 'weight_decay': 0.0}
        ], lr=self.learning_rate, momentum=self.momentum)
        return optimizer


class LitStandardClassifier(L.LightningModule):
    def __init__(self, model, criterion, learning_rate):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate

        # to access all batch outputs at the end of the epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Lists to record metrics per epoch (for plotting later)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == targets).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        output = {'loss': loss, 'acc': acc}
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean().item()
        avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean().item()
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)

        # free up the memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == targets).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        output = {'val_loss': loss, 'val_acc': acc}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().item()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean().item()
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # free up the memory
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.model.proj.parameters()) + list(self.model.head.parameters()),
            lr=self.learning_rate
        )
        return optimizer


# -----------------------------------------------------------------------------
# Plotting function
# -----------------------------------------------------------------------------
def plot_results_combined(epochs, train_losses, test_losses, train_accuracies, test_accuracies,
                          sum_head_weight_norms, standard_weight_norm, num_heads, output_dir, timestamp, head_norm_history=None):
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

    line2, = ax1.plot(epochs, test_losses[0], 'b--o', label="MultiHead Val Loss")
    line4, = ax1.plot(epochs, test_losses[1], 'r-o', label="SingleHead Val Loss")
    ax1.set_xlabel("Epoch", fontsize=BIGGER_SIZE)
    ax1.set_ylabel("Loss", color="k", fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.tick_params(axis='both', labelsize=MEDIUM_SIZE)

    ax2 = ax1.twinx()
    line6, = ax2.plot(epochs, test_accuracies[0], 'g--x', label="MultiHead Val Acc")
    line8, = ax2.plot(epochs, test_accuracies[1], 'm-x', label="SingleHead Val Acc")
    ax2.set_ylabel("Accuracy", color="k")
    ax2.tick_params(axis='y', labelcolor="k")

    lines = [line2, line4, line6, line8]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
               mode='expand', borderaxespad=0, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cnn_combined_loss_accuracy_{timestamp}.png")

    # Plot weight norm comparisons
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(top=0.8)
    head_labels = [f"Head {i + 1}" for i in range(num_heads)]
    ax2.bar(head_labels, sum_head_weight_norms, label="MultiHead Weight Norms", alpha=0.7)
    ax2.bar(["Head"], standard_weight_norm, label="SingleHead Weight Norm", alpha=0.7)
    ax2.set_ylabel("Weight Norm")
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0)
    ax2.grid(axis='y', linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cnn_weight_norms_comparison_{timestamp}.png")

    # Plot evolution of head Frobenius norms if available
    if head_norm_history is not None and len(head_norm_history) > 0:
        head_norm_history = np.array(head_norm_history)  # shape: (epochs, num_heads)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        for i in range(head_norm_history.shape[1]):
            ax3.plot(epochs[1:], head_norm_history[:, i], marker='o', label=f"Head {i + 1}")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Frobenius Norm")
        ax3.set_title("Evolution of Head Frobenius Norms")
        ax3.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/head_frobenius_norms_evolution_{timestamp}.png")


# -----------------------------------------------------------------------------
# Main script using Hydra and Lightning Trainer
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config_exp")
def main(cfg: DictConfig):
    print("Starting program.")

    batch_size = cfg.model.batch_size
    epochs = cfg.model.epochs
    learning_rate = cfg.model.learning_rate
    num_classes = cfg.model.num_classes
    hidden_dim = cfg.model.hidden_dim
    num_layers = cfg.model.num_layers

    # Using reg_rate now for the exponential penalty
    reg_rate = cfg.model.reg_rate

    output_dir = cfg.experiment.output_dir + f'/exp_reg{reg_rate}'
    data_dir = cfg.experiment.data_dir

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform)

    print("Set up dataloaders.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    print("Trying to load Resnet18.")
    # Load pretrained ResNet18
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone_feature_dims = [64, 128, 256, 512]
    print("Loaded Resnet18 successfully.")

    # Initialize the multihead model using fusion
    multihead_model = CNNMultiLayerFusionModel(resnet18, backbone_feature_dims, hidden_dim, num_classes)

    # Standard model
    backbone_out_dim = 512
    hidden_dim_std = 492
    standard_model = CNNStandardModel(resnet18, backbone_out_dim, hidden_dim_std, num_classes)

    criterion = nn.CrossEntropyLoss()

    # Print model summaries
    print("Multihead Model Summary:")
    summary(multihead_model, input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "kernel_size"])
    print("Standard Model Summary:")
    summary(standard_model, input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "kernel_size"])

    # Instantiate Lightning modules
    lit_multihead = LitMultiHeadClassifier(
        model=multihead_model,
        criterion=criterion,
        learning_rate=learning_rate,
        momentum=0.9,
        exp_reg_rate=reg_rate
    )
    lit_standard = LitStandardClassifier(
        model=standard_model,
        criterion=criterion,
        learning_rate=learning_rate
    )

    # Create Lightning trainer
    trainer1 = L.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        default_root_dir=f"./lightning_logs/results_exp_reg{reg_rate}",
    )
    trainer2 = L.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        default_root_dir=f"./lightning_logs/results_exp_reg{reg_rate}/standard_model"
    )

    print("Training MultiLayer Fusion Model:")
    trainer1.fit(lit_multihead, train_loader, test_loader)

    print("\nTraining Standard Model:")
    trainer2.fit(lit_standard, train_loader, test_loader)

    # Analyze weight norms
    multihead_weight_norms = [torch.linalg.norm(w).item() for w in multihead_model.fusion.sub_weights]
    standard_weight_norm = torch.linalg.norm(standard_model.head.weight).item()
    print("Multihead Fusion layer weight norms:", multihead_weight_norms)
    print("Standard model head weight norm:", standard_weight_norm)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results_combined(
        list(range(0, epochs + 1)),  # account for lightning's initial validation "sanity check" before 1. training epoch
        [lit_multihead.train_losses, lit_standard.train_losses],
        [lit_multihead.val_losses, lit_standard.val_losses],
        [lit_multihead.train_accuracies, lit_standard.train_accuracies],
        [lit_multihead.val_accuracies, lit_standard.val_accuracies],
        multihead_weight_norms,
        standard_weight_norm,
        num_layers,
        output_dir,
        timestamp,
        head_norm_history=lit_multihead.epoch_head_norms
    )

    # Compute normalized weight norms (RMS)
    multihead_norms_normalized = []
    for w in multihead_model.fusion.sub_weights:
        num_params = w.numel()
        normalized_norm = torch.linalg.norm(w).item() / math.sqrt(num_params)
        multihead_norms_normalized.append(normalized_norm)

    std_weight = standard_model.head.weight.data
    std_norm_normalized = torch.linalg.norm(std_weight).item() / math.sqrt(std_weight.numel())

    print("Multihead Fusion normalized weight norms:", multihead_norms_normalized)
    print("Standard model normalized weight norm:", std_norm_normalized)

    # Plot normalized weight norms comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(top=0.8)
    head_labels = [f"Head {i + 1}" for i in range(len(multihead_norms_normalized))]
    ax.bar(head_labels, multihead_norms_normalized, label="MultiHead Normalized Norm", alpha=0.7)
    ax.bar(["Head"], std_norm_normalized, label="SingleHead Normalized Norm", alpha=0.7)
    ax.set_ylabel("Normalized Weight Norm (RMS)")
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0)
    ax.grid(axis='y', linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normalized_weight_norms_comparison_{timestamp}.png")


if __name__ == "__main__":
    main()
