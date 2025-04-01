import torch
# Disable cuDNN to bypass the symbol lookup error (note: this may affect performance)
#torch.backends.cudnn.enabled = False

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import lightning as L
import torchmetrics

# PyTorch Geometric imports
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GIN
from torch_geometric.nn import global_add_pool, global_mean_pool

from torch.serialization import add_safe_globals
from torch_geometric.data import Data
add_safe_globals([Data])

from torch.utils.data import random_split

import numpy as np

import hydra
from omegaconf import OmegaConf, DictConfig


# -------------------------
# Data Module for QM9
# -------------------------
class QM9DataModule(L.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download QM9 dataset if not already present.
        QM9(root=self.data_dir)

    def setup(self, stage=None):
        dataset = QM9(root=self.data_dir)
        generator1 = torch.Generator().manual_seed(42)  # fix generator for reproducible results (having same train/test dataset for each weight decay)
        self.qm9_train, self.qm9_val, self.qm9_test = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

    def train_dataloader(self):
        return DataLoader(self.qm9_train, batch_size=self.batch_size, shuffle=True, num_workers=11, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.qm9_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.qm9_test, batch_size=self.batch_size)

# -------------------------
# Utility Function
# -------------------------
def frobenius_norm(model):
    norm = 0
    for param in model.parameters():
        norm += torch.norm(param, p="fro") ** 2
    return torch.sqrt(norm).item()

# -------------------------
# Lightning Module for GIN
# -------------------------
class LitGIN(L.LightningModule):

    def __init__(self, weight_decay, results, learning_rate=0.001,
                 target_index=0, in_channels=11, hidden_channels=64,
                 num_layers=5, out_channels=1, batch_size=64):
        super().__init__()
        # Create a GIN model
        self.gnn = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=0.1
        )
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.target_index = target_index

        self.frobenius_norm_value = None
        self.train_mae_final = None
        self.test_mae_final = None
        self.results = results

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, data):
        # data is a PyG Batch object.

        # data.x: Node feature matrix with shape [num_nodes, num_node_features]
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        # data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
        # Node position matrix with shape [num_nodes, num_dimensions]

        # Obtain node-level outputs.
        x = self.gnn(data.x, data.edge_index)

        # Aggregate node-level outputs to obtain graph-level predictions.
        out = global_mean_pool(x, data.batch)
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        target = batch.y[:, self.target_index]   # choose correct reference target
        loss = self.loss(out.view(-1), target)
        self.train_mae.update(out.view(-1), target)
        self.log_dict({'train_loss': loss, 'train_mae': self.train_mae}, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def on_train_end(self):
        frob_norm = frobenius_norm(self)
        self.frobenius_norm_value = frob_norm
        print(f"Frobenius Norm of the model after training: {frob_norm:.4f}")

        train_mae = self.trainer.callback_metrics.get("train_mae")
        if train_mae is not None:
            self.train_mae_final = train_mae.item() if isinstance(train_mae, torch.Tensor) else train_mae
        else:
            self.train_mae_final = float('nan')

        print(f"Final train MAE: {self.train_mae_final}")

    def test_step(self, batch, batch_idx):
        out = self(batch)
        target = batch.y[:, self.target_index]
        loss = self.loss(out.view(-1), target)
        self.test_mae.update(out.view(-1), target)
        self.log_dict({'test_loss': loss, 'test_mae': self.test_mae}, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def on_test_end(self):
        test_mae_value = self.test_mae.compute()
        self.test_mae_final = test_mae_value.item() if isinstance(test_mae_value, torch.Tensor) else test_mae_value
        print(f"Final test MAE: {self.test_mae_final}")

        self.results.append({
            "train_mae": self.train_mae_final,
            "test_mae": self.test_mae_final,
            "frobenius_norm": self.frobenius_norm_value,
            "margin": 1 / self.frobenius_norm_value,
            "train-test mae": self.train_mae_final - self.test_mae_final,
            "weight_decay": self.weight_decay,
        })

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def get_results(self):
        return self.results

# -------------------------
# Experiment Runner
# -------------------------
def run_experiments(
    num_runs=1,
    output_file="results.csv",
    weight_decay=0.0,
    num_epochs=5,
    batch_size=64,
    learning_rate=2e-5,
    target_index=0,
    in_channels=11,
    hidden_channels=64,
    num_layers=5,
    out_channels=1,
    data_dir='./data'
):
    results = []

    for run in range(num_runs):
        qm9_dm = QM9DataModule(data_dir=data_dir, batch_size=batch_size)
        model = LitGIN(
            weight_decay=weight_decay,
            results=results,
            learning_rate=learning_rate,
            target_index=target_index,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            batch_size=batch_size
        )

        trainer = L.Trainer(
            accelerator='gpu',
            max_epochs=num_epochs,
            default_root_dir=f"./results/results_wd_{weight_decay}",
            enable_checkpointing=False
        )
        trainer.fit(model=model, datamodule=qm9_dm)
        trainer.test(model=model, datamodule=qm9_dm)

        results = model.get_results()
        print(f"Run {run+1}/{num_runs} completed for weight_decay = {weight_decay}.")

    return results

# -------------------------
# Hydra Integration
# -------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    weight_decay_list = cfg.experiment.weight_decay if isinstance(cfg.experiment.weight_decay, list) else [cfg.experiment.weight_decay]
    print(f"weight decays to run: {weight_decay_list}")

    all_results = []
    for wd in weight_decay_list:
        print(f"Running experiments with weight_decay = {wd}")
        # Run experiments for this specific weight_decay
        results = run_experiments(
            num_runs=cfg.experiment.num_runs,
            weight_decay=wd,
            num_epochs=cfg.experiment.num_epochs,
            batch_size=cfg.experiment.batch_size,
            learning_rate=cfg.experiment.learning_rate,
            target_index=cfg.experiment.target_index,
            in_channels=cfg.experiment.in_channels,
            hidden_channels=cfg.experiment.hidden_channels,
            num_layers=cfg.experiment.num_layers,
            out_channels=cfg.experiment.out_channels,
            data_dir=cfg.experiment.data_dir,
        )
        all_results.extend(results)

    # Combine all results into a single DataFrame and save.
    df = pd.DataFrame(all_results)
    df.to_csv(f"./results/final_results_wd_{weight_decay_list[0]}.csv", index=False)
    print("Final results saved to final_results.csv")


if __name__ == "__main__":
    main()
