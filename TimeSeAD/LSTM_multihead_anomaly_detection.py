import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Union, Callable, List, Type
from torch.utils.data.dataloader import default_collate

# Import TimeSeAD repository modules
from timesead.data.smd_dataset import SMDDataset
from timesead.data.dataset import collate_fn
from timesead.data.transforms.window_transform import WindowTransform
from timesead.data.transforms.target_transforms import ReconstructionTargetTransform
from timesead.models.common import AE
from timesead.models import BaseModel
from timesead.models.reconstruction.lstm_ae import LSTMAEDecoderSimple, LSTMAEAnomalyDetector
from timesead.evaluation import Evaluator


#######################################
# Custom Callback to Track Head Norms
#######################################
class HeadNormTracker(L.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_norms = []  # Will store a list (per epoch) of lists (one norm per head)

    def on_validation_epoch_end(self, trainer, L_module):
        if hasattr(L_module, "decoder"):
            norms = []
            for decoder in L_module.decoder:
                # Compute L2 norm of all parameters in decoder.lstm
                total_norm_sq = 0.0
                for param in decoder.lstm.parameters():
                    total_norm_sq += param.norm(2).item() ** 2
                norm = math.sqrt(total_norm_sq)
                norms.append(norm)
            self.epoch_norms.append(norms)
            # Also log these norms using the Lightning logger (if desired)
            trainer.logger.log_metrics({"head_norms": np.mean(norms)}, step=trainer.current_epoch)
        else:
            pass


#######################################
# 1. Define a Dataset Wrapper to Support TimeSeAD Transforms
#######################################
class TSDatasetWrapper:
    def __init__(self, smd_dataset):
        self.dataset = smd_dataset

    def get_datapoint(self, idx):
        # Return the tuple as provided by SMDDataset.
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @property
    def seq_len(self):
        return self.dataset.seq_len


#######################################
# 2. Define a Simple LSTM Encoder
#######################################
class SimpleLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(SimpleLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        outputs, (h_n, c_n) = self.lstm(x)
        # Return last hidden state from each layer as a list.
        return [h_n[i] for i in range(h_n.size(0))]


#######################################
# 3. Standard Single-Head Autoencoder Module
#######################################
class LSTMAutoencoderLightning(AE, BaseModel, L.LightningModule):
    def __init__(self, input_dim, enc_hidden_dim=32, dec_hidden_dims=[40], num_layers=1, lr=1e-3, window_size=50):

        encoder = SimpleLSTMEncoder(input_dim, enc_hidden_dim, num_layers=num_layers)
        decoder = LSTMAEDecoderSimple(enc_hidden_dim, dec_hidden_dims, input_dim)
        super().__init__(encoder, decoder, return_latent=False)

        self.window_size = window_size
        self.lr = lr

    def encode(self, x: torch.Tensor):
        hidden = self.encoder(x)
        return hidden

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # We expect a single Tensor in a tuple, shaped (T,B,D)=(window_size,batch_dim,feature_dim)
        x = inputs[0]
        initial_hidden = self.encoder(x)
        x_recon = self.decoder(initial_hidden, seq_len=x.size(0))
        return x_recon

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self.forward((x,))  # shape: (T, B, D)
        loss = F.mse_loss(x_recon, x)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self.forward((x,))
        loss = F.mse_loss(x_recon, x)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


#######################################
# 4. Multi-Head Autoencoder Module
#######################################
class LSTMAutoencoderMultiHeadLightning(AE, BaseModel, L.LightningModule):
    def __init__(self, input_dim, enc_hidden_dim=32, dec_hidden_dims=[40],
                 num_layers=1, lr=1e-3, window_size=50, num_heads=3):
        """
        Multi-head LSTM autoencoder, using (time, batch, features) ordering.

        Each sample x has shape (T, D) with T = window_size * num_heads.
        After default PyTorch collation with batch size B, we get (T, B, D).
        We do a single pass through the encoder, then have 'num_heads' decoders,
        each reconstructing a subwindow of length 'window_size'.
        """
        # LSTM encoder expects (T, B, D) because batch_first=False
        encoder = SimpleLSTMEncoder(input_dim, enc_hidden_dim, num_layers=num_layers)
        # Create a decoder for each head. Each returns (window_size, B, input_dim)
        decoders = nn.ModuleList([
            LSTMAEDecoderSimple(enc_hidden_dim, dec_hidden_dims, input_dim)
            for _ in range(num_heads)
        ])
        super().__init__(encoder, decoders, return_latent=False)

        self.window_size = window_size
        self.num_heads = num_heads
        self.lr = lr

    def encode(self, x: torch.Tensor):
        hidden = self.encoder(x)
        return hidden

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        inputs: a 1-tuple containing x => (T,B,D), where T = window_size*num_heads
        Return shape => (T,B,D).
        """
        x = inputs[0]  # shape (T,B,D)
        initial_hidden = self.encoder(x)
        recons = []
        for i in range(self.num_heads):
            # Decode subwindow i => shape (window_size, B, D)
            sub_recon = self.decoder[i](initial_hidden, seq_len=self.window_size)
            recons.append(sub_recon)
        # Concatenate along the time dimension => (num_heads*window_size, B, D) == (T,B,D)
        return torch.cat(recons, dim=0)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # shape (B,T,D) from the loader
        x = x.permute(1, 0, 2)  # now (T,B,D)
        outputs = self.forward((x,))  # (T, B, D)

        # Option A: Single MSE across all time steps:
        loss = F.mse_loss(outputs, x)

        # Option B: Sum the MSE per subwindow and average:
        #loss = 0.0
        #for i in range(self.num_heads):
        #    subwin = x[i * self.window_size:(i + 1) * self.window_size, :, :]
        #    sub_pred = outputs[i * self.window_size:(i + 1) * self.window_size, :, :]
        #    loss += F.mse_loss(sub_pred, subwin)
        #loss /= self.num_heads

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # shape (B,T,D) from the loader
        x = x.permute(1, 0, 2)  # now (T,B,D)
        outputs = self.forward((x,))  # (T, B, D)

        # Option A: Single MSE across all time steps:
        loss = F.mse_loss(outputs, x)

        # Option B: Sum the MSE per subwindow and average:
        #loss = 0.0
        #for i in range(self.num_heads):
        #    subwin = x[i * self.window_size:(i + 1) * self.window_size, :, :]
        #    sub_pred = outputs[i * self.window_size:(i + 1) * self.window_size, :, :]
        #    loss += F.mse_loss(sub_pred, subwin)
        #loss /= self.num_heads

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


#######################################
# 5. Main Function: Data, Training, and Norm Tracking
#######################################
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Instantiate the SMD dataset and wrap it
    smd = SMDDataset(server_id=17, training=True)
    wrapped_dataset = TSDatasetWrapper(smd)

    input_dim = smd.num_features  # 38

    # hyperparameters
    use_multihead = cfg.experiment.use_multihead
    detect_anomaly = cfg.experiment.detect_anomaly
    epochs = cfg.experiment.epochs

    enc_hidden_dim = cfg.model.enc_hidden_dim
    dec_hidden_dims = cfg.model.dec_hidden_dims
    num_layers = cfg.model.num_layers
    lr = cfg.model.lr
    base_window_size = cfg.model.base_window_size

    num_heads = 3
    if use_multihead:
        segment_length = base_window_size * num_heads
    else:
        segment_length = base_window_size * num_heads
        num_heads = 1

    # Instantiate WindowTransform that provides the windows
    window_transform = WindowTransform(parent=wrapped_dataset, window_size=segment_length, step_size=1, reverse=False)
    # Chain the ReconstructionTargetTransform.
    recon_transform = ReconstructionTargetTransform(parent=window_transform, replace_labels=True)

    # Build a dataset of windows.
    class WindowedDataset(torch.utils.data.Dataset):
        def __init__(self, transform):
            self.transform = transform

        def __len__(self):
            return len(self.transform)

        def __getitem__(self, idx):
            inputs, targets = self.transform._get_datapoint_impl(idx)
            x = inputs[0]  # shape: (segment_length, input_dim)
            y = targets[0]
            return x, y

    windowed_dataset = WindowedDataset(recon_transform)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(windowed_dataset, [0.9, 0.1], generator=generator)

    # We define a custom collate that just wraps "y" in (y, ) so the fit function of the
    # anomaly detector code sees b_targets = (target,).
    # But we do not transpose x or y, so default_collate will produce (T,B,D) for them automatically (since T is first dim).
    def anomaly_collate_fn(batch):
        # batch is a list of (x, y) => each (T, D)
        # default_collate => x => (T,B,D), y => (T,B,D).
        # Then we wrap y => (y,).
        # So final => ( (T,B,D), ((T,B,D), ) )
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        X_coll = default_collate(xs)  # shape (T, B, D)
        Y_coll = default_collate(ys)  # shape (T, B, D)

        # Wrap Y_coll in a single-element tuple
        Y_coll = (Y_coll,)

        return (X_coll, Y_coll)

    # Define train/val/test loaders
    batch_dim = 1
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=anomaly_collate_fn)

    smd_test = SMDDataset(server_id=17, training=False)
    test_loader = DataLoader(smd_test, batch_size=16, shuffle=False)

    # Instantiate model
    if use_multihead:
        model = LSTMAutoencoderMultiHeadLightning(
            input_dim=input_dim,
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dims=dec_hidden_dims,
            num_layers=num_layers,
            lr=lr,
            window_size=base_window_size,  # Still 50 per sub-window
            num_heads=num_heads
        )
        # Create head norm tracker callback.
        head_norm_tracker = HeadNormTracker()
        callbacks = [head_norm_tracker]
    else:
        model = LSTMAutoencoderLightning(
            input_dim=input_dim,
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dims=dec_hidden_dims,
            num_layers=num_layers,
            lr=lr,
            window_size=segment_length
        )
        callbacks = []

    # Create Trainer with callbacks
    trainer = L.Trainer(max_epochs=epochs, accelerator="cpu", callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # After training, if using multihead, retrieve and plot head norms
    if use_multihead:
        epoch_norms = np.array(head_norm_tracker.epoch_norms)  # shape: (epochs, num_heads)
        epochs = np.arange(0, epoch_norms.shape[0])

        # Plot evolution of each head's norm over epochs
        plt.figure(figsize=(10, 6))
        for head in range(num_heads):
            plt.plot(epochs, epoch_norms[:, head], marker='o', label=f'Head {head + 1}')
        plt.xlabel("Epoch")
        plt.ylabel("L2 Norm of Decoder LSTM Parameters")
        plt.title("Evolution of Decoder Head Norms Over Epochs")
        plt.legend()
        plt.savefig(f"LSTM_multihead_norms_evolution_{timestamp}.png")
        plt.show()

        # Create a bar plot comparing the final norms
        final_norms = epoch_norms[-1, :]
        plt.figure(figsize=(8, 6))
        head_labels = [f"Head {i + 1}" for i in range(num_heads)]
        plt.bar(head_labels, final_norms, alpha=0.7)
        plt.ylabel("Final L2 Norm")
        plt.title("Final Decoder Head Weight Norm Comparison")
        plt.savefig(f"LSTM_multihead_final_norms_{timestamp}.png")
        plt.show()


    # Fit and test anomaly detector
    if detect_anomaly:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_evaluator = Evaluator()

        detector = LSTMAEAnomalyDetector(model).to(device)
        detector.fit(val_loader)

        labels, scores = detector.get_labels_and_scores(test_loader)

        results = []
        evaluation_metrics = ['best_ts_f1_score', 'ts_auprc', 'best_ts_f1_score_classic', 'ts_auprc_unweighted',
                              'best_f1_score', 'auprc']
        for metric in evaluation_metrics:
            test_score, info = test_evaluator.__getattribute__(metric)(labels, scores)
            print(f"{metric}: {test_score:.4f}")
            print(info)

            row = {
                "metric": metric,
                "test_score": test_score,
                "threshold": info.get("threshold", ""),
                "precision": info.get("precision", ""),
                "recall": info.get("recall", "")
            }
            results.append(row)

        df = pd.DataFrame(results, columns=["metric", "test_score", "threshold", "precision", "recall"])
        if use_multihead:
            df.to_csv(f"evaluation_results_multihead_{timestamp}.csv", index=False)
        else:
            df.to_csv(f"evaluation_results_singlehead_{timestamp}.csv", index=False)


if __name__ == '__main__':
    main()
