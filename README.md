# PyTorch Lightning & Hydra for ML Experiments

## Introduction
This repository demonstrates how to structure and streamline machine learning experiments using [PyTorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/). These tools help simplify PyTorch training loops, improve experiment reproducibility, and manage configurations efficiently.

## Installation
To install PyTorch Lightning and Hydra, use the following commands:

```bash
pip install lightning hydra-core
```

For specific versions or GPU support, refer to the official documentation:
- [PyTorch Lightning Installation](https://lightning.ai/docs/pytorch/stable/#install-lightning)
- [Hydra Installation](https://hydra.cc/docs/intro/#versions)

## Transforming Code into Lightning Code
When transitioning from vanilla PyTorch to PyTorch Lightning, follow these steps:

### 1. Move Computational Code into `LightningModule`
Define your model architecture in the `__init__` method of a `LightningModule`:

```python
import pytorch_lightning as L
import torch.nn as nn
import torch.optim as optim

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)  # Example layer
```

### 2. Set `forward` Hook
The `forward` method defines prediction/inference behavior:

```python
    def forward(self, x):
        return self.layer(x)  # Define inference
```

### 3. Move Optimizers to `configure_optimizers`
Define the optimizer in the `configure_optimizers` method:

```python
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
```

### 4. Move Training Logic to `training_step`
Handle training logic inside the `training_step` method:

```python
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)  # Logging
        return loss
```

### 5. Move Validation Logic to `validation_step`
Define validation behavior in `validation_step`:

```python
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
```

### 6. Remove Any `.cuda()` or Device Calls
Lightning manages hardware automatically, so remove explicit `.cuda()` or `device` calls.

### 7. Override `LightningModule` Hooks (Optional)
Lightning provides hooks like `on_train_start`, `on_epoch_end`, etc., which you can override if needed.

### 8. Initialize the `LightningModule`

```python
model = LitModel()
```

### 9. Initialize the `Lightning Trainer`
The `Trainer` automates various engineering tasks, including:
- Training loops
- Hardware device selection
- Handling `model.train()` and `model.eval()` states
- Optimizer zeroing (`zero_grad`)

```python
from pytorch_lightning import Trainer
trainer = Trainer(max_epochs=10)
```


### 11. Pass Any PyTorch `DataLoader` to the Trainer and Train the Model
Finally, pass the `LightningModule` and `DataLoader` to the `Trainer` to start training:

```python
trainer.fit(model, dataloader)
```

## Using Hydra for Configuration Management
[Hydra](https://hydra.cc/) simplifies handling experiment configurations. Define a YAML configuration file (`config.yaml`):

```yaml
trainer:
  max_epochs: 10
  gpus: 1  # Set to available GPUs
model:
  learning_rate: 0.001
```

Then, integrate Hydra in your script:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    model = MyModel()
    trainer = Trainer(**cfg.trainer)
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
```
