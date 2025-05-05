# HPC scripts for *Lightning + Hydra* experiments

This folder contains two kinds of files that help to launch any
experiment from the repository on a Slurm cluster:

| File(s) | Purpose                                                                                                                                                                                                                                                                                                         |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `wrapper_<flavor>.sb` | *Resource presets* – a thin Slurm batch script that requests the right number of GPUs/CPUs/RAM, sets up your software environment, prints job info, and finally **executes whatever command you pass on the `sbatch` CLI**.  Nothing experiment‑specific lives here so this script can be used for any experiment. |
| `run_experiment.sh` | A convenience **launcher**.  It turns a human‑friendly command with flags such as `--script`, `--wrapper gpu1`,`--env_name` into the correct `sbatch … wrapper_gpu1.sb python … --multirun …` call (or runs it locally when you use `--wrapper local`).                                                         |

---

## Wrapper scripts

| Wrapper                   | GPUs | CPUs | RAM | Max Time | Typical use‑case           |
|---------------------------|-----:|-----:|----:|----------|----------------------------|
| **`wrapper_gpu1_1d.sb`**  | 1 | 4 | 48 GB |  1 day   | Simple single‑GPU training |
| **`wrapper_gpu4_2d.sb`**  | 4 | 16 | 192 GB |  2 days  | Multi‑GPU training         |
| **`wrapper_cpu_only.sb`** | 0 | 8 | 32 GB |  8 h     | Fast CPU‑only programs     |


### Add your own wrapper

Copy one of the files, tweak the `#SBATCH` resource lines, give it a descriptive
name such as `wrapper_gpu8_5d.sb`, and it will be automatically usable via

```bash
./run_experiment.sh --wrapper gpu8_5d  ...
```

---

## `run_experiment.sh` – CLI reference

| Flag                             | Required | Default             | Description                                     |
|----------------------------------|----------|---------------------|-------------------------------------------------|
| `--script PATH`                  | ✓ | `main.py`           | Python entry‑point of the experiment            |
| `--wrapper NAME`                 | ✗ | `gpu1_1d`           | `gpu1_1d`, `gpu4_2d`, `cpu_only`, or `local`    |
| `--env_type (conda\|venv\|none)` | ✗ | `conda`             | How to activate Python                          |
| `--env_name NAME/Path`           | ✗ | `venv`              | Conda env name **or** venv path                 |
| `--job_name NAME`                | ✗ | *(wrapper default)* | Slurm job‑name                     |
| *anything else*                  | – | –                   | Passed verbatim to Hydra (`model.*`, `data.*`, …) |

**Local mode** (`--wrapper local`) executes the command in the current shell
without Slurm – e.g. for quick debugging.

---

## Usage examples

**Tip**: If permission is denied after you run the .sh script, set execute permission using the following command:

```bash
chmod +x run_experiment.sh
```

### 1) Training a ResNet experiment (default env, single GPU)

```bash
./run_experiment.sh \
    --script resnet_multihead.py \
    --job_name ResNet_reg_sweep \
    model.p=1.0 model.reg_rate=0.1,0.01,0.001
```

### 2) CPU‑only run for collecting data inside a Python venv

```bash
./run_experiment.sh \
    --script vis/collect_visdata.py \
    --wrapper cpu_only \
    --env_type venv \
    --env_name ~/venvs/lightning
```

### 3) Multi‑GPU sweep with a custom conda env

```bash
./run_experiment.sh \
    --script transformer_baseline.py \
    --wrapper gpu4_2d \
    --env_name roberta_env \
    data.batch_size=16 trainer.strategy=ddp
```

### 4) Quick local test (no Slurm)

```bash
./run_experiment.sh \
    --script CNN_multihead_prediction.py \
    --wrapper local \
    train.epochs=1
```

---

## Submitting without the launcher (optional)

If you prefer raw `sbatch`, just pass the wrapper and your full command:

```bash
sbatch --export=ALL,ENV_TYPE=conda,ENV_NAME=venv \
       wrapper_gpu1_1d.sb \
       python resnet_multihead.py --multirun \
       model.p=1e-5 model.reg_rate=0.0001
```

Everything after the wrapper filename becomes `$@` inside the wrapper and is executed verbatim.

---
