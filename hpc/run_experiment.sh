#!/bin/bash
# run_experiment.sh – convenience front‑end for HPC
# ----------------------------------------------------------------------
# Core flags
#   --script     Python file to run                      (default: main.py)
#   --wrapper    gpu1_1d | gpu4_2d | cpu_only | local    (default: gpu1_1d)
#   --env_type   conda | venv | none                     (default: conda)
#   --env_name   Environment name / path                 (default: venv)
#
# Everything after the recognised flags is forwarded unchanged to Hydra,
# so you can pass e.g. model.* or data.* overrides as usual.


# -e -> exit immediately if any simple command returns a non-zero status
# -u -> treat unset variables as an error and exit
# -o pipefail -> changes the exit status of a pipeline (cmd1 | cmd2 | cmd3)
# With pipefail set, the pipeline’s exit code becomes the first
# non‑zero exit code produced by any command in the chain.
# Without it, the exit code is that of the last command only.
set -euo pipefail

# ---------- defaults --------------------------------------------------
SCRIPT="main.py"
WRAPPER="gpu1_1d"
ENV_TYPE="conda"
ENV_NAME="venv"
JOB_NAME=""
EXTRA=""

# ---------- parse -----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --script)      SCRIPT=$2; shift 2 ;;
    --wrapper)     WRAPPER=$2; shift 2 ;;
    --env_type)    ENV_TYPE=$2; shift 2 ;;
    --env_name)    ENV_NAME=$2; shift 2 ;;
    --job_name)    JOB_NAME=$2; shift 2 ;;
    *)             EXTRA+=" $1"; shift ;;
  esac
done

# ---------- command ---------------------------------------------------
CMD="python $SCRIPT --multirun $EXTRA"

# ---------- launch ----------------------------------------------------
if [[ "$WRAPPER" == "local" ]]; then
  echo "Running locally:"
  echo "$CMD"
  eval "$CMD"
else
  mkdir -p logs
  echo "Submitting to Slurm via wrapper_${WRAPPER}.sb"
  sbatch --export=ALL,ENV_TYPE="$ENV_TYPE",ENV_NAME="$ENV_NAME" \
         ${JOB_NAME:+--job-name="$JOB_NAME"} \
         "wrapper_${WRAPPER}.sb" $CMD
fi
