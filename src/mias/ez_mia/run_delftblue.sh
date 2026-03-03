#!/bin/bash

#SBATCH --job-name=CAV-MIA-AISE-research
#SBATCH --partition=gpu-a100
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --output=/scratch/cosminvasilesc/EZ-MIA/outputs/logs/slurm-%j.out

# One-time setup for code config:
# export HF_HOME="/scratch/cosminvasilesc/HF_CACHE"
# export HF_TOKEN=<token>
# hf download stabilityai/stable-code-3b
# hf download --repo-type dataset codeparrot/github-code-clean

# PREFLIGHT CHECKS:
# - time
# - partition
# - limit inference ? --max_infer_samples=100 \
# - base config

set -euo pipefail

ROOT_DIR="/scratch/cosminvasilesc/EZ-MIA"
REPO_DIR="$ROOT_DIR/ez-mia"
CONDA_ENV_PATH="$ROOT_DIR/ENV"
HF_CACHE_DIR="/scratch/cosminvasilesc/HF_CACHE"

# Create output directory for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$ROOT_DIR/outputs/runs/${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "Output dir: $OUTDIR"
echo "=========================================="

# Load modules
module purge
module load miniconda3

export HF_HOME="$HF_CACHE_DIR"
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1

conda activate "$CONDA_ENV_PATH"

echo "Python:  $(which python) — $(python --version)"
# echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
# echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
echo "=========================================="

cd "$REPO_DIR"

# ── Step 1: Prepare the Dynamic Configuration ────────────────────────────────
echo "[$(date)] Preparing dynamic config for this run..."

# We create a temporary copy of the YAML in the output directory
RUN_CONFIG="$OUTDIR/exp_swallow_code_stablecode_base_run.yaml"
cp configs/exp_swallow_code_stablecode_base.yaml "$RUN_CONFIG"

# Use 'sed' to replace the empty save_artifacts_path: "" with our new $OUTDIR
sed -i "s|save_artifacts_path: \"\"|save_artifacts_path: \"$OUTDIR\"|g" "$RUN_CONFIG"


# ── Step 2: Run the EZ-MIA Pipeline ──────────────────────────────────────────
echo "[$(date)] Starting EZ-MIA attack pipeline..."

# Reduce tqdm update frequency for cleaner log files
export TQDM_MININTERVAL=10

# EZ-MIA runs as a module, passing our newly created run-specific YAML
python -u -m mia --config "$RUN_CONFIG"


echo "=========================================="
echo "Job completed at: $(date)"
echo "Results in: $OUTDIR"

echo "Copying logs..."
cp "$ROOT_DIR/outputs/logs/slurm-${SLURM_JOB_ID}.out" "$OUTDIR/"
echo "Done!"

echo "=========================================="
