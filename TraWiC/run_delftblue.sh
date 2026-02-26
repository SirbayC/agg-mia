#!/bin/bash

#SBATCH --job-name=CAV-MIA-AISE-research
#SBATCH --partition=gpu-a100
#SBATCH --time=09:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --output=/scratch/cosminvasilesc/TRAWIC/outputs/logs/slurm-%j.out

# PREFLIGHT CHECKS:
# - time
# - partition
# --limit_per_class=50

set -euo pipefail

ROOT_DIR="/scratch/cosminvasilesc/TRAWIC"
REPO_DIR="$ROOT_DIR/TraWiC"
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

# ── Step 1: Model Inference ────────────────────────────────────────────────────
echo "[$(date)] Running SantaCoder inference..."
python -u src/main_santacoder.py \
  --output_dir="$OUTDIR" \
  --limit_per_class=500

# ── Step 2: Build Classification Dataset ──────────────────────────────────────
echo "[$(date)] Building classification dataset..."
python -u src/data/dataset_builder.py \
  --input_dir="$OUTDIR" \
  --output_dir="$OUTDIR/rf_data" \
  --syntactic_threshold=100 \
  --semantic_threshold=20

# ── Step 3: Train Classifier ───────────────────────────────────────────────────
echo "[$(date)] Training Random Forest classifier..."
python -u src/inspector_train.py \
  --input_dir="$OUTDIR/rf_data" \
  --output_dir="$OUTDIR" \
  --syntactic_threshold=100 \
  --semantic_threshold=20 \
  --visualisation=True

# ── Step 4: Evaluate Classifier ────────────────────────────────────────────────
echo "[$(date)] Evaluating classifier..."
python -u src/inspector_test.py \
  --input_dir="$OUTDIR/rf_data" \
  --model_dir="$OUTDIR" \
  --output_dir="$OUTDIR" \
  --syntactic_threshold=100 \
  --semantic_threshold=20

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results in: $OUTDIR"

echo "Copying logs..."
cp "$ROOT_DIR/outputs/logs/slurm-${SLURM_JOB_ID}.out" "$OUTDIR/"
echo "Done!"

echo "=========================================="
