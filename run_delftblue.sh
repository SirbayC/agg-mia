#!/bin/bash

#SBATCH --job-name=CAV-MIA-AISE-research
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --output=/scratch/cosminvasilesc/AGG-MIA/outputs/logs/slurm-%j.out

####################################
# PREFLIGHT CHECKS:
# - time
# - partition
export MIA="trawic" # trawic / ezmia / miaadv / loss / mkp / pac / bow
export LLM="bigcode/starcoder2-3b" # bigcode/starcoder2-3b / bigcode/starcoder2-7b / bigcode/starcoder2-15b
export SAMPLE_FRACTION=0.01
####################################

set -euo pipefail

ROOT_DIR="/scratch/cosminvasilesc/AGG-MIA"
REPO_DIR="$ROOT_DIR/agg-mia"
CONDA_ENV_PATH="$ROOT_DIR/ENV"
HF_CACHE_DIR="/scratch/cosminvasilesc/HF_CACHE"

# Create output directory for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$ROOT_DIR/outputs/runs/${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

# Live GPU monitoring output
GPU_LOG="$OUTDIR/gpu_stats.log"

# Start full nvidia-smi monitor (every 20s)
(
  while true; do
    echo "=========================================="
    echo "$(date '+%F %T')"
    nvidia-smi || true
    echo
    sleep 20
  done
) > "$GPU_LOG" 2>&1 &
GPU_MON_PID=$!
echo "Started live GPU monitor (pid=$GPU_MON_PID): $GPU_LOG"

cleanup_monitors() {
  if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill "$GPU_MON_PID" 2>/dev/null || true
  fi
}
trap cleanup_monitors EXIT INT TERM

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
export HF_HUB_OFFLINE=1

conda activate "$CONDA_ENV_PATH"

echo "Python:  $(which python) — $(python --version)"
# echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
# echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
echo "=========================================="

cd "$REPO_DIR"

# Summary GPU info before running job
echo "GPU info before job:"
nvidia-smi

python -u -m src.main \
  --output_dir="$OUTDIR" \
  --mia="$MIA" \
  --model="$LLM" \
  --sample_fraction="$SAMPLE_FRACTION"

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results in: $OUTDIR"

echo "Copying logs..."
cp "$ROOT_DIR/outputs/logs/slurm-${SLURM_JOB_ID}.out" "$OUTDIR/"
echo "Done!"

echo "=========================================="
