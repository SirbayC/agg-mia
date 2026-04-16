#!/bin/bash

#SBATCH --job-name=CAV-MIA-AISE-research
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --output=/scratch/cosminvasilesc/AGG_MIA/outputs/logs/slurm-%j.out

####################################
# PREFLIGHT CHECKS:
# - time
# - partition
export MIA="trawic" # trawic / miaadv / loss / mkp / pac / bow
export LLM="bigcode/starcoder2-3b" # bigcode/starcoder2-3b / bigcode/starcoder2-7b / bigcode/starcoder2-15b
export SAMPLE_FRACTION=0.01
####################################

set -euo pipefail

ROOT_DIR="/scratch/cosminvasilesc/AGG_MIA"
REPO_DIR="$ROOT_DIR/agg-mia"
HF_CACHE_DIR="/scratch/cosminvasilesc/HF_CACHE"
UV_CACHE_DIR="/scratch/cosminvasilesc/UV_CACHE"

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
module load 2025
module load python

export HF_HOME="$HF_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR" 
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it once with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

cd "$REPO_DIR"

uv run python -u -m src.main \
  --output_dir="$OUTDIR" \
  --mia="$MIA" \
  --model="$LLM" \
  --sample_fraction="$SAMPLE_FRACTION" \
  --infer_engine vllm

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results in: $OUTDIR"

echo "Copying logs..."
cp "$ROOT_DIR/outputs/logs/slurm-${SLURM_JOB_ID}.out" "$OUTDIR/"
echo "Done!"

echo "=========================================="
