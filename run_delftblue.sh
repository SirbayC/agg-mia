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
export SAMPLE_FRACTION=0.1
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

# Resource monitoring output
SYS_LOG="$OUTDIR/system_stats.log"

# Measure GPU usage of your job (initialization)
GPU_ACCT_INIT=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
echo "GPU accounting snapshot at job start saved."

# Start CPU/RAM monitor (every 10s)
(
  while true; do
    echo "----- $(date '+%F %T') -----"
    free -h || true
    top -b -n 1 | head -n 20 || true
    sleep 10
  done
) > "$SYS_LOG" 2>&1 &
SYS_MON_PID=$!
echo "Started system monitor (pid=$SYS_MON_PID): $SYS_LOG"

# Ensure system monitor is stopped on exit, error, or cancellation
cleanup_monitors() {
  if [[ -n "${SYS_MON_PID:-}" ]]; then
    kill "$SYS_MON_PID" 2>/dev/null || true
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

# Measure GPU usage of your job (result)
echo "=========================================="
echo "GPU usage by job (accounting):"
nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$GPU_ACCT_INIT"
echo "=========================================="

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results in: $OUTDIR"

echo "Copying logs..."
cp "$ROOT_DIR/outputs/logs/slurm-${SLURM_JOB_ID}.out" "$OUTDIR/"
echo "Done!"

echo "=========================================="
