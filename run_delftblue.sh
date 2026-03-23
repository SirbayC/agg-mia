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

# Live GPU monitoring outputs
GPU_LOG="$OUTDIR/gpu_live_stats.csv"
GPU_APPS_LOG="$OUTDIR/gpu_compute_apps.csv"

# Measure GPU usage of your job (initialization)
GPU_ACCT_INIT=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
echo "GPU accounting snapshot at job start saved."

# Start lightweight live GPU monitor (every 10s)
echo "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu" > "$GPU_LOG"
echo "timestamp,pid,process_name,used_gpu_memory,gpu_uuid" > "$GPU_APPS_LOG"
(
  while true; do
    TS=$(date '+%F %T')

    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | /usr/bin/sed "s/^/$TS,/" >> "$GPU_LOG" || true

    APP_LINES=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n "${APP_LINES// }" ]] && [[ "$APP_LINES" != *"No running processes found"* ]]; then
      while IFS= read -r line; do
        [[ -n "$line" ]] && echo "$TS,$line" >> "$GPU_APPS_LOG"
      done <<< "$APP_LINES"
    else
      echo "$TS,NO_ACTIVE_COMPUTE_APPS,,," >> "$GPU_APPS_LOG"
    fi

    sleep 10
  done
) &
GPU_MON_PID=$!
echo "Started live GPU monitor (pid=$GPU_MON_PID): $GPU_LOG and $GPU_APPS_LOG"

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
module load slurm

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

srun python -u -m src.main \
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
