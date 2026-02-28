#!/bin/bash

#SBATCH --job-name=CAV-MIA-AISE-research
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --output=/scratch/cosminvasilesc/MIA-ADV/outputs/logs/slurm-%j.out

# PREFLIGHT CHECKS:
# - time
# - partition
# - limit inference ? --max_infer_samples=100 \

set -euo pipefail

ROOT_DIR="/scratch/cosminvasilesc/MIA-ADV"
REPO_DIR="$ROOT_DIR/MIA_Adv"
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

# ── Step 1: Perturb ────────────────────────────────────────────────────────────
# Copy raw APPS JSONs into OUTDIR, then perturb in-place there.
cp dataset/APPS/train_victim_APPS.json "$OUTDIR/"
cp dataset/APPS/test_victim_APPS.json  "$OUTDIR/"

echo "[$(date)] Running perturb.py (train)..."
python perturb.py \
  --input_dir "$OUTDIR" \
  --input_file train_victim_APPS.json \
  --output_file train_victim.json

echo "[$(date)] Running perturb.py (test)..."
python perturb.py \
  --input_dir "$OUTDIR" \
  --input_file test_victim_APPS.json \
  --output_file test_victim.json

# ── Step 2: Inference ──────────────────────────────────────────────────────────
echo "[$(date)] Running inference (train + test)..."
cd "$REPO_DIR/llm_inference"
python -u run_lm.py \
  --mode=victim \
  --data_dir="$OUTDIR" \
  --lit_file=./literals.json \
  --langs=python \
  --output_dir="$OUTDIR" \
  --pretrain_dir="bigcode/santacoder" \
  --log_file="$OUTDIR/inference.log" \
  --model_type=santacoder \
  --block_size=1024 \
  --eval_line \
  --logging_steps=500 \
  --seed=42 \
  --generate_method=top-k \
  --topk=50 \
  --temperature=0.8 \
  --max_infer_samples=100 \
  --bf16

cd "$REPO_DIR"

# ── Step 3: Classifier ─────────────────────────────────────────────────────────
echo "[$(date)] Running classifier..."
python classifier.py \
  --input_dir "$OUTDIR" \
  --true_file  "train_santacoder_victim_infer.txt" \
  --false_file "test_santacoder_victim_infer.txt" \
  --true_gt_file  "train_victim.json" \
  --false_gt_file "test_victim.json" \
  --feature_path "$OUTDIR/feature.npz" \
  --n_samples_per_class 2000 \
  --global_random_seed 140120031 \
  --random_state 676269283 \
  --random_state_test 212129145 \
  --dropout 0.1 \
  --batch_size 4 \
  --lr 1e-3 \
  --num_epochs 25 \
  --hidden_dims 512 512 512

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results in: $OUTDIR"

echo "Copying logs..."
cp "$ROOT_DIR/outputs/logs/slurm-${SLURM_JOB_ID}.out" "$OUTDIR/"
echo "Done!"

echo "=========================================="
