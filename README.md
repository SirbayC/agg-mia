# AGG-MIA
Research question: given a code sample, can an attack infer whether it was in the target model pretraining distribution (member vs non-member)?

### Core assumptions

- Data is provided as two parquet-backed sets:
  - seen: (presumed) members
  - unseen: (presumed) non-members
- Loader expects at least one parquet file in each folder:
  - data/seen
  - data/unseen
- Parquet rows contain content and optionally blob_id columns.

### Target model and data sources

Target model is StarCoder2, the bigcode/starcoder2-3b variant. As mentioned in its main paper, it has been traained on the-stack-v2-train-smol, files considered as seen.

For unseen files, this project relies on The Heap, deduplicated against The Stack v2 with exact_duplicates_stackv2 = false and near_duplicates_stackv2 = false.

### Validity limits

- Ground truth is dataset-defined membership proxy, not direct training log membership.
- Distribution mismatch between seen and unseen can inflate separability independently of true memorization.
- Compute allocation can strongly affect throughput and practical reproducibility (for example MIG vs full GPU).
- Very small sample_fraction can produce unstable per-run metrics and noisy timing.

## Implemented attacks

| Attack | Idea | Paper | Adapted implementation (forked from original) |
|---|---|---|---|
| trawic | Reconstruction-style features + random forest | [Trained Without My Consent: Detecting Code Inclusion In Language Models Trained on Code](https://arxiv.org/abs/2402.09299) | https://github.com/SirbayC/TraWiC |
| miaadv | Perturbation features + MLP | [Effective Code Membership Inference for Code Completion Models via Adversarial Prompts](https://arxiv.org/abs/2511.15107v1) | https://github.com/SirbayC/MIA_Adv |
| loss | Negative language-model loss | [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) | https://github.com/SirbayC/thesis-code-mia-mkp |
| mkp | Min-K token log-probability score | [Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789) | https://github.com/SirbayC/thesis-code-mia-mkp |
| pac | Polarized-Augment Calibration score | [Data Contamination Calibration for Black-box LLMs](https://arxiv.org/abs/2405.11930v1) | https://github.com/AISE-TUDelft/PoisonedChalice/blob/main/Pac.py |
| bow | TF-IDF + logistic regression shift detector | N/A (control baseline) | N/A (implemented in this repo) |

### TraWiC process

Step 1: Pre-processing (Element Extraction): TraWiC parses a target script to extract unique syntactic identifiers (e.g., variable, function, and class names) and semantic identifiers (e.g., comments, strings, and docstrings). It then masks these specific elements, splitting the surrounding code into a "prefix" and a "suffix".

Step 2: Inference (Model Querying): Using the Fill-In-the-Middle (FIM) technique, TraWiC feeds the prefix and suffix to the target Large Language Model and queries it to predict the missing masked element.

Step 3: Comparison (Hit Detection): The LLM's generated output is compared to the original masked element. Syntactic elements require an exact match to be counted as a "hit," while semantic elements use fuzzy matching (Levenshtein edit distance) to account for slight natural language variations.

Step 4: Classification (Dataset Inclusion Prediction): The normalized "hit rates" for each of the six element categories are fed into a Machine Learning classifier (such as a Random Forest). The classifier evaluates these scores and outputs a final binary decision indicating whether the script was likely included in the model's training dataset.

### AdvPrompt-MIA (miaadv) process

Step 1: Baseline Inference (Unperturbed Query): The original code prefix is fed into the target code completion model (the victim model) to generate an initial, unperturbed code completion.

Step 2: Adversarial Perturbation (Prompt Generation): The system automatically applies five types of semantics-preserving transformations (e.g., inserting dead loops, adding debug prints, or renaming variables) to the original code prefix to generate 11 different perturbed versions of the input.

Step 3: Perturbed Inference (Querying the Victim): The target code completion model is queried again, this time using all 11 of the perturbed inputs, and all of the resulting code completions are collected.

Step 4: Feature Extraction (Measuring Stability): The system compares all the generated outputs (both unperturbed and perturbed) against the actual ground-truth completion. It uses CodeBERT to calculate textual similarity and also calculates token perplexity (how confident the model was). These metrics—including their means and standard deviations—are combined into a 27-dimensional feature vector.

Step 5: Classification (Membership Prediction): The 27-dimensional feature vector is fed into a trained Multilayer Perceptron (MLP) neural network. The MLP evaluates the stability patterns in the vector and outputs a final binary decision: 1 (Member of the training dataset) or 0 (Non-Member). Again needs some data to train the MLP, but can be from a different dataset.

## Running MIAs

This section is only about executing experiments.

### 1) Install

Base dependencies:

```bash
uv sync --frozen
```

If uv is not installed yet:

```bash
python -m pip install -U uv
```


### 2) Local run

```bash
uv run python -u -m src.main \
  --mia trawic \
  --model bigcode/starcoder2-3b \
  --data_dir ./data \
  --sample_fraction 0.01 \
  --train_test_split 0.8 \
  --batch_size 1 \
  --seed 42 \
  --output_dir ./output_runs/local_test
```

Key CLI arguments:

- --mia: trawic | miaadv | loss | mkp | pac | bow
- --model: Hugging Face model id
- --infer_engine: hf | vllm (vllm currently supported for trawic)
- --sample_fraction: fraction loaded from each split
- --output_dir: destination for predictions and metrics

### 3) DelftBlue run

Use:

```bash
./submit_delftblue.sh
```

run_delftblue.sh controls cluster-side settings such as:

- MIA, LLM, SAMPLE_FRACTION

The job now runs with uv in locked, offline-safe mode (`uv run --frozen --no-sync ...`) on the DelftBlue Python module stack.
The Python version is pinned in [.python-version](.python-version).

### 3.1) Recreate DelftBlue environment (from scratch)

Run once on DelftBlue:

```bash
module purge
module load 2025
module load python

export ROOT_DIR="/scratch/cosminvasilesc/AGG_MIA"
export REPO_DIR="$ROOT_DIR/agg-mia"
export UV_CACHE_DIR="/scratch/cosminvasilesc/UV_CACHE"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd "$REPO_DIR"

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install --only-binary vllm vllm --torch-backend=cu128
uv lock
uv sync --frozen
```

Use check_env.py to make sure the environment is correctly set up. (Un-comment the line in run_delftblue.sh)

Submit jobs as usual:

```bash
./submit_delftblue.sh
```

Important: submit_delftblue.sh currently auto-commits and pushes local changes before submission.

### 4) Outputs and metrics

Each run writes:

- predictions.csv with columns: blob_id, label, score
- metrics.csv with:
  - roc auc
  - TPR @ 1% FPR
  - TPR @ 0.1% FPR

The DelftBlue run script also stores gpu_stats.log snapshots.

### 5) Multi-run analysis

Aggregate runs into an interactive dashboard:

```bash
uv run python -m src.results.scripts.unify_metrics <root_folder>
```

Recompute metrics from saved predictions:

```bash
uv run python -m src.results.scripts.recompute_metrics <predictions_csv>
```

---

## Data Preparation

To download (smol) files:

1. `module load 2025 && module load python`
2. Set `SWH_TOKEN` in the environment using: https://archive.softwareheritage.org/oidc/profile/#tokens
3. Run from repo root: `uv run python src/datasets/download_seen.py`
4. Run from repo root: `uv run python src/datasets/download_unseen.py`
5. Run from repo root: `uv run python src/datasets/show_parquet_samples.py`

---

## Development

This section is for modifying or extending the codebase.

### Workflow

1. Make local changes.
2. Sync and submit with:

```bash
./submit_delftblue.sh
```

3. See latest logs with:

```bash
tail -100f "$(ls -t | head -n 1)"
```

4. Copy outputs to local machine and clear remote run outputs with:

```bash
scp -r cosminvasilesc@login.delftblue.tudelft.nl:/scratch/cosminvasilesc/AGG-MIA/outputs/runs/* "C:\Coding_projects\AGG_MIA\output_runs" && ssh delftblue "rm -rf /scratch/cosminvasilesc/AGG-MIA/outputs/runs/*"
```

### Project map

```text
AGG_MIA/
├── pyproject.toml                # uv project metadata and dependencies
├── uv.lock                       # locked dependency resolution
├── .python-version               # pinned Python interpreter version for uv
├── analysis/                      # Research (preliminary) results
├── data/
│   ├── seen/
│   └── unseen/
├── src/
│   ├── main.py                    # CLI and experiment orchestration
│   ├── datasets/                  # Download and loading pipeline
│   ├── models/
│   │   └── loader.py              # Tokenizer/model loading and dtype selection
│   ├── mias/                      # Attack implementations and configs
│   └── results/                   # Prediction/metric export and aggregation scripts
├── run_delftblue.sh               # Cluster run script (SLURM job)
└── submit_delftblue.sh            # Sync + submit helper
```

### Adding a new attack

1. Implement a subclass of MIAttack under src/mias.
2. Add attack parameters in a config dataclass if needed.
3. Register the class in load_mia_class in src/main.py.
4. Keep train and evaluate interfaces compatible with text, blob_id, label schema.
5. Return one membership score per evaluated sample.

### Tests

```bash
pytest -q
```

Current tests are mainly in src/mias/trawic/tests.

## Future research paths

In order of priority:
1. experiment with llm acceleration to improve attack speed (vLLM)
1. miaadv: Experiment with the influence of having a whitespace around the split point, possibly interacting with the tokenization and affecting accuracy/ tokenize-then-split or split-then-tokenize (more tokens in either of these? -> what is being split)
1. trawic: Experiment with different max_total_tokens and max_elements_per_type (assumed tradeoff between attack speed and accuracy)
1. trawic: switch to parsing instead of using regex to find syntactic and semantic identifiers
1. miaadv: Since the paper does not describe how a code file is split into a (prefix,suffix) tuple, current implementation splits it randomly between the 40% and 60% of length mark, without accounting for syntax. Experiment with different splitting methods, assumed tradeoff between prefix length and attack accuracy
1. current results are significantly below reported roc-auc, can this be due to pretraining vs fine-tuning training data detection?