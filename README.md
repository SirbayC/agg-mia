# AGG-MIA

Unified pipeline for evaluating membership inference attacks (MIAs) on code language models.

Research question: given a code sample, can an attack infer whether it was in the target model pretraining distribution (member vs non-member)?

## Scope, Assumptions, and Limitations

### Scope

- Binary membership inference on code samples.
- Shared experiment driver for multiple attack families.
- Primary target models are StarCoder2 variants.

### Core assumptions

- Data is provided as two parquet-backed sets:
  - seen: (presumed) members
  - unseen: (presumed) non-members
- Loader expects at least one parquet file in each folder:
  - data/seen
  - data/unseen
- Parquet rows contain content and optionally blob_id columns.

### Target model and data sources

Target model is StarCoder2, with the following variants (and sources for seen files):

- bigcode/starcoder2-3b trained on the-stack-v2-train-smol
- bigcode/starcoder2-7b trained on the-stack-v2-train-smol
- bigcode/starcoder2-15b trained on the-stack-v2-train-full (TODO adapt dataset loading script)

For unseen files, this project relies on The Heap, deduplicated against The Stack v2 with:

- exact_duplicates_stackv2 = false
- near_duplicates_stackv2 = false

### Validity limits

- Ground truth is dataset-defined membership proxy, not direct training log membership.
- Distribution mismatch between seen and unseen can inflate separability independently of true memorization.
- Compute allocation can strongly affect throughput and practical reproducibility (for example MIG vs full GPU).
- Very small sample_fraction can produce unstable per-run metrics and noisy timing.

### Implemented attacks

| Attack | Idea | Paper | Adapted implementation (forked from original) |
|---|---|---|---|
| trawic | Reconstruction-style features + random forest | [Trained Without My Consent: Detecting Code Inclusion In Language Models Trained on Code](https://arxiv.org/abs/2402.09299) | https://github.com/SirbayC/TraWiC |
| miaadv | Perturbation features + MLP | [Effective Code Membership Inference for Code Completion Models via Adversarial Prompts](https://arxiv.org/abs/2511.15107v1) | https://github.com/SirbayC/MIA_Adv |
| loss | Negative language-model loss | [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) | https://github.com/SirbayC/thesis-code-mia-mkp |
| mkp | Min-K token log-probability score | [Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789) | https://github.com/SirbayC/thesis-code-mia-mkp |
| pac | Polarized-Augment Calibration score | [Data Contamination Calibration for Black-box LLMs](https://arxiv.org/abs/2405.11930v1) | https://github.com/AISE-TUDelft/PoisonedChalice/blob/main/Pac.py |
| bow | TF-IDF + logistic regression shift detector | N/A (control baseline) | N/A (implemented in this repo) |

#### TraWiC process

Step 1: Pre-processing (Element Extraction): TraWiC parses a target script to extract unique syntactic identifiers (e.g., variable, function, and class names) and semantic identifiers (e.g., comments, strings, and docstrings). It then masks these specific elements, splitting the surrounding code into a "prefix" and a "suffix".

Step 2: Inference (Model Querying): Using the Fill-In-the-Middle (FIM) technique, TraWiC feeds the prefix and suffix to the target Large Language Model and queries it to predict the missing masked element.

Step 3: Comparison (Hit Detection): The LLM's generated output is compared to the original masked element. Syntactic elements require an exact match to be counted as a "hit," while semantic elements use fuzzy matching (Levenshtein edit distance) to account for slight natural language variations.

Step 4: Classification (Dataset Inclusion Prediction): The normalized "hit rates" for each of the six element categories are fed into a Machine Learning classifier (such as a Random Forest). The classifier evaluates these scores and outputs a final binary decision indicating whether the script was likely included in the model's training dataset.

#### AdvPrompt-MIA (miaadv) process

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
pip install -r requirements.txt
```

Optional acceleration:

```bash
pip install --no-build-isolation flash-attn
```

If flash-attn is unavailable, use sdpa or auto attention backend.

### 2) Local run

```bash
python -u -m src.main \
  --mia trawic \
  --model bigcode/starcoder2-3b \
  --attn_implementation auto \
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
- --attn_implementation: auto | flash_attention_2 | sdpa | eager
- --sample_fraction: fraction loaded from each split
- --output_dir: destination for predictions and metrics

### 3) DelftBlue run

Use:

```bash
./submit_delftblue.sh
```

run_delftblue.sh controls cluster-side settings such as:

- MIA, LLM, SAMPLE_FRACTION
- ATTN_IMPL
- INSTALL_FLASH_ATTN

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
python -m src.results.scripts.unify_metrics <root_folder>
```

Recompute metrics from saved predictions:

```bash
python -m src.results.scripts.recompute_metrics <predictions_csv>
```

---

## Data Preparation

To download (smol) files:

1. `module load miniconda3 && conda activate /scratch/cosminvasilesc/AGG-MIA/ENV`
2. Set `SWH_TOKEN` in the environment using: https://archive.softwareheritage.org/oidc/profile/#tokens
3. Run from repo root: `python src/datasets/download_seen.py`
4. Run from repo root: `python src/datasets/download_unseen.py`
5. Run from repo root: `python src/datasets/show_parquet_samples.py`

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
tail -f "$(ls -t | head -n 1)"
```

4. Copy outputs to local machine and clear remote run outputs with:

```bash
scp -r cosminvasilesc@login.delftblue.tudelft.nl:/scratch/cosminvasilesc/AGG-MIA/outputs/runs/* "C:\Coding_projects\AGG_MIA\output_runs" && ssh delftblue "rm -rf /scratch/cosminvasilesc/AGG-MIA/outputs/runs/*"
```

### Project map

```text
AGG_MIA/
├── analysis/                      # Research (preliminary) results
├── data/
│   ├── seen/
│   └── unseen/
├── src/
│   ├── main.py                    # CLI and experiment orchestration
│   ├── datasets/                  # Download and loading pipeline
│   ├── models/
│   │   └── loader.py              # Tokenizer/model loading, dtype, attention backend
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