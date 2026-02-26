AdvPrompt-MIA
===============

AdvPrompt-MIA is a membership-inference attack (MIA) framework for code-completion models. It applies a suite of semantics-preserving code perturbations, measures how a victim model’s outputs change (via text similarity and perplexity), aggregates these into feature vectors, and trains a binary MLP classifier to decide whether an input was seen during training.

Features
--------
- Five categories of adversarial prompts (11 total variants per input):  
  • Dead control branches (IDC)  
  • Dead loops (IDL)  
  • Unused variable declarations (IRV)  
  • Identifier renaming (VR)  
  • Debug print statements (IDP)  
- For each perturbed input, compute:  
  • Textual similarity vs. ground-truth completion (CodeBERT embeddings + cosine similarity)  
  • Relative perplexity change (CodeGPT perplexity, normalized)  
- Aggregate 27-dimensional feature vectors: similarity for original + each perturbed output (mean & std), perplexity changes for each (mean & std)  
- Train a 3-layer MLP (ReLU + dropout) to predict membership  

Installation
------------
Requirements:
- Python 3.8+  
- pip  

Install dependencies:
```
pip install torch transformers numpy scikit-learn
```

Repository Structure
-------------------
```
.
├── dataset/
│   └── humaneval/                # HumanEval JSON files
│   └── APPS/                     # APPS JSON files
│
└── baseline/                     # Four baseline MIA implementations
│   ├── BUGOTCHA/
│   ├── GT-Match/
│   ├── PPL-Rank/
│   └── BUZZER-b/
│
├── llm_finetune/
│   └── pipline.sh                # Script to fine-tune victim models
│   └──...
│
├── llm_inference/
│   └── infer.sh                  # Script to run victim model inference
│   └──...  
│
├── entry.sh                      # Orchestrates the full pipeline
├── perturb.py                    # Generates perturbed code samples
├── classifier.py                 # Extracts features and trains/evaluates MLP
└── README.md                     # This file
```

Full Pipeline (entry.sh)
------------------------
To reproduce experiments on HumanEval with CodeLlama-7B:

```bash
# 1. Fine-tune victim models
cd finetune
./pipline.sh
cd ..

# 2. Generate perturbed training samples
python perturb.py \
  --input_dir dataset/humaneval/ \
  --input_file train_victim_humaneval.json \
  --output_file train_victim.json

# 3. Generate perturbed test samples
python perturb.py \
  --input_dir dataset/humaneval/ \
  --input_file test_victim_humaneval.json \
  --output_file test_victim.json

# 4. Run victim model inference on original and perturbed inputs
cd llm_inference
./infer.sh
cd ..

# 5. Extract features, train and evaluate the MLP attack
python classifier.py \
  --input_dir "dataset/humaneval/" \
  --true_file "train_CodeLlama-7b-hf_victim_infer.txt" \
  --false_file "test_CodeLlama-7b-hf_victim_infer.txt" \
  --true_gt_file "train_victim.json" \
  --false_gt_file "test_victim.json" \
  --feature_path "feature.npz" \
  --n_samples_per_class 40 \
  --global_random_seed 725982103 \
  --random_state 876886030 \
  --random_state_test 1478597768 \
  --dropout 0.1 \
  --batch_size 4 \
  --lr 1e-3 \
  --num_epochs 25 \
  --hidden_dims 512 512 512
```

Perturbation Module (perturb.py)
-------------------------------
Generates semantically equivalent variants for each code sample.

Quick Start
1. Prepare `data/input.jsonl`, each line:
   ```json
   {
     "id": "0",
     "input": "<s> def foo(x):\n    return x + 1",
     "gt": "def foo(x): return x + 1"
   }
   ```
2. Run:
   ```
   python perturb.py \
     --input_dir ./data/ \
     --input_file input.jsonl \
     --output_file perturbed.jsonl
   ```
3. Output: `data/perturbed.jsonl` containing original + 11 variants.

Command-Line Arguments
- `--input_dir`: Directory of input file  
- `--input_file`: Input JSONL filename  
- `--output_file`: Output JSONL filename  

Perturbation Types

- Inserting Dead Control Branches (IDC)
    `false_if_fixed`: Insert a fixed `if False:` or `if "key"!="key":` statement
    `false_if_random`: Use the same `if` condition but randomly select the `if` body from existing code lines

- Inserting Redundant Variable Declarations (IRV)
    `unused_var_existing`: Declare a new variable initialized with an existing variable or constant
    `unused_var_random`: Declare a new variable initialized with a random constant

- Variable and Method Renaming (VR)
    `rename_var`: Randomly rename a variable to `var_<rand>`
    `rename_method`: Randomly rename a method to `method_<rand>`

- Inserting Debug Print Statements (IDP)
    `print_enter`: Insert `print("Debug: Entering method ...")` at the start of a method body
    `print_variable`: Insert `print("Debug: Variable ...")` immediately after the first occurrence of a variable definition

- Inserting Dead Loops (IDL)
    `false_loop_for`: Insert a `for` loop that never executes
    `false_loop_while`: Insert a `while` loop that never executes

Classifier Module (classifier.py)
-------------------------------
Implements feature extraction and MLP training/evaluation.

Dependencies
```
pip install torch transformers numpy scikit-learn
```

Quick Start
1. Prepare in `./dataset/`:
   - `train_codes.txt`, `train_labels.json`  
   - `test_codes.txt`, `test_labels.json`
2. First run (features will be saved to `feature.npz`):
   ```bash
   python classifier.py \
     --input_dir "./dataset/" \
     --true_file "train_codes.txt" \
     --false_file "test_codes.txt" \
     --true_gt_file "train_labels.json" \
     --false_gt_file "test_labels.json" \
     --feature_path "feature.npz" \
     --n_samples_per_class 40 \
     --global_random_seed 42 \
     --random_state 123 \
     --random_state_test 456 \
     --dropout 0.1 \
     --batch_size 4 \
     --lr 1e-3 \
     --num_epochs 25 \
     --hidden_dims 512 512 512
   ```
3. The script will extract features, train the MLP, and print Loss, Accuracy, TPR, FPR, AUC.

Arguments
- `--input_dir`: Directory with model outputs & GT files  
- `--true_file`, `--false_file`: Model outputs on member/non-member samples  
- `--true_gt_file`, `--false_gt_file`: Ground-truth JSON for member/non-member  
- `--feature_path`: Path to `.npz` feature file  
- `--n_samples_per_class`: Number of samples per class  
- `--global_random_seed`, `--random_state`, `--random_state_test`  
- `--dropout`, `--batch_size`, `--lr`, `--num_epochs`, `--hidden_dims`

Sample Output
```
Saved features to feature.npz: X.shape=(240, 14), y.shape=(240,)

Final Test Loss: 0.1054, Test Accuracy: 90.00%
=== Binary classification metrics ===
TPR (Recall)=88.00%   FPR=8.00%
AUC: 0.9623
```

Baselines
---------
Four implemented under `baseline/`:
- BUGOTCHA  
- GT-Match  
- PPL-Rank  
- BUZZER-b  

Supported Victim Models:CodeLlama 7B, Deepseek-Coder 7B, StarCoder2 7B, Phi-2 2.7B,WizardCoder 7B 

Victim Model Fine-tuning Hyperparameters
----------------------------------------

To ensure reproducibility, we include full victim-model fine-tuning configs in our public replication package. Main settings:

- Epochs: 5  
- Learning rate: 8e-5  
- Model-specific tuning:  
  - Phi-2 (2.7B): full fine-tuning  
  - Other 7B models: LoRA parameter-efficient fine-tuning  
    - LoRA hyperparameters: rank r = 16, scaling factor α = 32, dropout = 0.1  
- Batch size: 2 (with gradient accumulation over 16 steps)  
- Optimizer: Adam  
  - weight decay = 1e-2  
  - epsilon = 1e-8  
  - gradient clipping (max norm) = 1  
- Hardware: two NVIDIA A6000 GPUs (48 GB each)  
- Hyperparameter search: random search over alternative configs confirmed these settings yield the best functional correctness across all victim models  

For complete scripts and parameter details, see `llm_finetune/pipeline.sh` in the replication package.


Experimental Results
--------------------
AdvPrompt-MIA outperforms all baselines and generalizes across models.

Performance of AdvPrompt-MIA on CodeLlama 7B:
- HumanEval: TPR=85%, FPR=5%,  AUC=0.97  
- APPS:       TPR=90%, FPR=14%, AUC=0.95  

Performance of AdvPrompt-MIA on other models (HumanEval):
- Deepseek-Coder 7B: TPR=80%, FPR=10%, AUC=0.91  
- StarCoder2 7B:      TPR=75%, FPR=10%, AUC=0.92  
- Phi-2 2.7B:         TPR=90%, FPR=25%, AUC=0.92  
- WizardCoder 7B:     TPR=75%, FPR=5%,  AUC=0.95  

Performance of AdvPrompt-MIA and baselines on other models (APPS):  
- Deepseek-Coder 7B  
  - AdvPrompt-MIA: TPR=82%, FPR=19%, AUC=0.89  
  - GOTCHA:        TPR=61%, FPR=43%, AUC=0.63  
  - GT-Match:      TPR=49%, FPR=51%, AUC=0.50  
  - PPL-Rank:      TPR=52%, FPR=49%, AUC=0.51  
  - BUZZER-b:      TPR=52%, FPR=48%, AUC=0.51  

- StarCoder2 7B  
  - AdvPrompt-MIA: TPR=84%, FPR=16%, AUC=0.91  
  - GOTCHA:        TPR=65%, FPR=38%, AUC=0.72  
  - GT-Match:      TPR=51%, FPR=50%, AUC=0.49  
  - PPL-Rank:      TPR=53%, FPR=47%, AUC=0.53  
  - BUZZER-b:      TPR=54%, FPR=46%, AUC=0.54  

- Phi-2 2.7B  
  - AdvPrompt-MIA: TPR=79%, FPR=21%, AUC=0.85  
  - GOTCHA:        TPR=62%, FPR=34%, AUC=0.72  
  - GT-Match:      TPR=50%, FPR=50%, AUC=0.50  
  - PPL-Rank:      TPR=53%, FPR=47%, AUC=0.50  
  - BUZZER-b:      TPR=50%, FPR=50%, AUC=0.50  

- WizardCoder 7B  
  - AdvPrompt-MIA: TPR=76%, FPR=25%, AUC=0.81  
  - GOTCHA:        TPR=63%, FPR=34%, AUC=0.71  
  - GT-Match:      TPR=49%, FPR=51%, AUC=0.50  
  - PPL-Rank:      TPR=52%, FPR=48%, AUC=0.52  
  - BUZZER-b:      TPR=52%, FPR=48%, AUC=0.51  

Citation
--------
“Effective Code Membership Inference for Code Completion Models via Adversarial Prompts”  2025