import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Lazy-loaded feature models — populated by init_feature_models() called from AdvMIA.__init__.
# Nothing is loaded at import time so other MIA methods don't consume VRAM unnecessarily.
_CODEBERT_NAME = "microsoft/codebert-base"
_CODEGPT_NAME = "microsoft/CodeGPT-small-py"

device = None
model = None
tokenizer = None
codegpt_tokenizer = None
codegpt_model = None


def init_feature_models(target_device: torch.device = None) -> None:
    """Load CodeBERT and CodeGPT into memory. Safe to call multiple times."""
    global device, model, tokenizer, codegpt_tokenizer, codegpt_model
    if model is not None:
        return  # already initialised
    device = target_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(_CODEBERT_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(_CODEBERT_NAME)
    codegpt_tokenizer = GPT2Tokenizer.from_pretrained(_CODEGPT_NAME)
    codegpt_model = GPT2LMHeadModel.from_pretrained(_CODEGPT_NAME).to(device)
    codegpt_model.eval()


class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        """
        hidden_dims: a list of the width of every hidden layers
        e.g. [128, 64, 32] represent three hidden layers, with widths of 128, 64 and 32
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        # Last layer mapped to output
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_mlp(model: nn.Module,
              X_train: torch.Tensor,
              y_train: torch.Tensor,
              X_val: torch.Tensor = None,
              y_val: torch.Tensor = None,
              device: torch.device = None,
              batch_size: int = 2,
              lr: float = 1e-3,
              num_epochs: int = 50):
    """
    basic training function, with optional evaluating dataset output.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # === instantiate LR scheduler ===
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(loader.dataset)

        msg = f"Epoch {epoch}/{num_epochs}  train_loss={avg_loss:.4f}"
        if X_val is not None and y_val is not None:
            val_loss, val_acc = evaluate_mlp(model, X_val, y_val, device)
            msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4%}"
        # print(msg)
        # === update learning rate fater every epoch===
        scheduler.step()


def evaluate_mlp(model: nn.Module,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 device: torch.device = None):
    """
    calculate average cross entropy loss and accuracy
    """
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        loss = F.nll_loss(probs.log(), y).item()
        preds = probs.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return loss, acc


def extract_embeddings(text):
    if model is None or tokenizer is None:
        raise RuntimeError("Feature models not initialised — call init_feature_models() first.")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # calculate with GPU
    return outputs.squeeze(0).cpu().numpy()  # convert to CPU then convert to NumPy


def compute_similarity(x, y):
    # print(x, y)
    sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    v_x = extract_embeddings(x)
    v_y = extract_embeddings(y)
    return sim(v_x, v_y)


def calculate_perplexity(text, model, tokenizer):
    """
    exp(loss)
    """
    if text == "":
        return 100.0
    max_position_embeddings = model.config.max_position_embeddings
    inputs = tokenizer.encode(text)
    if len(text) > max_position_embeddings:
        inputs = inputs[-max_position_embeddings:]
    input_ids = torch.tensor(inputs).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        # 2) check input_ids
        if input_ids.numel() == 0:
            raise ValueError("The input has been truncated to an empty sequence! Please check the max_length or the text itself.")
        if torch.isnan(input_ids).any():
            raise ValueError("input_ids contains Nan!")

        # 3) forward calculation
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # HF new version uses output.loss and outputs.logits
        logits = outputs.logits

        # 4) checkout loss / logits
        if torch.isnan(loss) or torch.isinf(loss):
            return 1e6
            print(f"‼️ loss exception：{loss.item()}")
            # print logits distribution
            print(f"logits min/max: {logits.min().item()}/{logits.max().item()}")
            raise ValueError("NaN/Inf appear when calculate loss!")

        # 5) calculate perplexity
        ppl = torch.exp(loss)
        if torch.isinf(ppl) or torch.isnan(ppl):
            print(f"‼️ perplexity exception：exp({loss.item()}) = {ppl.item()}")
            raise ValueError("exp(loss) gets NaN/Inf！")

        return ppl.cpu().item()

    outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss).cpu().numpy()


def compute_features(y, y_pred, y_preds_perturbed):
    """Compute features for MIA classification."""
    sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    v_y = extract_embeddings(y)
    v_y_pred = extract_embeddings(y_pred)
    v_y_preds_perturbed = [extract_embeddings(y_p) for y_p in y_preds_perturbed]

    similarities = [sim(v_y, v_y_p) for v_y_p in v_y_preds_perturbed]

    base_sim = compute_similarity(y, y_pred)
    perturbed_sims = [compute_similarity(y, y_p) for y_p in y_preds_perturbed]
    # perturbed_sims = [compute_similarity(y_pred, y_p) for y_p in y_preds_perturbed]
    all_sims = [base_sim] + perturbed_sims
    sim_stats = [
        np.mean(all_sims),    # average
        np.std(all_sims)     # standard deviation
    ]
    base_ppl = calculate_perplexity(y_pred, codegpt_model, codegpt_tokenizer)
    perturbed_ppls = [calculate_perplexity(py, codegpt_model, codegpt_tokenizer) for py in y_preds_perturbed]
    # all_ppls = [base_ppl] + perturbed_ppls
    all_ppls = perturbed_ppls
        
    all_ppls = [(ppl - base_ppl) / base_ppl for ppl in all_ppls]

    ppl_stats = [
        np.mean(all_ppls),    # average
        np.std(all_ppls)     # standard deviation
    ]

    feature_vector = (
        all_sims +      
        sim_stats 
        +     
        all_ppls +
        ppl_stats
    )

    return feature_vector
