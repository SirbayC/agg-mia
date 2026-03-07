from dataclasses import dataclass, field
from typing import List


@dataclass
class MIAAdvParams:
    # --- StarCoder2 inference ---
    max_input_tokens: int = 1024      # Truncation length for tokenizer input
    max_new_tokens: int = 128         # Max tokens to generate per sample

    # --- MLP architecture ---
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    dropout: float = 0.1
    num_classes: int = 2

    # --- MLP training ---
    batch_size: int = 4
    lr: float = 1e-3
    num_epochs: int = 25
