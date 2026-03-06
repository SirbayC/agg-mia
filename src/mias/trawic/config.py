from dataclasses import dataclass
from typing import Literal

@dataclass
class TraWiCParams:
    syntactic_threshold: int = 100  # Exact match for syntax elements
    semantic_threshold: int = 20    # Fuzzy match threshold for semantic elements

    n_estimators: int = 100
    max_depth: int = 20
    max_features: Literal['sqrt', 'log2'] = "sqrt"
    criterion: Literal['gini', 'entropy', 'log_loss'] = "gini"
    random_state: int = 42
    n_jobs: int = -1
    
    max_context: int = 3000  # Maximum context characters for infill prompts
    max_generated_tokens: int = 50  # Maximum tokens to generate during infill
    max_total_tokens: int = 2048  # Maximum total tokens (input + output) for model
    temperature: float = 0.2  # Sampling temperature for generation
    top_p: float = 0.95  # Nucleus sampling parameter
