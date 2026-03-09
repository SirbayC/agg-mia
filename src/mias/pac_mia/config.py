from dataclasses import dataclass


@dataclass
class PACMIAParams:
    sequence_length: int = 4096  # Max token length for input truncation
    near_count: int = 30         # Number of highest-prob tokens for the "near" pole
    far_count: int = 5           # Number of lowest-prob tokens for the "far" pole
    m_ratio: float = 0.3         # Fraction of tokens to swap when generating mutants
    n_samples: int = 5           # Number of mutated adjacent samples per text

