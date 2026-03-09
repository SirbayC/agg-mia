from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BoWMIAParams:
    # TF-IDF
    max_features: int = 50_000       # Vocabulary size cap
    ngram_range: tuple = (1, 2)      # Unigrams + bigrams
    sublinear_tf: bool = True        # Apply log(1 + tf) scaling

    # Logistic regression
    C: float = 1.0
    max_iter: int = 1000
    solver: Literal["lbfgs", "saga", "liblinear"] = "saga"
    n_jobs: int = -1
