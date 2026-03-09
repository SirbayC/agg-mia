from dataclasses import dataclass


@dataclass
class MKPMIAParams:
    k: float = 0.1                    # Fraction of lowest-probability tokens to average
    sequence_length: int = 2048       # Max token length (truncation) or sliding window size
    use_sliding_window: bool = False  # If True, use non-overlapping sliding window instead of truncation
