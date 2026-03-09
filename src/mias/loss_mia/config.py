from dataclasses import dataclass


@dataclass
class LossMIAParams:
    sequence_length: int = 2048  # Maximum token length for input truncation
