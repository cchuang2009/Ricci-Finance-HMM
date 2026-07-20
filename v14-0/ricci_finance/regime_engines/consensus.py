from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class ConsensusResult:
    states: np.ndarray
    agreement: np.ndarray
    disagreement: np.ndarray


def majority_consensus(state_sequences: dict[str, np.ndarray]) -> ConsensusResult:
    if not state_sequences:
        raise ValueError("At least one state sequence is required.")

    lengths = {len(sequence) for sequence in state_sequences.values()}
    if len(lengths) != 1:
        raise ValueError("All state sequences must have equal length.")

    count = lengths.pop()
    consensus = np.empty(count, dtype=int)
    agreement = np.empty(count, dtype=float)

    for index in range(count):
        values = [int(sequence[index]) for sequence in state_sequences.values()]
        winner, winner_count = Counter(values).most_common(1)[0]
        consensus[index] = winner
        agreement[index] = winner_count / len(values)

    return ConsensusResult(
        states=consensus,
        agreement=agreement,
        disagreement=agreement < 1.0,
    )
