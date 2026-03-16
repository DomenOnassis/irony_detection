from __future__ import annotations

from typing import Sequence

import numpy as np


def build_api_features(corpus: Sequence[str], task: str) -> np.ndarray:
    """Placeholder for external API-derived features.

    Keep this function signature and replace internals later with your current APIs
    (CoreNLP, handcrafted sentiment/intensity features, topic chunks, etc.).
    """
    _ = task
    return np.zeros((len(corpus), 0), dtype=np.float32)
