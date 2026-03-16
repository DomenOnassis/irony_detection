from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gensim
import numpy as np


@dataclass
class Word2VecAPI:
    """Compatibility layer with load_word2vec_format + [] + `in` API style."""

    keyed_vectors: gensim.models.KeyedVectors

    @classmethod
    def load_word2vec_format(cls, fname: str, binary: bool = True) -> "Word2VecAPI":
        kv = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=binary)
        return cls(kv)

    def __contains__(self, token: str) -> bool:
        return token in self.keyed_vectors

    def __getitem__(self, token: str) -> np.ndarray:
        return self.keyed_vectors[token]

    @property
    def vector_size(self) -> int:
        return self.keyed_vectors.vector_size


def maybe_load_emoji2vec(path: Optional[str], binary: bool = True) -> Optional[gensim.models.KeyedVectors]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return gensim.models.KeyedVectors.load_word2vec_format(str(p), binary=binary)
