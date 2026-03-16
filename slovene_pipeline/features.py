from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from nltk.tokenize import TweetTokenizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from word2vec_api import Word2VecAPI


def build_tfidf(
    corpus: Sequence[str],
    fitted_vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    if fitted_vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=tokenizer,
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(corpus)
    else:
        vectorizer = fitted_vectorizer
        matrix = vectorizer.transform(corpus)
    return matrix, vectorizer


def _tweet_to_mean_vector(
    tweet: str,
    w2v_model: Word2VecAPI,
    emoji_model,
    tokenizer,
    out_dim: int,
) -> np.ndarray:
    vectors = []
    for tok in tokenizer(tweet):
        if tok in w2v_model:
            vectors.append(w2v_model[tok])
        elif emoji_model is not None and tok in emoji_model:
            v = emoji_model[tok]
            if v.shape[0] == out_dim:
                vectors.append(v)
            elif v.shape[0] < out_dim:
                vectors.append(np.concatenate([v, np.zeros(out_dim - v.shape[0], dtype=np.float32)]))
            else:
                vectors.append(v[:out_dim])

    if not vectors:
        return np.zeros(out_dim, dtype=np.float32)
    return np.mean(np.stack(vectors, axis=0), axis=0)


def build_w2v_mean(
    corpus: Sequence[str],
    w2v_model: Word2VecAPI,
    emoji_model=None,
) -> np.ndarray:
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    out_dim = w2v_model.vector_size
    rows = [_tweet_to_mean_vector(tweet, w2v_model, emoji_model, tokenizer, out_dim) for tweet in corpus]
    return np.vstack(rows)


def combine_features(tfidf_x: sparse.csr_matrix, dense_x: np.ndarray) -> sparse.csr_matrix:
    dense_sparse = sparse.csr_matrix(dense_x)
    return sparse.hstack([tfidf_x, dense_sparse], format="csr")
