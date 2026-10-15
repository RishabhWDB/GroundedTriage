"""Near-duplicate detection shared by the split builder (data.build) and the
audit (data.audit).

Near-duplicate = cosine > NEAR_DUP between word-unigram TF-IDF vectors
(lower-cased, sublinear tf). Unigrams ignore word order, so this flags more
pairs than an n-gram measure would. Tobi-Bueck contains several LLM-reworded
versions of the same ticket, so near-duplicates are grouped (connected
components) and every group is kept inside one split.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

NEAR_DUP = 0.9


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, sublinear_tf=True, dtype=np.float32)


def similar_pairs(
    x: sparse.csr_matrix, y: sparse.csr_matrix, threshold: float = NEAR_DUP, chunk: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i in x, j in y) with cosine > threshold (rows are L2-normalised)."""
    rows, cols = [], []
    for start in range(0, x.shape[0], chunk):
        sim = (x[start : start + chunk] @ y.T).tocoo()
        keep = sim.data > threshold
        rows.append(sim.row[keep] + start)
        cols.append(sim.col[keep])
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def near_duplicate_groups(texts: list[str], threshold: float = NEAR_DUP) -> np.ndarray:
    """Component label per text: texts linked by a chain of near-duplicate pairs
    share a label. Labels are arbitrary integers; callers should key groups by
    something order-independent (e.g. the smallest row id in the group)."""
    if not texts:
        return np.zeros(0, dtype=np.int64)
    x = vectorizer().fit_transform(texts)
    i, j = similar_pairs(x, x, threshold)
    n = len(texts)
    graph = sparse.coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels
