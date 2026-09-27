"""Second-moment helpers used by the alignment objective and by coherence scoring.

Ref: Methods Sec. 6.4 -- matching alone admits a representation in which metabolite
information sits on arbitrary axes, so the covariance structure of a region is matched
as well as its mean. These helpers are the numpy side of that statement; the
differentiable side lives in ``objectives.alignment``.
"""

from __future__ import annotations

import numpy as np


def centred(matrix: np.ndarray) -> np.ndarray:
    """Column-centred copy of a row-major sample matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expected a two-dimensional sample matrix")
    mean: np.ndarray = values.mean(axis=0, keepdims=True)
    result: np.ndarray = values - mean
    return result


def covariance(matrix: np.ndarray, *, ridge: float = 1e-8) -> np.ndarray:
    """Sample covariance with a ridge so that the matrix stays invertible."""
    values = centred(matrix)
    n_rows = values.shape[0]
    if n_rows < 2:
        raise ValueError("a covariance needs at least two samples")
    product: np.ndarray = values.T @ values
    cov: np.ndarray = product / float(n_rows - 1)
    scale = float(np.trace(cov)) / float(cov.shape[0])
    shrink = ridge * max(scale, 1.0)
    return cov + shrink * np.eye(cov.shape[0], dtype=np.float64)


def correlation(matrix: np.ndarray) -> np.ndarray:
    """Correlation matrix, guarded against zero-variance columns."""
    cov = covariance(matrix, ridge=0.0)
    sd: np.ndarray = np.sqrt(np.clip(np.diag(cov), 1e-24, None))
    outer: np.ndarray = np.outer(sd, sd)
    corr: np.ndarray = cov / outer
    diag: np.ndarray = np.clip(np.diag(corr), -1.0, 1.0)
    np.fill_diagonal(corr, diag)
    return corr


def gram(matrix: np.ndarray) -> np.ndarray:
    """Linear kernel over rows, normalised by the number of samples."""
    values = np.asarray(matrix, dtype=np.float64)
    product: np.ndarray = values @ values.T
    gram_matrix: np.ndarray = product / float(max(values.shape[0], 1))
    return gram_matrix


def between_group_covariance(
    matrix: np.ndarray, groups: np.ndarray, *, ridge: float = 0.0
) -> np.ndarray:
    """Covariance of group centroids.

    Ref: Methods Sec. 6.5 -- the variance penalty acts on the between-site covariance
    of the representation, so the quantity to shrink is the spread of the site means
    rather than the spread of individual rows.
    """
    values = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(groups)
    if values.shape[0] != labels.shape[0]:
        raise ValueError("group labels must align with matrix rows")
    unique = np.unique(labels)
    centroids = np.empty((unique.size, values.shape[1]), dtype=np.float64)
    for position, label in enumerate(unique):
        mask = labels == label
        block: np.ndarray = values[mask]
        centroids[position] = block.mean(axis=0)
    if centroids.shape[0] < 2:
        return np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    return covariance(centroids, ridge=ridge)


def between_group_trace(matrix: np.ndarray, groups: np.ndarray) -> float:
    """Total between-group variance of a representation."""
    cov = between_group_covariance(matrix, groups)
    return float(np.trace(cov))


def frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius distance between two matrices of equal shape."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("frobenius distance needs matching shapes")
    delta: np.ndarray = left - right
    return float(np.sqrt(np.sum(delta * delta)))


def whitening_operator(matrix: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Symmetric inverse square root of a covariance, for coherence scoring."""
    cov = covariance(matrix, ridge=eps)
    values, vectors = np.linalg.eigh(cov)
    inverse: np.ndarray = 1.0 / np.sqrt(np.clip(values, eps, None))
    operator: np.ndarray = (vectors * inverse) @ vectors.T
    return operator


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Mean squared canonical correlation between two column spaces."""
    left, _ = np.linalg.qr(np.asarray(basis_a, dtype=np.float64))
    right, _ = np.linalg.qr(np.asarray(basis_b, dtype=np.float64))
    cross: np.ndarray = left.T @ right
    singular: np.ndarray = np.linalg.svd(cross, compute_uv=False)
    clipped: np.ndarray = np.clip(singular, 0.0, 1.0)
    overlap: float = float(np.mean(clipped * clipped))
    return overlap
