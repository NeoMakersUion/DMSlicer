"""
Rotation utilities.

This module provides small, self-contained rotation helpers built on NumPy.

Functions:
    rotate_z_to_vector: Construct a 3×3 rotation matrix that maps +Z to a
    target direction vector.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np


def rotate_z_to_vector(v: Iterable[float]) -> np.ndarray:
    """Return a 3×3 rotation matrix that maps +Z to the given direction.

    Given z = [0, 0, 1]^T, computes R ∈ SO(3) such that:
        R @ z ≈ u,
    where u = v / ||v|| is the normalized input vector. Uses Rodrigues' rotation
    formula with special handling for u ≈ z and u ≈ -z to avoid instabilities.

    Args:
        v: Target direction vector (x, y, z). Must be non-zero and finite.

    Returns:
        A numpy array of shape (3, 3) representing the rotation matrix.

    Raises:
        ValueError: If the input vector has near-zero norm or non-finite values.

    Example:
        >>> import numpy as np
        >>> R = rotate_z_to_vector([0, 0, 1])
        >>> np.allclose(R, np.eye(3))
        True
        >>> u = np.array([1.0, 1.0, 1.0])
        >>> R = rotate_z_to_vector(u)
        >>> z = np.array([0.0, 0.0, 1.0])
        >>> target = u / np.linalg.norm(u)
        >>> np.allclose(R @ z, target)
        True
    """
    v = np.asarray(v, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(v)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Input vector must be non-zero and finite")
    u = v / norm

    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(np.clip(np.dot(z, u), -1.0, 1.0))

    if np.isclose(c, 1.0):
        return np.eye(3, dtype=np.float64)

    if np.isclose(c, -1.0):
        k = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(k, k)

    axis = np.cross(z, u)
    s = float(np.linalg.norm(axis))
    if s == 0.0:
        return np.eye(3, dtype=np.float64)
    k = axis / s
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
    return R

