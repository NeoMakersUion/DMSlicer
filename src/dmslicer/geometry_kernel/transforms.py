"""
Geometry transforms.

This module provides small, self-contained transformation utilities used across
the geometry kernel.

Functions:
    rotate_z_to_vector: Construct a 3x3 rotation that maps the world Z-axis to
    a target direction vector.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np


def rotate_z_to_vector(v: Iterable[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix that maps +Z to the given direction.

    Let z = [0, 0, 1]^T. This function computes R ∈ SO(3) such that:
        R @ z ≈ u,
    where u = v / ||v|| is the normalized input vector. The construction uses
    Rodrigues' rotation formula. Special cases are handled for u ≈ z and
    u ≈ -z to avoid numerical instabilities.

    Args:
        v: Target direction vector (x, y, z). Must be non-zero.

    Returns:
        A numpy array of shape (3, 3) representing the rotation matrix.

    Raises:
        ValueError: If the input vector has near-zero norm.

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

    # If already aligned with +Z
    if np.isclose(c, 1.0):
        return np.eye(3, dtype=np.float64)

    # If opposite to +Z (180-degree rotation). Pick any axis orthogonal to Z.
    if np.isclose(c, -1.0):
        # Choose X-axis as rotation axis (could choose any orthonormal axis to z)
        k = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # 180-degree rotation: R = I + 2*K*K^T - 2I along axis k
        # Using Rodrigues with sin(theta)=0, cos(theta)=-1:
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]], dtype=np.float64)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(k, k)

    # General case: axis = z × u, angle = arccos(c)
    axis = np.cross(z, u)
    s = float(np.linalg.norm(axis))
    if s == 0.0:
        # Should not happen due to previous branches, but keep safe
        return np.eye(3, dtype=np.float64)
    k = axis / s
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float64)
    # Rodrigues: R = I + sinθ*K + (1−cosθ)*K^2
    # Here sinθ = s, cosθ = c when k is unit axis and z×u has norm s = sinθ.
    R = np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
    return R

