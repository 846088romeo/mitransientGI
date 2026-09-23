"""
Utilities for ghost imaging simulation and reconstruction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _validate_patterns(patterns: np.ndarray) -> Tuple[np.ndarray, int, int]:
    patterns = np.asarray(patterns, dtype=np.float64)
    if patterns.ndim != 3:
        raise ValueError(
            "patterns must have shape (num_patterns, height, width)"
        )
    num_patterns, height, width = patterns.shape
    if num_patterns == 0 or height == 0 or width == 0:
        raise ValueError("patterns must be non-empty")
    return patterns, height, width


def simulate_bucket_measurements(
    patterns: np.ndarray,
    target: np.ndarray,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate bucket measurements for a set of illumination patterns.
    :param patterns: pattern stack with shape ``(num_patterns, height, width)``
    :param target: target image with shape ``(height, width)``
    :param noise_std: optional Gaussian noise standard deviation
    :param seed: optional RNG seed for reproducibility
    """
    patterns, height, width = _validate_patterns(patterns)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (height, width):
        raise ValueError("target shape must match pattern spatial dimensions")

    measurements = np.tensordot(patterns, target, axes=((1, 2), (0, 1)))
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        measurements = measurements + rng.normal(
            loc=0.0, scale=noise_std, size=measurements.shape
        )
    return measurements


def reconstruct_ghost_image(
    patterns: np.ndarray,
    measurements: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Reconstruct a ghost image from patterns and bucket measurements.
    This uses a covariance-based correlation reconstruction:
    ``I(x, y) = <(B - <B>) (P(x, y) - <P(x, y)>)>``.
    """
    patterns, _, _ = _validate_patterns(patterns)
    measurements = np.asarray(measurements, dtype=np.float64).reshape(-1)
    if measurements.shape[0] != patterns.shape[0]:
        raise ValueError(
            "measurements length must match number of patterns"
        )

    centered_measurements = measurements - np.mean(measurements)
    centered_patterns = patterns - np.mean(patterns, axis=0, keepdims=True)
    reconstruction = np.tensordot(
        centered_measurements, centered_patterns, axes=(0, 0)
    ) / measurements.shape[0]

    if normalize:
        max_abs = np.max(np.abs(reconstruction))
        if max_abs > 0:
            reconstruction = reconstruction / max_abs
    return reconstruction


def reconstruct_transient_ghost_image(
    patterns: np.ndarray,
    transient_measurements: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Reconstruct a transient ghost image sequence from time-resolved buckets.
    :param patterns: pattern stack with shape ``(num_patterns, height, width)``
    :param transient_measurements: bucket tensor with shape
        ``(num_patterns, temporal_bins)``
    :returns: reconstructed tensor with shape ``(height, width, temporal_bins)``
    """
    patterns, height, width = _validate_patterns(patterns)
    transient_measurements = np.asarray(
        transient_measurements, dtype=np.float64
    )
    if transient_measurements.ndim != 2:
        raise ValueError(
            "transient_measurements must have shape (num_patterns, temporal_bins)"
        )
    if transient_measurements.shape[0] != patterns.shape[0]:
        raise ValueError(
            "transient_measurements first dimension must match number of patterns"
        )

    temporal_bins = transient_measurements.shape[1]
    output = np.zeros((height, width, temporal_bins), dtype=np.float64)
    for t in range(temporal_bins):
        output[:, :, t] = reconstruct_ghost_image(
            patterns=patterns,
            measurements=transient_measurements[:, t],
            normalize=normalize,
        )
    return output
