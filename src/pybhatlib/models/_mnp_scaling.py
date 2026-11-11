"""Scale coordinates for identified differenced MNP kernel errors."""

import numpy as np

from pybhatlib.utils._logistic import logitmod
from pybhatlib.utils._safe_reparam import safe_exp


def validate_scaling(scaling):
    if scaling not in ("first", "sumsq"):
        raise ValueError("scaling must be 'first' or 'sumsq'")


def kernel_scales(free, scaling):
    """Decode dim-1 optimizer coordinates into dim natural scales.

    First-scale coordinates are log scales; sumsq coordinates are logits of
    squared scales relative to the first squared scale.
    """
    validate_scaling(scaling)
    free = np.asarray(free, dtype=np.float64)
    if scaling == "first":
        return np.concatenate([[1.0], safe_exp(free)])
    return np.sqrt(logitmod(np.concatenate([[0.0], free])))


def kernel_scale_jacobian(scales, scaling):
    """Return d scale[k] / d free[j], with free parameters in rows."""
    validate_scaling(scaling)
    scales = np.asarray(scales, dtype=np.float64)
    basis = np.eye(len(scales))[1:]
    if scaling == "first":
        return basis * scales[None, :]
    return 0.5 * (basis - np.square(scales[1:])[:, None]) * scales[None, :]


def direct_kernel_scales(free, scaling):
    """Restore the omitted reference scale from natural free scales."""
    validate_scaling(scaling)
    free = np.asarray(free, dtype=np.float64)
    if scaling == "first":
        return np.concatenate([[1.0], free])
    reference = 1.0 - float(np.square(free).sum())
    if reference <= 0.0:
        raise ValueError("sumsq kernel scales must have sum of free squared scales < 1")
    return np.concatenate([[np.sqrt(reference)], free])


def encode_kernel_scales(scales, scaling):
    """Encode natural scales satisfying the selected normalization."""
    validate_scaling(scaling)
    scales = np.asarray(scales, dtype=np.float64)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("kernel scales must be finite and positive")
    if scaling == "first":
        return np.log(scales[1:])
    return 2.0 * (np.log(scales[1:]) - np.log(scales[0]))
