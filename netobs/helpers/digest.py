# Copyright 2022-2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from jax import numpy as jnp


def iqr_clip_real(x: jnp.ndarray, iqr_range: float = 3.0) -> jnp.ndarray:
    """Clip outliers from real-valed array with interquartile range (IQR).

    Args:
        x: array to be clipped (must be real)
        iqr_range: how many times the data outside the IQR will be clipped

    Returns:
        Clipped array
    """
    q1 = jnp.quantile(x, 0.25, axis=0, keepdims=True)
    q3 = jnp.quantile(x, 0.75, axis=0, keepdims=True)
    iqr = q3 - q1
    cutoff = iqr_range * iqr
    return jnp.clip(x, q1 - cutoff, q3 + cutoff)


def iqr_clip(x: jnp.ndarray, iqr_range: float = 3.0) -> jnp.ndarray:
    """Clip outliers with interquartile range (IQR).

    Args:
        x: array to be clipped
        iqr_range: how many times the data outside the IQR will be clipped

    Returns:
        Clipped array
    """
    if jnp.iscomplexobj(x):
        return iqr_clip_real(x.real, iqr_range) + 1j * iqr_clip_real(x.imag, iqr_range)
    return iqr_clip_real(x, iqr_range)


def robust_mean(x: jnp.ndarray, iqr_range: float = 3.0) -> jnp.ndarray:
    """Take the mean after clipping.

    Args:
        x: input array
        iqr_range: how many times the data outside the IQR will be clipped

    Returns:
        Array averaged in the first dim.
    """
    clipped = iqr_clip(x, iqr_range)
    return jnp.nanmean(clipped, axis=0)


def robust_mean_std(
    x: jnp.ndarray, iqr_range: float = 3.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Take the mean and estimate stderr after clipping.

    Args:
        x: input array
        iqr_range: how many times the data outside the IQR will be clipped

    Returns:
        A tuple of
        - Array averaged in the first dim.
        - Std of the mean.
    """
    clipped = iqr_clip(x, iqr_range)
    if jnp.iscomplexobj(clipped):
        std = jnp.nanstd(clipped.real, axis=0) + 1j * jnp.nanstd(clipped.imag, axis=0)
    else:
        std = jnp.nanstd(clipped, axis=0)
    return (jnp.nanmean(clipped, axis=0), std / jnp.sqrt(clipped.shape[0]))


def weighted_sum(a: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Taking average, but align array and weights.

    Ftuple[jnp.ndarray, jnp.ndarray]:tuple[jnp.ndarray, jnp.ndarray]:or example, (ijkl) * (ij) -> (kl).

    Args:
        a: JAX array.
        weights: the weight for averaging.

    Returns:
        sum of weighted array
    """
    axis = tuple(range(len(weights.shape)))
    weights = weights.reshape(weights.shape + (1,) * (a.ndim - weights.ndim))
    return jnp.mean(a * weights, axis=axis)
