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

from typing import Any, Callable, Sequence, TypeVar, cast

import jax
from jax import numpy as jnp

F = TypeVar("F", bound=Callable[..., Any])


def chunk_vmap(func: F, in_axes: Sequence[None | int], chunks: int | str | None) -> F:
    """VMap a function in chunks.

    Only supports `vmap`ping along the 0th axis for one argument.

    Args:
        func: Function to be vmapped.
        in_axes: VMap input axes. Only support `None` or `0`.
        chunks: number of chunks to be split.

    Returns:
        Chunk vmapped function.
    """
    # For convenience reading from options
    if isinstance(chunks, str):
        chunks = int(chunks)
    if chunks == 1 or chunks is None:
        return jax.vmap(func, in_axes=in_axes)

    vmapped_func = jax.vmap(func, in_axes=in_axes)
    argnum = list(in_axes).index(0)

    def chunked_func(*args):
        data = args[argnum]
        data = data.reshape((chunks, -1, data.shape[-1]))

        def func_per_chunk(_, x):
            return None, vmapped_func(*args[:argnum], x, *args[argnum + 1 :])

        result = jax.lax.scan(func_per_chunk, None, data)[1]
        if isinstance(result, tuple):
            return tuple(jnp.reshape(r, (-1, *r.shape[2:])) for r in result)
        else:
            return jnp.reshape(result, (-1, *result.shape[2:]))

    return cast(F, chunked_func)
