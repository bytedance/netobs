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

from inspect import signature
from typing import Any, Callable

import jax
from jax import numpy as jnp


def make_kinetic_energy(
    f: Callable[[Any, jnp.ndarray, Any], Any], local_kinetic_energy: Callable
):
    """Make a duck-type-patched kinetic energy function exposing atoms.

    Args:
        f: the network function.
        local_kinetic_energy: the factory of local kinetic energy.

    Returns:
        The wrapped kinetic energy function. Signature (params, x, atoms) -> ke.
    """

    def duck_packed_network(params_and_atoms: tuple[Any, Any], x: jnp.ndarray):
        """Evaluate the wavefunction, but with duck-type-patched params."""
        params, atoms = params_and_atoms
        return f(params, x, atoms)

    ke = local_kinetic_energy(duck_packed_network)

    def wrapped_kinetic_energy(params: Any, x: jnp.ndarray, atoms: Any) -> jnp.ndarray:
        """Evaluate local kinetic energy, wrapping FermiNet's function."""
        return ke((params, atoms), x)

    return wrapped_kinetic_energy


def grad_with_system(
    f: Callable[..., jnp.ndarray],
    arg: str,
    args_before: int | None = None,
    jaxfun: Callable = jax.grad,
):
    """Make grad of functions like f(*args, electrons, system).

    The last two args must be `electrons as `system`.

    Args:
        f: function to grad.
        arg: with respect to which argument to take grad
            To grad with electrons, set `arg="electrons"`.
            To grad with things inside `system`, use the key, e.g. "atoms".
        args_before: number of arguments before "electrons".
            Leaving it empty to automatically detect.
        jaxfun: the type of gradient to take, e.g. `jax.grad`.

    Raises:
        ValueError: failing to detect the function signature

    Returns:
        The grad function.
    """
    if args_before is None:
        args_before = len(signature(f).parameters) - 2
        if args_before < 0:
            raise ValueError("Unable to determine function signature")

    if arg == "electrons":
        return jaxfun(f, argnums=args_before)

    def wrap_f(*args):
        *args, x, system = args
        return f(*args, {**system, arg: x})

    grad_local_energy = jaxfun(wrap_f, argnums=args_before + 1)

    def wrap_grad(*args):
        *args, system = args
        return grad_local_energy(*args, system[arg], system)

    return wrap_grad
