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

from jax import numpy as jnp

from netobs.systems import System


class MolecularSystem(System):
    charges: jnp.ndarray
    "(natom,) Nuclear charges of the atoms."

    spins: tuple[int, int]


def calculate_r_ae(
    electrons: jnp.ndarray, system: MolecularSystem
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the electron-atom distance.

    ae[i, j] is a vector points from atom j to electron i.

    Args:
        electrons: (..., nelec*ndim)
        system: Molecular system.

    Returns:
        A tuple of ae and r_ae.
            ae: atom-electron vector. Shape (..., nelectron, natom, 3).
            r_ae: atom-electron distance. Shape (..., nelectron, natom, 1).
    """
    shape = (*electrons.shape[:-1], -1, 1, 3)
    ae = jnp.reshape(electrons, shape) - system["atoms"]
    r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
    return ae, r_ae
