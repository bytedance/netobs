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

from functools import partial
from typing import Any, TypedDict

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from netobs.adaptors import NetworkAdaptor, WalkingStep
from netobs.systems.molecule import MolecularSystem


class HydrogenAuxData(TypedDict):
    mcmc_width: jnp.ndarray


@register_pytree_node_class
class SimpleHydrogen(NetworkAdaptor):
    """A simple adaptor for hydrogen atom wavefunction. Used for testing."""

    quality: float
    "How good the wavefunction is "

    def __init__(self, config: Any, args: list[str]) -> None:
        super().__init__(config, args)
        self.quality = float(config or 1)

    def restore(
        self, ckpt_file: str | None = None
    ) -> tuple[Any, jnp.ndarray, MolecularSystem, Any]:
        batch_size = int(ckpt_file or 2048)
        key = jax.random.PRNGKey(42)
        kr, kt, kp = jax.random.split(key, 3)
        shape = (1, batch_size, 1)
        r = jax.random.exponential(kr, shape)
        theta = jax.random.uniform(kt, shape, maxval=jnp.pi)
        phi = jax.random.uniform(kp, shape, maxval=2 * jnp.pi)
        data = jnp.concatenate(
            [
                r * jnp.sin(theta) * jnp.cos(phi),
                r * jnp.sin(theta) * jnp.sin(phi),
                r * jnp.cos(theta),
            ],
            axis=-1,
        )

        self.atoms = jnp.array([[0.0, 0.0, 0.0]])
        system = MolecularSystem(
            atoms=self.atoms, charges=jnp.array([1]), spins=(1, 0), ndim=3
        )
        aux_data = HydrogenAuxData(mcmc_width=jnp.array([0.6]))
        return (jnp.array([0]), data, system, aux_data)

    def call_signed_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: MolecularSystem
    ):
        del params
        # _, r_ae = calculate_r_ae(electrons, system)
        # r_ae = r_ae[0, 0, 0]  # 1st electron, 1st atom
        r_ae = jnp.linalg.norm(electrons - system["atoms"])
        return 1, -r_ae * self.quality - r_ae**2 * (1 - self.quality)

    def make_walking_step(
        self,
        steps: int,
        system: MolecularSystem,
    ) -> WalkingStep[HydrogenAuxData]:
        def walk(
            key: jnp.ndarray,
            params: jnp.ndarray,
            electrons: jnp.ndarray,
            aux_data: HydrogenAuxData,
        ) -> tuple[jnp.ndarray, HydrogenAuxData]:
            stddev = aux_data["mcmc_width"]

            def mh_update(_, data):
                x1, key, lp_1 = data
                key, subkey = jax.random.split(key)
                x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
                lp_2 = 2.0 * self.batch_network(params, x2, system)
                ratio = lp_2 - lp_1

                key, subkey = jax.random.split(key)
                cond = ratio > jnp.log(jax.random.uniform(key, shape=ratio.shape))
                x_new = jnp.where(cond[..., None], x2, x1)
                lp_new = jnp.where(cond, lp_2, lp_1)

                return x_new, key, lp_new

            logprob = 2.0 * self.batch_network(params, electrons, system)
            new_data, *_ = jax.lax.fori_loop(
                0, steps, mh_update, (electrons, key, logprob)
            )
            return new_data, aux_data

        return jax.pmap(walk, in_axes=0)

    def call_local_kinetic_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: MolecularSystem,
    ) -> jnp.ndarray:
        del key
        n = electrons.shape[0]
        eye = jnp.eye(n)
        grad_f = self.make_network_grad("electrons")

        def grad_f_closure(x):
            return grad_f(params, x, system)

        primal, dgrad_f = jax.linearize(grad_f_closure, electrons)

        return -0.5 * jax.lax.fori_loop(
            0, n, lambda i, val: val + dgrad_f(eye[i])[i], 0.0
        ) - 0.5 * jnp.sum(primal**2)

    def call_local_potential_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: MolecularSystem,
    ) -> jnp.ndarray:
        del params, key
        r_ae = jnp.linalg.norm(electrons - system["atoms"])
        # _, r_ae = calculate_r_ae(electrons, system)
        # r_ae = r_ae[0, 0, 0]  # 1st electron, 1st atom
        return -1 / r_ae

    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def batch_network(self, params, electrons, system):
        return self.call_network(params, electrons, system)


DEFAULT = SimpleHydrogen
