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

import jax
from jax import numpy as jnp

from netobs.observables import Estimator, Observable
from netobs.systems import solid
from netobs.systems.solid import SolidSystem


class Polarization(Observable[SolidSystem]):
    def shapeof(self, system) -> tuple[int, ...]:
        return (system["ndim"],)


class PolarizationEstimator(Estimator[SolidSystem]):
    observable_type = Polarization

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.antithetic = estimator_options.get("antithetic", False)
        self.batch_network = jax.pmap(
            jax.vmap(adaptor.call_network, in_axes=(None, 0, None)),
            in_axes=(0, 0, None),
        )
        self.recvec = solid.recvec(system)
        self.latvec = system["latvec"]
        atom_sum = jnp.sum(system["charges"][:, None] * system["atoms"], axis=0)
        self.atom_phase = jnp.exp(-1j * jnp.dot(self.recvec, atom_sum))

    def empty_val_state(self, steps: int):
        dtype = self.options.get("dtype", "complex64")
        return {"phase": jnp.zeros((steps, *self.observable.shape), dtype)}, {}

    def batch_phase(self, x: jnp.ndarray) -> jnp.ndarray:
        electrons = jnp.reshape(x, (*x.shape[:2], -1, self.system["ndim"]))
        elec_exponent = jnp.sum(
            jnp.einsum("ij,dbkj->dbik", self.recvec, electrons), axis=-1
        )
        return jnp.exp(1j * elec_exponent)

    def batch_phase_antithetic(
        self, params: jnp.ndarray, x: jnp.ndarray, system: SolidSystem
    ) -> jnp.ndarray:
        local_phase = self.batch_phase(x)
        mirrored_local_phase = 1 / local_phase
        psi0 = self.batch_network(params, x, system)
        psi1 = self.batch_network(params, -x, system)
        mirrored_weight = jnp.abs(jnp.exp((psi1 - psi0) * 2))[..., None]
        return (local_phase + mirrored_weight * mirrored_local_phase) / 2

    def evaluate(self, i, params, key, data, system, state, aux_data):
        del i, aux_data, key
        if self.antithetic:
            phase = self.batch_phase_antithetic(params, data, system)
        else:
            phase = self.batch_phase(data)
        return {"phase": phase}, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state
        mean_phase = jnp.mean(all_values["phase"], axis=0) * self.atom_phase
        mean_p = -1 * jnp.dot(
            self.latvec.T, jnp.imag(jnp.log(mean_phase)) / (2 * jnp.pi)
        )
        return {"polarization": mean_p}


DEFAULT = PolarizationEstimator
