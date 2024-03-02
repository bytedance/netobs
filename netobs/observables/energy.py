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


class Energy(Observable):
    def shapeof(self, system) -> tuple[int, ...]:
        return ()


class EnergyEstimator(Estimator):
    observable_type = Energy

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.batch_kinetic_energy = jax.pmap(
            jax.vmap(adaptor.call_local_kinetic_energy, in_axes=(None, None, 0, None)),
            in_axes=(0, 0, 0, None),
        )
        self.batch_potential_energy = jax.pmap(
            jax.vmap(
                adaptor.call_local_potential_energy, in_axes=(None, None, 0, None)
            ),
            in_axes=(0, 0, 0, None),
        )

    def empty_val_state(self, steps: int):
        empty_values = {
            name: jnp.zeros((steps, *self.observable.shape), self.options.get("dtype"))
            for name in ("kinetic", "potential")
        }
        return (empty_values, {})

    def evaluate(self, i, params, key, electrons, system, state, aux_data):
        del i, aux_data
        return {
            "kinetic": jnp.mean(
                self.batch_kinetic_energy(params, key, electrons, system), axis=(0, 1)
            ),
            "potential": jnp.mean(
                self.batch_potential_energy(params, key, electrons, system), axis=(0, 1)
            ),
        }, state

    def digest(self, all_values, state):
        del state
        kinetic, potential = all_values["kinetic"], all_values["potential"]
        return {
            "energy": kinetic + potential,
            "potential": potential,
            "kinetic": kinetic,
        }


DEFAULT = EnergyEstimator  # Useful in CLI
