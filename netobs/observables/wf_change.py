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


class WFChange(Observable):
    def shapeof(self, system) -> tuple[int, ...]:
        return ()


class WFChangeEstimator(Estimator):
    r"""Change of wavefunction after atomic displacement.

    See Fig. 3 in K. Nakano et. al., Phys. Rev. B 103, L121110 (2021)

    \frac{\Braket{\psi|\psi'}}{\sqrt{\Braket{\psi|\psi}\Braket{\psi'|\psi'}}}
    = \frac{\Braket{\psi|\frac{\psi'}{\psi}|\psi}}{\Braket{\psi|\psi}}
      \sqrt{\frac{\Braket{\psi|\psi}}{\Braket{\psi|\frac{\psi'^{2}}{\psi^2}|\psi}}}
    """

    observable_type = WFChange

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.batch_network = jax.pmap(
            jax.vmap(adaptor.call_network, in_axes=(None, 0, None)),
            in_axes=(0, 0, None),
        )

    def empty_val_state(self, steps: int):
        empty_values = {
            name: jnp.zeros((steps, *self.observable.shape), self.options.get("dtype"))
            for name in ("numerator", "denominator")
        }
        return (empty_values, {})

    def evaluate(self, i, params, key, data, system, state, aux_data):
        del i, aux_data, key
        logpsi_0 = self.batch_network(params, data, system)
        new_atoms = system["atoms"].at[0, 0].add(5e-3)
        logpsi_1 = self.batch_network(params, data, {**system, "atoms": new_atoms})
        return {
            "numerator": jnp.exp(logpsi_1 - logpsi_0),
            "denominator": jnp.exp(2 * (logpsi_1 - logpsi_0)),
        }, state

    def digest(self, all_values, state):
        del state
        numerator = jnp.mean(all_values["numerator"])
        denominator = jnp.mean(all_values["denominator"])
        return {"overlap": numerator / jnp.sqrt(denominator)}


DEFAULT = WFChangeEstimator  # Useful in CLI
