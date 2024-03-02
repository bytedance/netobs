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

from typing import Any

from jax import numpy as jnp

from netobs.observables import Estimator, Observable


class Density(Observable):
    def shapeof(self, system) -> tuple[int, ...]:
        return ()


class DensityEstimator(Estimator):
    observable_type = Density

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)

        bins = self.options.get("bins", 50)
        if isinstance(bins, int):
            bins = [bins] * system["ndim"]
        self.hist_bins = jnp.array(bins)

        hist_range = self.options.get("range", None)
        if hist_range is None:
            self.hist_range = None
        else:
            self.hist_range = jnp.array(hist_range)

    def empty_val_state(
        self, steps: int
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del steps
        return {}, {
            "map": jnp.zeros(self.hist_bins),
            "range": self.hist_range,
        }

    def evaluate(
        self, i, params, key, data, system, state, aux_data
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del i, params, aux_data, key
        x = jnp.reshape(data, (-1, system["ndim"]))
        if state["range"] is None:
            max_coord = jnp.amax(x, axis=0)
            min_coord = jnp.amin(x, axis=0)
            state["range"] = jnp.c_[min_coord, max_coord]
        state["map"] += jnp.histogramdd(x, self.hist_bins, state["range"])[0]
        return {}, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del all_values, state
        return {}


DEFAULT = DensityEstimator  # Useful in CLI
