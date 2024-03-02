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

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from jax import numpy as jnp
from typing_extensions import Self

from netobs.adaptors import NetworkAdaptor
from netobs.systems import System

X = TypeVar("X", bound=System)


class Observable(ABC, Generic[X]):
    def __init__(self, options: dict, system: X) -> None:
        self.options = options
        self.shape = self.shapeof(system)

    @abstractmethod
    def shapeof(self, system: X) -> tuple[int, ...]:
        """Get the shape of the observable.

        Args:
            system: Evaluating system

        Returns:
            The shape of the observable.
            For example, if one need the force on every atom, the shape is (natom, ndim)
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__


class Estimator(Generic[X]):
    """Create an estimator for some observable.

    Args:
        adaptor: Network adaptor.
        system: System to observe.
        estimator_options: options for the estimator.
        observable_options: Observable information.
    """

    system: X
    observable_type: type[Observable[X]]

    def __init__(
        self,
        adaptor: NetworkAdaptor[X],
        system: X,
        estimator_options: dict,
        observable_options: dict,
    ):
        self.system = system
        self.observable = self.observable_type(observable_options, system)
        self.adaptor = adaptor
        self.options = estimator_options

    @property
    def name(self) -> str:
        return f"{self.observable.name}-{self.__class__.__name__}"

    @abstractmethod
    def evaluate(
        self,
        i: int,
        params: jnp.ndarray,
        key: jnp.ndarray,
        data: jnp.ndarray,
        system: X,
        state: dict[str, Any],
        aux_data: dict[str, Any],
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        """Evaluate the observable at step `i`.

        Args:
            i: step index starting from 0.
            params: network parameters.
            key: jax random key.
            data: electron positions. Shape (ndevice, nbatch, nelectrons*ndim).
            system: system containing atomic info.
            state: auxiliary state.
            aux_data: auxiliary network data.

        Returns:
            - observable parts
                Should be a dict because many estimators can have different parts
                which will be assembled in the `digest` function.
            - updated state
        """

    def digest(
        self, all_values: dict[str, jnp.ndarray], state: dict[str, Any]
    ) -> dict[str, jnp.ndarray]:
        """Calcualte the results from the values and state.

        Args:
            all_values: observerble parts at each step.
            state: auxiliary state

        Returns:
            A dictionary containing the results.
        """
        del state
        return all_values

    def empty_val_state(
        self, steps: int
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        """Create an empty state for restore.

        Args:
            steps: Steps to run.

        Returns:
            A tuple of:
            - Dictionary of empty array as a holder of the obvervable parts of each step
            - Empty auxiliary state
        """
        dtype = self.options.get("dtype")
        empty_values = {"value": jnp.zeros((steps, *self.observable.shape), dtype)}
        return (empty_values, {})

    def tree_flatten(self) -> tuple[tuple, Self]:
        """Flatten the class as PyTree."""
        return ((), self)

    @classmethod
    def tree_unflatten(cls, aux_data: Self, children: tuple) -> Self:
        """Unflatten the class as PyTree."""
        del children
        return aux_data
