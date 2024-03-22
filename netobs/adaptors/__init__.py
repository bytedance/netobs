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

"""This file holds common types and interfaces for networks.

All networks need to implement or integrate with NetworkAdaptor class to work with this
codebase. For example,

```python
@register_pytree_node_class
class AwesomeNetAdaptor(NetworkAdaptor):
    def call_network(...):
        ...

    # And other methods ...
```

Usually, to restore and evaluate the network, one may need more arguments than those
passed through arguments of the methods. Then `__init__` is the perfect place to
initialize your adaptor with some configurations and many more. There is no restriction
to the `__init__` method, since all the functions only take an instance of adaptor.

And an instance of `AwesomeNetAdaptor` should then be passed to `evaluate_observable`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Protocol, TypeVar

import jax
from jax import numpy as jnp
from typing_extensions import Self

from netobs.helpers.grad import grad_with_system
from netobs.systems import System

X = TypeVar("X", bound=System)
D = TypeVar("D")


class WalkingStep(Protocol[D]):
    """MCMC walking step function."""

    def __call__(
        self,
        key: jnp.ndarray,
        params: jnp.ndarray,
        electrons: jnp.ndarray,
        aux_data: D,
    ) -> tuple[jnp.ndarray, D]:
        pass


class NetworkAdaptor(ABC, Generic[X]):
    """Common interface for different networks.

    Args:
        config: Anything telling the adaptor how to load network config.
            But a string is preferred because it makes life easier for command line.
        args: Command line args to modify config.
    """

    def __init__(self, config: Any, args: list[str]) -> None:
        super().__init__()

    def tree_flatten(self) -> tuple[tuple, Self]:
        """Flatten the class as PyTree.

        Since the adaptor is usually static and doesn't store any data, we just return
        empty `children` list and `self` as `aux_data`.

        Useful when JIT-ting and `pmap`-ping some methods of the adaptor class.

        Returns:
            A tuple of children (which itself is an empty tuple) and the adaptor itself.
        """
        return ((), self)

    @classmethod
    def tree_unflatten(cls, aux_data: Self, children: tuple) -> Self:
        """Unflatten the class as PyTree.

        Args:
            aux_data: the adaptor itself
            children: ignored

        Returns:
            The adaptor it self.
        """
        del children
        return aux_data

    @abstractmethod
    def restore(self, restore_option: Any) -> tuple[Any, jnp.ndarray, X, Any]:
        """Restore checkpoint with given options.

        Args:
            restore_option: anything needed for restoring the network from checkpoint.
                It's just the `options.network_restore` passed to `evaluate_observable`.

        Returns:
            Restored data:
            - params
            - system
            - aux_data
        """

    @abstractmethod
    def call_signed_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: X
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the sign and log magnitude of the wavefunction.

        If the network is complex-valued, return log phase and log psi.

        Args:
            params: network parameters.
            electrons: electron positions, shape (nelectrons*ndim), where ndim is the
                dimensionality of the system.
            system: system containing atomic info.

        Returns:
            The sign (or phase) and log magnitude of the wavefunction.
        """

    def call_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: X
    ) -> jnp.ndarray:
        """Get the log magnitude of the wavefunction.

        Neither sign nor phase factor should be included.

        Args:
            params: network parameters.
            electrons: electron positions, shape (nelectrons*ndim), where ndim is the
                dimensionality of the system.
            system: system containing atomic info.

        Returns:
            The log magnitude of the wavefunction.
        """
        return self.call_signed_network(params, electrons, system)[1]

    @abstractmethod
    def make_walking_step(
        self,
        batch_log_psi: Callable[[jnp.ndarray, jnp.ndarray, X], jnp.ndarray],
        steps: int,
        system: X,
    ) -> WalkingStep:
        """Make MCMC walking step.

        Args:
            batch_log_psi: Batched function returning log psi.
                (params, (nbatch, nelec*ndim), system) -> (nbatch,)
                We receive it as input because the importance sampling can be based on
                another psi_G instead of base psi_T.
            steps: Number of MCMC steps to run.
            system: system containing atomic info.

        Returns:
            Function with the same signature as WalkingStep
        """

    def make_burnin_step(
        self,
        batch_log_psi: Callable[[jnp.ndarray, jnp.ndarray, X], jnp.ndarray],
        steps: int,
        system: X,
    ) -> WalkingStep:
        """Make MCMC burnin step.

        By default, this is the same as make_walking_step.

        Args:
            batch_log_psi: Batched function returning log psi.
            steps: Number of MCMC steps to run.
            system: system containing atomic info.

        Returns:
            Function with the same signature as WalkingStep
        """
        return self.make_walking_step(batch_log_psi, steps, system)

    def call_local_energy(
        self, params: jnp.ndarray, key: jnp.ndarray, electrons: jnp.ndarray, system: X
    ) -> jnp.ndarray:
        """Get local energy.

        Args:
            params: Network params.
            key: JAX random keys.
            electrons: Electon positions.
            system: System containing atomic info.

        Returns:
            Local energy at this point.
        """
        key0, key1 = jax.random.split(key)
        ke = self.call_local_kinetic_energy(params, key0, electrons, system)
        pe = self.call_local_potential_energy(params, key1, electrons, system)
        return ke + pe

    @abstractmethod
    def call_local_kinetic_energy(
        self, params: jnp.ndarray, key: jnp.ndarray, electrons: jnp.ndarray, system: X
    ) -> jnp.ndarray:
        """Get local kinetic energy.

        Args:
            params: Network params.
            key: JAX random keys.
            electrons: Electon positions.
            system: System containing atomic info.

        Returns:
            Local kinetic energy at this point.
        """

    @abstractmethod
    def call_local_potential_energy(
        self, params: jnp.ndarray, key: jnp.ndarray, electrons: jnp.ndarray, system: X
    ) -> jnp.ndarray:
        """Get local potential energy.

        Args:
            params: Network params.
            key: JAX random keys.
            electrons: Electon positions.
            system: System containing atomic info.

        Returns:
            Local potential energy at this point.
        """

    def make_network_grad(self, arg: str, jaxfun: Callable = jax.grad):
        """Create gradient function of network.

        Useful when the gradient handling is different in your network.
        Note that if your network is complex-valued, you don't need to implement
        this function since `self.call_network` must be real.

        Args:
            arg: the name of arguments to calculate gradient, e.g. "electrons"/"atoms".
            jaxfun: the type of gradient to take, e.g. `jax.grad`.

        Returns:
            The gradient function.
        """
        return grad_with_system(self.call_network, arg, jaxfun=jaxfun)  # type: ignore

    def make_signed_network_grad(self, arg: str, jaxfun: Callable = jax.grad):
        """Create gradient function of network, taking sign into consideration.

        Useful when the gradient handling is different in your network,
        e.g. the network is complex-valued.

        Args:
            arg: the name of arguments to calculate gradient, e.g. "electrons"/"atoms".
            jaxfun: the type of gradient to take, e.g. `jax.grad`.

        Returns:
            The gradient function.
        """
        return grad_with_system(self.call_network, arg, jaxfun=jaxfun)  # type: ignore

    def make_local_energy_grad(self, arg: str, jaxfun: Callable = jax.grad):
        """Create gradient function of local energy.

        Useful when the gradient handling is different in your network,
        e.g. the network is complex-valued.

        Args:
            arg: the name of arguments to calculate gradient, e.g. "electrons"/"atoms".
            jaxfun: the type of gradient to take, e.g. `jax.grad`.

        Returns:
            The gradient function.
        """
        return grad_with_system(self.call_local_energy, arg, jaxfun=jaxfun)  # type: ignore


# Utility protocols
class LogPsiNet(Protocol):
    def __call__(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: System
    ) -> jnp.ndarray:
        pass
