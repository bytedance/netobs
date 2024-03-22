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

from functools import partial, wraps
from typing import Callable, cast

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netobs.adaptors import LogPsiNet, NetworkAdaptor
from netobs.helpers.chunk_vmap import chunk_vmap
from netobs.helpers.digest import robust_mean
from netobs.helpers.grad import grad_with_system
from netobs.logging import logger
from netobs.observables import Estimator, Observable
from netobs.systems import System
from netobs.systems.molecule import MolecularSystem, calculate_r_ae
from netobs.systems.solid import MinimalImageDistance, SolidSystem


class Force(Observable):
    def shapeof(self, system) -> tuple[int, ...]:
        return (system["atoms"].shape[0], system["ndim"])


def make_antithetic(system: System, call_network: LogPsiNet, r_core: float = 0.5):
    if "latvec" in system:
        system = cast(SolidSystem, system)
        dist = MinimalImageDistance(system["latvec"])
        logger.info("Using periodic version of Antithetic")

        def get_closest_r_ae(electrons, system):
            return dist.closest_r_ae(electrons, system["atoms"])
    else:
        logger.info("Using molecular version of Antithetic")
        get_closest_r_ae = calculate_r_ae

    @partial(jax.pmap, in_axes=(0, 0, None))
    @partial(jax.vmap, in_axes=(None, 0, None))
    def batch_mirror(params: jnp.ndarray, x: jnp.ndarray, system: System):
        ae, r_ae = get_closest_r_ae(x, system)
        idx_closest_atom = jnp.argmin(r_ae, axis=1)
        closest_r_ae = jnp.min(r_ae, axis=1, keepdims=True)
        # Shape (nelectron, natom, 3)
        closest_ae = jax.vmap(lambda x, ind: x[ind])(ae, idx_closest_atom)
        is_core_electron = closest_r_ae < r_core
        electrons = x.reshape((-1, 1, 3))
        electrons_mirrored = jnp.where(
            is_core_electron, electrons - 2 * closest_ae, electrons
        )
        x_mirrored = jnp.reshape(electrons_mirrored, (-1,))
        mirrored_weight = jnp.exp(
            (call_network(params, x_mirrored, system) - call_network(params, x, system))
            * 2
        )
        return x_mirrored, mirrored_weight

    return batch_mirror


class Bare(Estimator[System]):
    """The naive estimator."""

    observable_type = Force

    def __init__(
        self,
        adaptor: NetworkAdaptor[System],
        system: System,
        estimator_options: dict,
        observable_options: dict,
    ):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.enable_zb = estimator_options.get("zb", False)
        self.r_core = estimator_options.get("r_core", 0)
        if self.r_core > 0:
            logger.info("Antithetic enabled")
            self.batch_mirror = make_antithetic(
                system, adaptor.call_network, self.r_core
            )
        self.grad_potential = jax.pmap(
            jax.vmap(
                grad_with_system(adaptor.call_local_potential_energy, "atoms"),
                in_axes=(None, None, 0, None),
            ),
            in_axes=(0, 0, 0, None),
        )
        self.batch_local_energy = jax.pmap(
            jax.vmap(adaptor.call_local_energy, in_axes=(None, None, 0, None)),
            in_axes=(0, 0, 0, None),
        )
        self.batch_f_deriv_atom = jax.pmap(
            jax.vmap(adaptor.make_network_grad("atoms"), in_axes=(None, 0, None)),
            in_axes=(0, 0, None),
        )

    def empty_val_state(self, steps: int):
        term_shape = (steps, *self.observable.shape)
        dtype = self.options.get("dtype")
        empty_values = {
            "hfm_term": jnp.zeros(term_shape, dtype),
        }
        if self.enable_zb:
            empty_values.update(
                {
                    "el": jnp.zeros(steps, dtype),
                    "el_term": jnp.zeros(term_shape, dtype),
                    "pulay_coeff": jnp.zeros(term_shape, dtype),
                }
            )
        return empty_values, {}

    def evaluate(self, i, params, key, data, system, state, aux_data):
        del i, aux_data
        hfm_term = -jnp.mean(
            self.grad_potential(params, key, data, system), axis=(0, 1)
        )
        if self.r_core > 0:
            data_mirrored, mirrored_weight = self.batch_mirror(params, data, system)
            hfm_anti = -jnp.mean(
                self.grad_potential(params, key, data_mirrored, system)
                * mirrored_weight[..., None, None],
                axis=(0, 1),
            )
            hfm_term = (hfm_term + hfm_anti) / 2
        values = {"hfm_term": hfm_term}
        if self.enable_zb:
            pulay_coeff = 2 * self.batch_f_deriv_atom(params, data, system)
            el = self.batch_local_energy(params, key, data, system)
            values["el"] = jnp.mean(el)
            values["el_term"] = jnp.mean(
                -el[..., None, None] * pulay_coeff, axis=(0, 1)
            )
            values["pulay_coeff"] = jnp.mean(pulay_coeff, axis=(0, 1))
        return values, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state
        values = {}
        hf_term = all_values["hfm_term"]
        if self.enable_zb:
            energy_mean = robust_mean(all_values["el"])
            values["energy"] = all_values["el"]
            # Don't use mean of ev_term_coeff, because it will increase fluctuations
            product = all_values["pulay_coeff"] * energy_mean
            values["force_biased"] = hf_term
            values["force"] = hf_term + all_values["el_term"] + product
        else:
            values["force"] = hf_term
        return values


@register_pytree_node_class
class AC(Estimator[MolecularSystem]):
    r"""AC type Zero-variance zero-bias estimator based on \tilde{\psi}_{min}.

    \tilde{\psi}_{min} is the "minimal" form removing the singular part.
    Based on R. Assaraf and M. Caffarel, J. Chem. Phys. 119, 10536 (2003).

    Eq. (73) in the paper.
    """

    observable_type = Force

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.enable_zb = self.options.get("zb", False)
        self.batch_local_energy = jax.pmap(
            jax.vmap(adaptor.call_local_energy, in_axes=(None, None, 0, None)),
            in_axes=(0, 0, 0, None),
        )
        self.batch_Q = jax.pmap(jax.vmap(self.Q, in_axes=(0, None)), in_axes=(0, None))
        self.grad_Q = jax.jacfwd(self.Q, argnums=0)
        self.grad_f = adaptor.make_network_grad("electrons")

    def empty_val_state(self, steps: int):
        term_shape = (steps, *self.observable.shape)
        dtype = self.options.get("dtype")
        empty_values = {
            "hf_term": jnp.zeros(term_shape, dtype),
        }
        if self.enable_zb:
            empty_values.update(
                {
                    "el": jnp.zeros(steps, dtype),
                    "el_term": jnp.zeros(term_shape, dtype),
                    "ev_term_coeff": jnp.zeros(term_shape, dtype),
                }
            )
        return empty_values, {}

    def evaluate(self, i, params, key, data, system, state, aux_data):
        del i, aux_data
        values = {"hf_term": jnp.mean(self.batch_zv(params, data, system), axis=(0, 1))}
        if self.enable_zb:
            pulay_coeff = 2 * self.batch_Q(data, system)
            el = self.batch_local_energy(params, key, data, system)
            values["el"] = jnp.mean(el)
            values["el_term"] = jnp.mean(
                -el[..., None, None] * pulay_coeff, axis=(0, 1)
            )
            values["ev_term_coeff"] = jnp.mean(pulay_coeff, axis=(0, 1))
        return values, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state
        values = {}
        if self.enable_zb:
            energy_mean = robust_mean(all_values["el"])
            values["energy"] = all_values["el"]
            # Don't use mean of ev_term_coeff, because it will increase fluctuations
            product = all_values["ev_term_coeff"] * energy_mean
            values["force_zv"] = all_values["hf_term"]
            values["force"] = all_values["hf_term"] + all_values["el_term"] + product
        else:
            values["force"] = all_values["hf_term"]
        return values

    @partial(jax.pmap, in_axes=(None, 0, 0, None))
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def batch_zv(
        self,
        params: jnp.ndarray,
        x: jnp.ndarray,
        system: MolecularSystem,
    ) -> jnp.ndarray:
        atoms, charges = system["atoms"], system["charges"]
        aa = atoms[None, ...] - atoms[:, None]
        # || (1, natom, ndim) - (natom, 1, ndim) || = (natom, natom)
        r_aa = jnp.linalg.norm(aa, axis=-1)
        # f_aa_matrix[0, 1] points from atom 0 to atom 1, so its force on atom 1
        # Shapes are: charges (natom); aa (natom, natom, 3); r_aa (natom, natom, 1)
        f_aa_matrix = jnp.nan_to_num(
            (charges[None, ..., None] * charges[..., None, None])
            * aa
            / r_aa[..., None] ** 3
        )
        f_aa = jnp.sum(f_aa_matrix, axis=0)
        dot_term = jnp.dot(self.grad_Q(x, system), self.grad_f(params, x, system))
        return f_aa + dot_term

    def Q(self, x: jnp.ndarray, system: MolecularSystem) -> jnp.ndarray:
        """The Q matrix. Shape (natom, ndim).

        Based on Eq. (70) in the paper.

        Args:
            x: Shape (nelec*ndim). Electron positions.
            system: system containing atomic info.

        Returns:
            The Q matrix.
        """
        ae, r_ae = calculate_r_ae(x, system)
        return jnp.sum(system["charges"][..., None] * ae / r_ae, axis=0)


def elec_reshaped(f):
    @wraps(f)
    def reshaped_f(*args):
        return jnp.reshape(f(*args), (-1, 1, 3))

    return reshaped_f


class SWCT(Estimator[System]):
    """Space warp coordinate transformation estimator.

    Eq. (14) of S. Sorella and L. Capriotti, J. Chem. Phys. 133, 234111 (2010).
    """

    observable_type = Force

    neighboring_r_ae: Callable[[jnp.ndarray, System], jnp.ndarray]
    "Dynammic function for calculating r_ae in molecular and solid systems."

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.warp = bool(self.options.get("warp", True))

        if "latvec" in system:
            dist = MinimalImageDistance(system["latvec"])
            self.neighboring_r_ae = lambda x, s: dist.neighboring_r_ae(x, s["atoms"])[1]
            logger.info("Using periodic version of SWCT")
        else:
            self.neighboring_r_ae = lambda x, s: calculate_r_ae(x, s)[1][None, ...]
            logger.info("Using molecular version of SWCT")

        # Remind x is in shape (nelectrons*ndim,)
        self.f_deriv_elec = elec_reshaped(adaptor.make_network_grad("electrons"))
        # Remind adaptor.call_network returns the log magnitude of the wavefunction
        self.el_deriv_elec = elec_reshaped(adaptor.make_local_energy_grad("electrons"))
        chunks = self.options.get("split_chunks")
        self.batch_local_energy = jax.pmap(
            jax.vmap(adaptor.call_local_energy, in_axes=(None, None, 0, None)),
            in_axes=(0, 0, 0, None),
        )
        self.batch_f_deriv_atom = jax.pmap(
            jax.vmap(adaptor.make_network_grad("atoms"), in_axes=(None, 0, None)),
            in_axes=(0, 0, None),
        )
        # Only `el_deriv_something` needs to be chunked
        self.batch_el_deriv_atom = jax.pmap(
            chunk_vmap(
                adaptor.make_local_energy_grad("atoms"), (None, None, 0, None), chunks
            ),
            in_axes=(0, 0, 0, None),
        )
        self.batch_hfm_warp = jax.pmap(
            chunk_vmap(self.hfm_warp_term, (None, None, 0, None), chunks),
            in_axes=(0, 0, 0, None),
        )
        self.batch_pulay_coeff_warp = jax.pmap(
            jax.vmap(self.pulay_coeff_warp_term, in_axes=(None, 0, None)),
            in_axes=(0, 0, None),
        )

    def empty_val_state(self, steps: int):
        term_shape = (steps, *self.observable.shape)
        dtype = self.options.get("dtype")
        values = ["hfm_bare", "pulay_bare", "el_term_bare"]
        if self.warp:
            values += ["hfm_warp", "pulay_warp", "el_term_warp"]
        empty_values = {name: jnp.zeros(term_shape, dtype) for name in values}
        empty_values["el"] = jnp.zeros(steps, dtype)
        return empty_values, {}

    def evaluate(self, i, params, key, data, system, state, aux_data):
        del i, aux_data
        hfm_bare = -jnp.mean(
            self.batch_el_deriv_atom(params, key, data, system), axis=(0, 1)
        )
        pulay_bare = 2 * self.batch_f_deriv_atom(params, data, system)
        el = self.batch_local_energy(params, key, data, system)
        values = {
            "hfm_bare": hfm_bare,
            "pulay_bare": jnp.mean(pulay_bare, axis=(0, 1)),
            "el": jnp.mean(el),
            "el_term_bare": jnp.mean(-el[..., None, None] * pulay_bare, axis=(0, 1)),
        }
        if self.warp:
            pulay_warp = self.batch_pulay_coeff_warp(params, data, system)
            values["hfm_warp"] = jnp.mean(
                self.batch_hfm_warp(params, key, data, system), axis=(0, 1)
            )
            values["pulay_warp"] = jnp.mean(pulay_warp, axis=(0, 1))
            values["el_term_warp"] = jnp.mean(
                -el[..., None, None] * pulay_warp, axis=(0, 1)
            )
        return values, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state

        energy_mean = robust_mean(all_values["el"])
        pulay_coeff = all_values["pulay_bare"]
        hfm_term = all_values["hfm_bare"]
        el_term = all_values["el_term_bare"]

        force = hfm_term + el_term + pulay_coeff * energy_mean
        values = {"energy": all_values["el"], "force": force}

        if self.warp:
            pulay_coeff += all_values["pulay_warp"]
            hfm_term += all_values["hfm_warp"]
            el_term += all_values["el_term_warp"]
            values["force_no_warp"] = values["force"]
            values["force"] = hfm_term + el_term + pulay_coeff * energy_mean

        return values

    def hfm_warp_term(
        self, params: jnp.ndarray, key: jnp.ndarray, x: jnp.ndarray, system: System
    ) -> jnp.ndarray:
        omega_mat = self.omega(system, x)
        return -jnp.sum(omega_mat * self.el_deriv_elec(params, key, x, system), axis=0)

    def pulay_coeff_warp_term(
        self, params: jnp.ndarray, x: jnp.ndarray, system: System
    ) -> jnp.ndarray:
        omega_mat = self.omega(system, x)
        omega_grad = self.omega_jacfwd(system, jnp.reshape(x, (-1, 3)))
        return 2 * jnp.sum(
            omega_mat * self.f_deriv_elec(params, x, system) + omega_grad / 2, axis=0
        )

    def omega(self, system: System, x: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate the \omega matrix.

        Args:
            system: system containing atomic info.
            x: electron positions. Shape (nelectrons*ndim,).

        Returns:
            \omega matrix, shape (nelectron, natom, 1)
        """
        r_ae = self.neighboring_r_ae(x, system)
        # Remind r_ae is in shape (ncell, nelectron, natom, 1)
        f_mat = self.decay_function(r_ae)
        return jnp.sum(f_mat, axis=0) / f_mat.sum(axis=(0, 2))[:, None, :]

    @partial(jax.vmap, in_axes=(None, None, 0))
    @partial(jax.jacfwd, argnums=2)
    def omega_jacfwd(self, system: System, x: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate the derivative for \omega matrix by electron postion.

        Args:
            system: system containing atomic info.
            x: single electron position. Shape (nelctron, ndim).
                Shape for undecorated: (ndim,)

        Returns:
            Derivative of \omega matrix. Shape (nelctron, natom, ndim).
                Shape for undecorated: (natom,)
        """
        r_ae = self.neighboring_r_ae(x, system)
        # r_ae has shape (ncell, 1, ntom, 1)
        f_mat = self.decay_function(r_ae[:, 0, :, 0])  # shape (ncell, natom)
        return jnp.sum(f_mat, axis=0) / f_mat.sum(axis=(0, 1))

    @staticmethod
    def decay_function(r_ae: jnp.ndarray) -> jnp.ndarray:
        """The fast decaying function, aka F.

        Args:
            r_ae: Relative distances of the electrons.

        Returns:
            The value of the decaying function.
        """
        return 1 / r_ae**4
