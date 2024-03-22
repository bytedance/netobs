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
from netobs.systems import System, solid
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
        hfm_term = -self.grad_potential(params, key, data, system)
        if self.r_core > 0:
            data_mirrored, mirrored_weight = self.batch_mirror(params, data, system)
            hfm_anti = (
                -self.grad_potential(params, key, data_mirrored, system)
                * mirrored_weight[..., None, None]
            )
            hfm_term = (hfm_term + hfm_anti) / 2
        values = {"hfm_term": hfm_term}
        if self.enable_zb:
            pulay_coeff = 2 * self.batch_f_deriv_atom(params, data, system)
            el = self.batch_local_energy(params, key, data, system)
            values["el"] = el
            values["el_term"] = -el[..., None, None] * pulay_coeff
            values["pulay_coeff"] = pulay_coeff
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


def exp1(x):
    """Swamee and Ohija approximation for E1 function.

    See https://doi.org/10.1111%2Fj.1745-6584.2003.tb02608.x

    Default inplementaion in `jax.scipy.special.exp1` is accurate but slow.
    See https://github.com/google/jax/issues/13543.
    """
    return (
        jnp.log((0.56146 / x + 0.65) * (1 + x)) ** (-7.7)
        + x**4 * jnp.exp(7.7 * x) * (2 + x) ** (3.7)
    ) ** -0.13


@register_pytree_node_class
class MinAC(Estimator[System]):
    r"""AC type Zero-variance zero-bias estimator based on \tilde{\psi}_{min}.

    \tilde{\psi}_{min} is the "minimal" form removing the singular part.
    Based on R. Assaraf and M. Caffarel, J. Chem. Phys. 119, 10536 (2003).

    Eq. (73) in the paper.
    """

    observable_type = Force

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.enable_zb = self.options.get("zb", False)
        self.r_core = estimator_options.get("r_core", 0)
        if self.r_core > 0:
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
        self.grad_f = adaptor.make_network_grad("electrons")
        if "latvec" in system:
            # NOTE: Only support default Ewald settings below.
            self.dist = MinimalImageDistance(system["latvec"])
            recvec = solid.recvec(system, norm_to=1)
            self.alpha = 5.0 / jnp.amin(1 / jnp.linalg.norm(recvec, axis=1))

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
        f_bare = -self.grad_potential(params, key, data, system)
        zv_term = self.batch_zv(params, data, system)
        hfm_term = f_bare + zv_term

        if self.r_core > 0:
            data_mirrored, mirrored_weight = self.batch_mirror(params, data, system)
            f_bare_mirror = -self.grad_potential(params, key, data_mirrored, system)
            zv_mirror = self.batch_zv(params, data_mirrored, system)
            hfm_mirror = (f_bare_mirror + zv_mirror) * mirrored_weight[..., None, None]
            hfm_term = (hfm_term + hfm_mirror) / 2

        values = {"hf_term": hfm_term}

        if self.enable_zb:
            pulay_coeff = 2 * self.batch_f_deriv_atom(params, data, system)
            el = self.batch_local_energy(params, key, data, system)
            values["el"] = el
            values["el_term"] = -el[..., None, None] * pulay_coeff
            values["ev_term_coeff"] = pulay_coeff
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

    def batch_zv(self, params: jnp.ndarray, x: jnp.ndarray, system: System):
        if "latvec" in system:
            return self.batch_zv_solid(params, x, cast(SolidSystem, system))
        else:
            return self.batch_zv_molecular(params, x, cast(MolecularSystem, system))

    @partial(jax.pmap, in_axes=(None, 0, 0, None))
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def batch_zv_molecular(
        self, params: jnp.ndarray, x: jnp.ndarray, system: MolecularSystem
    ) -> jnp.ndarray:
        ae, r_ae = calculate_r_ae(x, system)
        f_ae = jnp.sum(system["charges"][..., None] * ae / r_ae**3, axis=0)
        dot_term = jnp.dot(
            self.grad_Q_molecular(x, system), self.grad_f(params, x, system)
        )
        return -f_ae + dot_term

    @partial(jax.jacfwd, argnums=1)
    def grad_Q_molecular(self, x: jnp.ndarray, system: MolecularSystem) -> jnp.ndarray:
        """The gradient of the Q matrix in the molecular case.

        Undecorated: the Q matrix in the molecular case.

        Args:
            x: Shape (nelec*ndim). Electron positions.
            system: system containing atomic info.

        Returns:
            nabla Q. Shape (natom, ndim, nelec*ndim).
            Undecorated: the Q matrix. Shape (natom, ndim).
        """
        ae, r_ae = calculate_r_ae(x, system)
        return jnp.sum(system["charges"][..., None] * ae / r_ae, axis=0)

    @partial(jax.pmap, in_axes=(None, 0, 0, None))
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def batch_zv_solid(
        self, params: jnp.ndarray, x: jnp.ndarray, system: SolidSystem
    ) -> jnp.ndarray:
        dot_term = jnp.dot(self.grad_Q_solid(x, system), self.grad_f(params, x, system))
        return -self.lap_Q_solid(x, system) / 2 - dot_term

    def grad_Q_solid(self, x: jnp.ndarray, system: SolidSystem) -> jnp.ndarray:
        r"""Calculate \nabla Q.

        shape (natom, ndim, nelec*ndim)
        """
        # shape (ncell, nelec, natom, 3), (ncell, nelec, natom, 1)
        ae, r_ae = self.dist.neighboring_r_ae(x, system["atoms"])
        # (ncell, nelec, natom, 1) -> (ncell, natom, 1, nelec, 1)
        r_ae = jnp.transpose(r_ae, (0, 2, 3, 1))[..., None]
        _, nelec, natom, ndim = ae.shape
        # (ncell, nelec, natom, ndim, ndim) -> (ncell, natom, ndim, nelec, ndim)
        quadratic_mat = jnp.transpose(ae[..., None] * ae[..., None, :], (0, 2, 3, 1, 4))

        ar = self.alpha * r_ae
        quadratic_mat *= jax.lax.erfc(ar) / r_ae**3
        diag_mat = -jnp.eye(ndim)[:, None, :] * (
            self.alpha / jnp.sqrt(jnp.pi) * -exp1(ar**2) + jax.lax.erfc(ar) / r_ae
        )
        return system["charges"][:, None, None] * jnp.reshape(
            jnp.sum(quadratic_mat + diag_mat, axis=0), (natom, ndim, nelec * ndim)
        )

    def lap_Q_solid(self, x: jnp.ndarray, system: SolidSystem) -> jnp.ndarray:
        r"""Calculate \nabla^2 Q.

        shape (natom, ndim)
        """
        # shape (ncell, nelec, natom, 3), (ncell, nelec, natom, 1)
        ae, r_ae = self.dist.neighboring_r_ae(x, system["atoms"])
        ar = self.alpha * r_ae
        coeff = self.alpha / jnp.sqrt(jnp.pi)
        return system["charges"][:, None] * jnp.sum(
            (jax.lax.erfc(ar) / r_ae - coeff * jnp.exp(-(ar**2))) * 2 * ae / r_ae**2,
            axis=(0, 1),
        )


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
        self.r_core = estimator_options.get("r_core", 0)
        if self.r_core > 0:
            self.batch_mirror = make_antithetic(
                system, adaptor.call_network, self.r_core
            )

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
        hfm_bare = -self.batch_el_deriv_atom(params, key, data, system)
        pulay_bare = 2 * self.batch_f_deriv_atom(params, data, system)
        el = self.batch_local_energy(params, key, data, system)
        if self.r_core > 0:
            m_data, weights = self.batch_mirror(params, data, system)
            m_hfm_bare = -self.batch_el_deriv_atom(params, key, m_data, system)
            weights = weights[..., None, None]
            hfm_bare = (hfm_bare + m_hfm_bare * weights) / 2
        values = {
            "hfm_bare": hfm_bare,
            "pulay_bare": pulay_bare,
            "el": el,
            "el_term_bare": -el[..., None, None] * pulay_bare,
        }
        if self.warp:
            pulay_warp = self.batch_pulay_coeff_warp(params, data, system)
            hfm_warp = self.batch_hfm_warp(params, key, data, system)
            if self.r_core > 0:
                m_hfm_warp = self.batch_hfm_warp(params, key, m_data, system)
                hfm_warp = (hfm_warp + m_hfm_warp * weights) / 2
            values["hfm_warp"] = hfm_warp
            values["pulay_warp"] = pulay_warp
            values["el_term_warp"] = -el[..., None, None] * pulay_warp

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


@register_pytree_node_class
class FastWarp(Estimator[System]):
    observable_type = Force

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.r_core = estimator_options.get("r_core", 0)
        if self.r_core != 0:
            self.batch_mirror = make_antithetic(
                system, adaptor.call_network, self.r_core
            )
        self.nelec = sum(system["spins"])
        self.grad_potential = jax.pmap(
            jax.vmap(
                grad_with_system(adaptor.call_local_potential_energy, "atoms"),
                in_axes=(None, None, 0, None),
            ),
            in_axes=(0, 0, 0, None),
        )
        self.ep_deriv_elec = grad_with_system(
            adaptor.call_local_potential_energy, "electrons", jaxfun=jax.value_and_grad
        )
        self.batch_kinetic_energy = jax.pmap(
            jax.vmap(adaptor.call_local_kinetic_energy, in_axes=(None, None, 0, None)),
            in_axes=(0, 0, 0, None),
        )
        if "latvec" in system:
            dist = MinimalImageDistance(system["latvec"])
            self.neighboring_r_ae = lambda x, s: dist.neighboring_r_ae(x, s["atoms"])[1]
            logger.info("Using periodic version of FastWarp")
        else:
            self.neighboring_r_ae = lambda x, s: calculate_r_ae(x, s)[1][None, ...]
            logger.info("Using molecular version of FastWarp")
        self.grad_f = adaptor.make_signed_network_grad("electrons")
        self.f_deriv_atom = adaptor.make_signed_network_grad("atoms")
        self.hess_f = adaptor.make_signed_network_grad("electrons", jaxfun=jax.hessian)
        self.batch_hfm_warp = jax.pmap(
            jax.vmap(self.hfm_warp_term, (None, None, 0, None)), in_axes=(0, 0, 0, None)
        )
        self.omega_hess = jax.vmap(
            jax.hessian(self.omega_single_electron, argnums=1), in_axes=(None, 0)
        )
        self.omega_jacfwd = jax.vmap(
            jax.jacfwd(self.omega_single_electron, argnums=1), in_axes=(None, 0)
        )

    def empty_val_state(self, steps: int):
        term_shape = (steps, *self.observable.shape)
        dtype = self.options.get("dtype")
        names = ("hfm_term", "pulay_bare", "pulay_warp", "el_term_bare", "el_term_warp")
        empty_values = {
            "el": jnp.zeros((steps,), dtype),
            **{name: jnp.zeros(term_shape, dtype) for name in names},
        }
        return empty_values, {}

    def evaluate(self, i, params, key, data, system, state, aux_data):
        del i, aux_data
        f_bare = -self.grad_potential(params, key, data, system)
        hfm_warp, el, pulay_bare, pulay_warp = self.batch_hfm_warp(
            params, key, data, system
        )
        if self.r_core != 0:  # Enable antithetic
            data_mirrored, mirrored_weight = self.batch_mirror(params, data, system)
            m_f_bare = -self.grad_potential(params, key, data_mirrored, system)
            m_hfm_warp, *_ = self.batch_hfm_warp(params, key, data_mirrored, system)
            mirrored_weight = mirrored_weight[..., None, None]
            f_bare = (f_bare + m_f_bare * mirrored_weight) / 2
            hfm_warp = (hfm_warp + m_hfm_warp * mirrored_weight) / 2
        values = {
            # TODO: inspect why NaN happens
            "hfm_term": f_bare + hfm_warp,
            "el": el,
            "el_term_bare": -jnp.real(jnp.conjugate(el[..., None, None]) * pulay_bare),
            "el_term_warp": -jnp.real(jnp.conjugate(el[..., None, None]) * pulay_warp),
            "pulay_bare": pulay_bare.real,
            "pulay_warp": pulay_warp.real,
        }
        return values, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state
        el_term = all_values["el_term_bare"] + all_values["el_term_warp"]
        pulay_coeff = all_values["pulay_bare"] + all_values["pulay_warp"]
        energy_mean = jnp.mean(all_values["el"])
        return {
            "force_biased": all_values["hfm_term"],
            "energy": all_values["el"],
            "force": all_values["hfm_term"] + el_term + pulay_coeff * energy_mean,
        }

    def hfm_warp_term(
        self, params: jnp.ndarray, key: jnp.ndarray, x: jnp.ndarray, system: System
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        omega_mat = self.omega(system, x)

        potential_energy, ep_grad = self.ep_deriv_elec(params, key, x, system)
        pot_term = omega_mat * jnp.reshape(ep_grad, (-1, 1, 3))

        n = self.nelec
        omega_jacfwd = self.omega_jacfwd(system, jnp.reshape(x, (n, 3)))
        grad_f = self.grad_f(params, x, system)
        hess_f = self.hess_f(params, x, system)
        kinetic_energy = -0.5 * (jnp.trace(hess_f) + jnp.sum(grad_f**2))

        grad_f = jnp.reshape(grad_f, (n, 3))
        hess_f = jnp.reshape(hess_f, (n, 3, n, 3))

        # (n, 3, 3)
        grad_grad_f_per_elec = jnp.einsum("ij,ik->ijk", grad_f, grad_f)
        # Take out 3x3 block diagonal from nx3xnx3 matrix
        hess_f_per_elec = jnp.moveaxis(jnp.diagonal(hess_f, axis1=0, axis2=2), 2, 0)
        hess_psi_per_elec = grad_grad_f_per_elec + hess_f_per_elec
        o1_term = jnp.sum(
            omega_jacfwd[..., None] * hess_psi_per_elec[:, None, ...], axis=-2
        )

        grad_f = jnp.reshape(grad_f, (n, 1, 3))
        omega_hess = self.omega_hess(system, jnp.reshape(x, (n, 3)))
        lap_omega = jnp.trace(omega_hess, axis1=-2, axis2=-1)[..., None]
        # Autograd for omega has numerical issue around nucleus,
        # but the value there is simply 0.
        lap_omega = jnp.nan_to_num(lap_omega, posinf=0.0, neginf=0.0)
        o2_term = lap_omega * grad_f / 2

        local_energy = potential_energy + kinetic_energy
        pulay_bare = 2 * self.f_deriv_atom(params, x, system)
        pulay_warp = 2 * jnp.sum(omega_mat * grad_f + omega_jacfwd / 2, axis=0)

        hfm_warp_term = -jnp.sum(pot_term + o1_term + o2_term, axis=0)
        return hfm_warp_term, local_energy, pulay_bare, pulay_warp

    def omega(self, system: System, x: jnp.ndarray) -> jnp.ndarray:
        r_ae = self.neighboring_r_ae(x, system)
        # Remind r_ae is in shape (ncell, nelectron, natom, 1)
        f_mat = self.decay_function(r_ae)
        return jnp.sum(f_mat, axis=0) / f_mat.sum(axis=(0, 2))[:, None, :]

    def omega_single_electron(self, system: System, x: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate \omega matrix by single electron postion.

        Args:
            system: system containing atomic info.
            x: single electron position. Shape: (ndim,)

        Returns:
            Derivative of \omega matrix. Shape: (natom,)
        """
        r_ae = self.neighboring_r_ae(x, system)
        # r_ae has shape (ncell, 1, ntom, 1)
        f_mat = self.decay_function(r_ae[:, 0, :, 0])  # shape (ncell, natom)
        return jnp.sum(f_mat, axis=0) / f_mat.sum(axis=(0, 1))

    @staticmethod
    def decay_function(r_ae: jnp.ndarray) -> jnp.ndarray:
        return 1 / r_ae**4
